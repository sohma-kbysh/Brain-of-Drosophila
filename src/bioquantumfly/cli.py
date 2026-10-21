"""BioQuantumFly command-line interface (spec section 26)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from bioquantumfly import __version__
from bioquantumfly.config import Config, load_config, save_config

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Simulation of biologically hosted ephemeral quantum cryptography on "
         "the MaleCNS connectome. Research PoC; see README for what is NOT claimed.",
)
console = Console()


def _resolve(config_path: Optional[Path], **overrides) -> Config:
    cfg = load_config(config_path)
    for key, value in overrides.items():
        if value is None:
            continue
        section, _, field = key.partition(".")
        if field:
            setattr(getattr(cfg, section), field, value)
        else:
            setattr(cfg, key, value)
    return cfg


def _profile_for(cfg: Config, workers: int | None):
    from bioquantumfly.runtime.profiles import load_warnings, resolve_profile
    from bioquantumfly.runtime.resources import detect_resources

    resources = detect_resources(Path.cwd())
    profile = resolve_profile(
        cfg.runtime.profile,
        workers=workers if workers is not None else cfg.runtime.workers,
        blas_threads=cfg.runtime.blas_threads,
        max_memory_gb=cfg.runtime.max_memory_gb,
        resources=resources,
    )
    for warning in load_warnings(resources, profile):
        console.print(f"[yellow]warning:[/yellow] {warning}")
    return resources, profile


def _run_experiment(
    name: str, cfg: Config, workers: int | None, resume: bool, tag: str | None = None
) -> Path:
    from bioquantumfly.experiments.registry import get_experiment, make_runner
    from bioquantumfly.runtime.provenance import build_metadata, results_dir, write_metadata
    from bioquantumfly.runtime.sweep import run_sweep

    definition = get_experiment(name)
    resources, profile = _profile_for(cfg, workers)
    specs = definition.build_specs(cfg)
    outdir = results_dir(cfg.results_dir, tag or name)

    console.print(
        f"[bold]{name}[/bold] ({definition.spec_number}) - {len(specs)} runs, "
        f"{profile.workers} workers, {profile.blas_threads} BLAS thread(s)"
    )
    if definition.speculative:
        console.print("[red]This experiment uses the SPECULATIVE quantum-to-neural "
                      "coupling. Results are exploratory.[/red]")

    start = time.perf_counter()
    with console.status("running..."):
        df = run_sweep(
            specs, make_runner(name, cfg), outdir,
            workers=profile.workers, blas_threads=profile.blas_threads, resume=resume,
        )
    elapsed = time.perf_counter() - start

    save_config(cfg, outdir / "config.yaml")
    if not df.empty:
        df.to_csv(outdir / "metrics.csv", index=False)
    write_metadata(outdir, build_metadata(
        experiment=name, config=cfg.to_dict(), seed=cfg.runtime.seed,
        resources=resources, malecns_version=cfg.connectome.dataset,
        physical_model=cfg.quantum.physical_regime,
        speculative_coupling_enabled=cfg.coupling.enabled,
        extra={"n_runs": len(specs), "n_rows": len(df),
               "wall_seconds": elapsed, "profile": profile.to_dict()},
    ))
    console.print(f"[green]done[/green] {len(df)} rows in {elapsed:.1f}s -> {outdir}")
    return outdir


# ---------------------------------------------------------------------------
@app.command()
def version() -> None:
    """Print the package version."""
    console.print(f"bioquantumfly {__version__}")


@app.command("check-resources")
def check_resources() -> None:
    """Report the detected execution environment (addendum sections 2, 3)."""
    from bioquantumfly.connectome.male_cns import token_is_present
    from bioquantumfly.runtime.resources import detect_resources

    r = detect_resources(Path.cwd())
    table = Table(show_header=False, box=None)
    table.add_row("hostname", r.hostname)
    table.add_row("cpu", f"{r.cpu_model or 'unknown'}")
    table.add_row("logical CPUs", f"{r.logical_cpu_count} (usable {r.usable_cpu_count})")
    table.add_row("memory", f"{r.total_memory_gb:.0f} GB total, "
                            f"{r.available_memory_gb:.0f} GB available"
                  if r.total_memory_gb else "unknown")
    table.add_row("load average", str(r.load_average))
    table.add_row("NUMA nodes", f"{r.numa_nodes} (numactl {'yes' if r.numa_available else 'no'})")
    table.add_row("scheduler", f"{r.scheduler.kind or 'none'} "
                               f"{'(inside allocation)' if r.scheduler.inside_allocation else ''}")
    for label, fs in (("project fs", r.project_filesystem),
                      ("temp fs", r.temporary_filesystem),
                      ("/dev/shm", r.shm_filesystem)):
        if fs:
            flag = " [red]ROOT FILESYSTEM[/red]" if fs.is_root_filesystem else ""
            table.add_row(label, f"{fs.mountpoint} - {fs.available_gb:.0f} GB free{flag}")
    table.add_row("NEUPRINT_TOKEN", "present" if token_is_present() else "not set")
    console.print(table)
    if r.is_heavily_loaded:
        console.print("[yellow]machine is heavily loaded; prefer a small profile[/yellow]")


@app.command("fetch-subgraph")
def fetch_subgraph_cmd(
    seed_type: Optional[str] = typer.Option(None, help="neuron type to seed from"),
    seed_body_id: list[int] = typer.Option([], help="explicit body ids to seed from"),
    hops: int = typer.Option(2, help="k-hop neighbourhood radius"),
    min_weight: int = typer.Option(5, help="minimum synapse count per connection"),
    max_neurons: int = typer.Option(20_000, help="guard against an over-broad seed"),
    dataset: str = typer.Option("male-cns:v1.0"),
    output: Path = typer.Option(Path("data/processed/male_cns"), help="output directory"),
) -> None:
    """Fetch a MaleCNS k-hop subgraph from neuPrint and cache it.

    Requires NEUPRINT_TOKEN. The full connectivity table is never downloaded.
    """
    from bioquantumfly.connectome.graph import save_processed
    from bioquantumfly.connectome.male_cns import (
        NeuPrintConfig, TokenMissingError, fetch_subgraph,
    )
    from bioquantumfly.runtime.resources import check_disk_space

    if not seed_type and not seed_body_id:
        raise typer.BadParameter("provide --seed-type or at least one --seed-body-id")
    check_disk_space(output.parent if output.parent.exists() else Path.cwd(),
                     estimated_gb=1.0, raise_on_fail=False)
    try:
        graph = fetch_subgraph(
            seed_type=seed_type, seed_body_ids=seed_body_id or None,
            config=NeuPrintConfig(dataset=dataset, hops=hops, min_weight=min_weight),
            max_neurons=max_neurons,
        )
    except TokenMissingError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    save_processed(graph, output)
    console.print(json.dumps(graph.summary(), indent=2))
    console.print(f"[green]saved[/green] -> {output}")


@app.command("preprocess-malecns")
def preprocess_cmd(
    input_dir: Path = typer.Option(..., help="directory with neurons.parquet + edges.parquet"),
    output: Path = typer.Option(Path("data/processed/male_cns")),
    min_weight: int = typer.Option(5),
) -> None:
    """Convert a cached neuPrint result into the optimised local representation."""
    import pandas as pd

    from bioquantumfly.connectome.graph import build_graph, save_processed

    neurons = pd.read_parquet(input_dir / "neurons.parquet")
    edges = pd.read_parquet(input_dir / "edges.parquet")
    query = input_dir / "query.json"
    provenance = json.loads(query.read_text()) if query.exists() else {}
    graph = build_graph(neurons, edges, provenance=provenance, min_weight=min_weight)
    save_processed(graph, output)
    console.print(json.dumps(graph.summary(), indent=2))


@app.command("simulate-quantum")
def simulate_quantum(
    config: Optional[Path] = typer.Option(None, "--config"),
    workers: Optional[int] = typer.Option(None),
    resume: bool = typer.Option(True),
) -> None:
    """Experiments 1 and 5: memory decay and host self-erasure."""
    _run_experiment("quantum_memory_decay", _resolve(config), workers, resume)


@app.command("simulate-eve")
def simulate_eve(
    config: Optional[Path] = typer.Option(None, "--config"),
    experiment: str = typer.Option(
        "eve_attack_sweep",
        help="eve_attack_sweep | information_disturbance | eve_ancilla_persistence "
             "| swap_negative_control",
    ),
    workers: Optional[int] = typer.Option(None),
    resume: bool = typer.Option(True),
) -> None:
    """Experiments 2, 3, 4, 6 and 7: the adversary."""
    _run_experiment(experiment, _resolve(config), workers, resume)


@app.command("simulate-neural")
def simulate_neural(
    config: Optional[Path] = typer.Option(None, "--config"),
    workers: Optional[int] = typer.Option(None),
    resume: bool = typer.Option(True),
) -> None:
    """Experiment 8: connectome LIF baseline, quantum coupling OFF."""
    _run_experiment("neural_baseline", _resolve(config), workers, resume)


@app.command("simulate-hybrid")
def simulate_hybrid(
    config: Optional[Path] = typer.Option(None, "--config"),
    workers: Optional[int] = typer.Option(None),
    resume: bool = typer.Option(True),
) -> None:
    """Experiment 9: SPECULATIVE quantum-to-neural tamper propagation."""
    _run_experiment("hybrid_tamper_propagation", _resolve(config), workers, resume)


@app.command("host-placement")
def host_placement_cmd(
    config: Optional[Path] = typer.Option(None, "--config"),
    workers: Optional[int] = typer.Option(None),
    resume: bool = typer.Option(True),
) -> None:
    """Where would a microscopic perturbation be most amplified? (addendum 28)"""
    _run_experiment("host_placement", _resolve(config), workers, resume)


@app.command()
def sweep(
    config: Optional[Path] = typer.Option(None, "--config"),
    experiment: str = typer.Option("feasibility_map"),
    workers: Optional[int] = typer.Option(None),
    resume: bool = typer.Option(True, help="skip runs already marked complete"),
) -> None:
    """Run a resumable parameter sweep."""
    _run_experiment(experiment, _resolve(config), workers, resume)


@app.command()
def figures(
    results: Optional[Path] = typer.Option(None, help="results root to read from"),
    output: Path = typer.Option(Path("results/figures")),
) -> None:
    """Regenerate figures from the newest results of each experiment."""
    from bioquantumfly.analysis.report import generate_figures

    made = generate_figures(Path(results or "results"), output)
    for p in made:
        console.print(f"[green]figure[/green] {p}")


@app.command()
def reproduce(
    what: str = typer.Argument("all", help="'all' or an experiment name"),
    config: Optional[Path] = typer.Option(None, "--config"),
    workers: Optional[int] = typer.Option(None),
    quick: bool = typer.Option(False, help="smaller grids, for a fast end-to-end check"),
) -> None:
    """Run the main experiments and regenerate every figure."""
    from bioquantumfly.analysis.report import generate_figures, write_initial_report
    from bioquantumfly.experiments.registry import EXPERIMENTS

    cfg = _resolve(config)
    if quick:
        cfg.quantum.T2_seconds = [4e-9, 1e-6, 1e-3]
        cfg.quantum.n_carriers = 16
        cfg.connectome.synthetic_n_neurons = 300
        cfg.host_selection.n_hosts = 8
        cfg.neural.duration_s = 0.15
        cfg.coupling.gain_sweep = [0.0, 0.1, 1.0]

    names = list(EXPERIMENTS) if what == "all" else [what]
    for name in names:
        _run_experiment(name, cfg, workers, resume=True)

    out = Path(cfg.results_dir) / "figures"
    for p in generate_figures(Path(cfg.results_dir), out):
        console.print(f"[green]figure[/green] {p}")
    report = write_initial_report(Path(cfg.results_dir), cfg)
    console.print(f"[green]report[/green] {report}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
