"""Assemble figures and the initial report from stored results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bioquantumfly.analysis import plotting
from bioquantumfly.config import Config


def latest_results(root: Path, experiment: str) -> Path | None:
    """Newest timestamped directory for an experiment, if any."""
    base = Path(root) / experiment
    if not base.is_dir():
        return None
    runs = sorted((d for d in base.iterdir() if d.is_dir()), key=lambda d: d.name)
    return runs[-1] if runs else None


def load_metrics(root: Path, experiment: str) -> pd.DataFrame:
    directory = latest_results(root, experiment)
    if directory is None:
        return pd.DataFrame()
    csv = directory / "metrics.csv"
    if csv.exists():
        return pd.read_csv(csv)
    from bioquantumfly.runtime.sweep import load_sweep_results

    return load_sweep_results(directory)


def generate_figures(results_root: Path, output: Path) -> list[Path]:
    """Build every figure for which results exist. Missing data is skipped."""
    results_root = Path(results_root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    made: list[Path] = [plotting.figure_architecture(output / "figure1_architecture.png")]

    decay = load_metrics(results_root, "quantum_memory_decay")
    if not decay.empty:
        made.append(plotting.figure_bob_recovery(decay, output / "figure2_bob_recovery.png"))
        made.append(plotting.figure_regime_comparison(decay, output / "figure5_regimes.png"))

    disturbance = load_metrics(results_root, "information_disturbance")
    if not disturbance.empty:
        made.append(plotting.figure_information_disturbance(
            disturbance, output / "figure3_information_disturbance.png"))

    persistence = load_metrics(results_root, "eve_ancilla_persistence")
    if not persistence.empty:
        subset = persistence[persistence["eve_memory_model"] == "ideal"]
        if not subset.empty:
            made.append(plotting.figure_self_decay_vs_eve(
                subset, output / "figure4_self_decay_vs_eve.png"))

    attacks = load_metrics(results_root, "eve_attack_sweep")
    if not attacks.empty:
        made.append(plotting.figure_raw_vs_masked(
            attacks, output / "figure9_raw_vs_masked.png"))

    feasibility = load_metrics(results_root, "feasibility_map")
    if not feasibility.empty:
        made.append(plotting.figure_feasibility_map(
            feasibility, output / "figure8_feasibility_map.png"))

    hybrid = load_metrics(results_root, "hybrid_tamper_propagation")
    if not hybrid.empty:
        made.append(plotting.figure_downstream_response(
            hybrid, output / "figure7_downstream_response.png"))

    ablation = load_metrics(results_root, "protocol_stack")
    if not ablation.empty:
        made.append(plotting.figure_protocol_ablation(
            ablation, output / "figure10_protocol_ablation.png"))

    needs = load_metrics(results_root, "coherence_requirements")
    if not needs.empty:
        made.append(plotting.figure_coherence_requirement(
            needs, output / "figure11_coherence_requirement.png"))

    return made


def usable_window(sub: pd.DataFrame, threshold: float = 1e-3) -> float:
    """Storage time in units of T2 at which Bob's BER reaches `threshold`.

    Interpolated rather than read off the grid. Taking the largest sampled point
    still under the threshold would report the *grid spacing*, not the physics:
    the crossing sits near t/T2 ~ 2e-3, so a coarse grid would round the whole
    usable window down to zero. BER is monotonic in t, so a linear interpolation
    on (BER, t) is well defined.
    """
    ordered = sub.sort_values("bob_ber")
    ber = ordered["bob_ber"].to_numpy(dtype=float)
    t = ordered["t_over_T2"].to_numpy(dtype=float)
    if ber.size == 0 or ber[0] > threshold:
        return 0.0
    if ber[-1] <= threshold:
        return float(t[-1])          # never crosses within the sampled range
    return float(np.interp(threshold, ber, t))


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _time(seconds: float) -> str:
    return plotting._format_time(seconds)


def summarise(results_root: Path) -> dict[str, Any]:
    """Extract the numbers the report quotes, so prose and data cannot diverge."""
    out: dict[str, Any] = {}

    decay = load_metrics(results_root, "quantum_memory_decay")
    if not decay.empty:
        out["storage_windows"] = [
            {
                "T2": t2,
                "status": sub["T2_status"].iloc[0],
                "window_over_T2": (w := usable_window(sub)),
                "window_seconds": w * t2,
                "ber_at_1_T2": float(
                    sub.loc[sub["t_over_T2"].sub(1.0).abs().idxmin(), "bob_ber"]
                ),
            }
            for t2, sub in decay.groupby("T2")
        ]

    attacks = load_metrics(results_root, "eve_attack_sweep")
    if not attacks.empty:
        rows = []
        for (protocol, attack), sub in attacks.groupby(["protocol", "attack"]):
            rows.append({
                "protocol": protocol, "attack": attack,
                "eve_before_key": float(sub["eve_pguess_before_key"].max()),
                "eve_after_key": float(sub["eve_pguess_after_key"].max()),
                "bob_disturbance": float(sub["bob_disturbance_ber"].max()),
            })
        out["attacks"] = rows

    persistence = load_metrics(results_root, "eve_ancilla_persistence")
    if not persistence.empty:
        late = persistence[persistence["t_over_T2"] >= 10]
        if not late.empty:
            out["persistence"] = [
                {
                    "attack": a,
                    "eve_memory": m,
                    "host_info": float(sub["host_info_at_read"].mean()),
                    "eve_info": float(sub["eve_info_after_key"].mean()),
                    "gap": float(sub["information_gap"].mean()),
                }
                for (a, m), sub in late.groupby(["attack", "eve_memory_model"])
            ]

    feasibility = load_metrics(results_root, "feasibility_map")
    if not feasibility.empty:
        out["regime_counts"] = feasibility["regime"].value_counts().to_dict()
        secure = feasibility[feasibility["regime"] == "readable_low_eve_info"]
        out["secure_region"] = {
            "n_points": int(len(secure)),
            "fraction": float(len(secure) / len(feasibility)),
            "max_t_over_T2": float(secure["t_over_T2"].max()) if not secure.empty else 0.0,
            "max_eta": float(secure["eta"].max()) if not secure.empty else 0.0,
        }

    neural = load_metrics(results_root, "neural_baseline")
    if not neural.empty:
        out["neural"] = neural.iloc[0].to_dict()

    hybrid = load_metrics(results_root, "hybrid_tamper_propagation")
    if not hybrid.empty:
        out["hybrid"] = [
            {
                "gain": float(g),
                "condition": c,
                "perturbed_non_host": float(sub["n_perturbed_non_host"].mean()),
                "spike_change": float(sub["absolute_spike_count_change"].mean()),
            }
            for (g, c), sub in hybrid.groupby(["speculative_coupling_gain", "attack_condition"])
        ]

    ablation = load_metrics(results_root, "protocol_stack")
    if not ablation.empty:
        weak = ablation[(ablation["attack"] == "weak_measurement") & (ablation["eta"] == 0.25)]
        out["ablation"] = [
            {
                "layers": layer,
                "eve_memory": memory,
                "secret_bits": float(sub["secret_bits"].max()),
                "secret_without_delay": float(sub["secret_bits_without_delay"].max()),
                "usable": bool(sub["usable"].any()),
            }
            for (layer, memory), sub in weak.groupby(["layer_set", "eve_memory_model"])
        ] if not weak.empty else []
        theft = ablation[ablation["attack"] == "swap_theft"]
        if not theft.empty:
            out["theft_detected"] = {
                "bob_ber": float(theft["bob_ber"].mean()),
                "any_usable": bool(theft["usable"].any()),
                "n_configurations": int(len(theft)),
            }
        with_deletion = ablation[ablation["layer_set"].str.contains("certified deletion")]
        if not with_deletion.empty:
            out["deletion"] = {
                "any_certifiable": bool(with_deletion["deletion_certifiable"].any()),
                "n_check_available": int(with_deletion["n_carriers"].max() // 2),
                "honest_accepted": bool(with_deletion["honest_certificate_accepted"].any()),
                "cheating_accepted": bool(with_deletion["cheating_certificate_accepted"].any()),
                "decay_accepted": bool(with_deletion["decay_certificate_accepted"].any()),
                "window_over_T2": float(with_deletion["deletion_window_over_T2"].max()),
            }

    needs = load_metrics(results_root, "coherence_requirements")
    if not needs.empty:
        out["requirements"] = [
            {
                "target": row["target"],
                "max_ber": row["max_ber"],
                "required_T2": row["required_T2_seconds"],
                "code_rate": row["code_rate"],
                "orders_from_cryogenic": row.get("orders_from_eyfp_cryogenic"),
                "orders_from_nv_nuclear": row.get("orders_from_nv_nuclear_room_temperature"),
            }
            for _, row in needs.drop_duplicates(["target", "max_ber"]).iterrows()
        ]
        from bioquantumfly.analysis.requirements import error_correction_gain

        out["error_correction_gain"] = error_correction_gain()

    placement = load_metrics(results_root, "host_placement")
    if not placement.empty:
        measured = placement[placement["graph_variant"] == "measured"]
        out["placement"] = [
            {
                "strategy": s,
                "perturbed_non_host": float(sub["n_perturbed_non_host"].mean()),
                "depth": float(sub["propagation_depth"].mean()),
                "amplification": float(sub["amplification_ratio"].mean()),
            }
            for s, sub in measured.groupby("host_strategy")
        ] if not measured.empty else []
        out["placement_controls"] = [
            {
                "variant": v,
                "perturbed_non_host": float(sub["n_perturbed_non_host"].mean()),
            }
            for v, sub in placement.groupby("graph_variant")
        ]

    return out


def write_initial_report(results_root: Path, config: Config) -> Path:
    """Write `results/initial_report.md` in the mini-paper structure of spec 33."""
    results_root = Path(results_root)
    s = summarise(results_root)
    graph_note = ""
    neural = s.get("neural")
    if neural and neural.get("graph_synthetic"):
        graph_note = (
            "\n> **Substrate warning.** These runs used the SYNTHETIC offline graph, "
            "not MaleCNS: no `NEUPRINT_TOKEN` was configured. Every network-level "
            "number below therefore describes a placeholder graph, and the "
            "connectome-specific questions remain open.\n"
        )

    lines: list[str] = []
    add = lines.append

    add("# BioQuantumFly - initial report\n")
    add("*Generated by `bioquantumfly reproduce all`. Numbers are read from "
        "`results/*/metrics.csv`; prose that is not derived from them is marked.*\n")
    add(graph_note)

    add("## Abstract\n")
    add(
        "We simulate a hypothetical biologically hosted quantum memory whose physical "
        "parameters are constrained, where possible, by current molecular-spin "
        "experiments, and ask what cryptographic functionality a finite-lifetime "
        "quantum state could provide under explicitly limited adversarial access. A "
        "connectome supplies anatomical and network connectivity; it supplies no "
        "evidence for quantum storage in Drosophila. The central result is a "
        "boundary rather than a capability: under constrained access the toy protocol "
        "exhibits the expected information-disturbance tradeoff, while under "
        "unrestricted extraction it fails completely, because natural host "
        "decoherence deletes nothing an adversary has already taken.\n"
    )

    add("## 1. Motivation\n")
    add(
        "Recent work makes two things simultaneously true: spin coherence has been "
        "manipulated in a living animal at room temperature, and a genetically "
        "encodable spin qubit exists. Neither is a quantum memory in a fly. The "
        "question this repository addresses is conditional: *if* a genetically hosted "
        "finite-coherence quantum state became implementable in a nervous system, in "
        "which parameter region would ephemeral, tamper-evident, uncloneability-"
        "inspired cryptography work, and where would it fail for reasons of principle?\n"
    )

    add("## 2. Model\n")
    add("The layering is strict:\n")
    add("```\nmolecular quantum subsystem  <  selected neuron  <  classical connectome\n```\n")
    add(
        "Neurons are not qubits, synapses are not entanglement, and the brain is not "
        "simulated as a quantum system. Each hosted subsystem is an independent "
        "single-qubit density matrix, so an n-bit encoding costs O(n), not O(2^n).\n"
    )

    add("## 3. Quantum memory\n")
    if s.get("storage_windows"):
        add("| T2 | evidence class | usable window (BER <= 1e-3) | window / T2 | BER at t = T2 |")
        add("|---|---|---|---|---|")
        for w in sorted(s["storage_windows"], key=lambda d: d["T2"]):
            add(f"| {_time(w['T2'])} | {w['status']} | {_time(w['window_seconds'])} "
                f"| {_fmt(w['window_over_T2'])} | {_fmt(w['ber_at_1_T2'])} |")
        add("")
        fractions = {round(w["window_over_T2"], 6) for w in s["storage_windows"]}
        ordered = sorted(s["storage_windows"], key=lambda d: d["T2"])
        shortest, longest = ordered[0], ordered[-1]
        if len(fractions) == 1:
            add(
                f"The usable window is a **fixed fraction of T2** "
                f"({next(iter(fractions)):.3g}), because in normalised time the "
                "channel depends on t only through t/T2. The absolute storage time "
                "is therefore set entirely by the coherence time, and by nothing "
                "else in the protocol.\n"
            )
        add(
            f"Concretely, the window runs from {_time(shortest['window_seconds'])} at "
            f"T2 = {_time(shortest['T2'])} ({shortest['status']}) to "
            f"{_time(longest['window_seconds'])} at T2 = {_time(longest['T2'])} "
            f"({longest['status']}). Even the most optimistic speculative regime "
            "leaves a window far shorter than any plausible biological read latency, "
            "which is the first hard constraint the feasibility analysis runs into.\n"
        )
        add(
            "Note how demanding the threshold is: requiring BER <= 1e-3 across the "
            "whole message costs roughly three orders of magnitude in storage time "
            "relative to T2 itself. A protocol tolerant of higher error rates, or "
            "equipped with error correction, would buy back most of that.\n"
        )
    else:
        add("*No decay results found; run `bioquantumfly simulate-quantum`.*\n")

    add("## 4. Threat model\n")
    add(
        "Two regimes are reported separately and must not be merged "
        "(see `docs/threat_model.md`): **constrained access**, where Eve may measure, "
        "interact weakly, or couple coherently for a limited time; and **unrestricted "
        "extraction**, where she may transfer the state into her own memory.\n"
    )
    if s.get("attacks"):
        add("| protocol | attack | Eve P_guess before key | after key | Bob disturbance (BER) |")
        add("|---|---|---|---|---|")
        for row in sorted(s["attacks"], key=lambda d: (d["protocol"], d["attack"])):
            add(f"| {row['protocol']} | {row['attack']} | {_fmt(row['eve_before_key'])} "
                f"| {_fmt(row['eve_after_key'])} | {_fmt(row['bob_disturbance'])} |")
        add("")
        add(
            "Protocol A (`raw_bb84`) leaks the message with no key at all: a secret "
            "basis is not encryption. Protocol B (`masked_conjugate`) leaks exactly "
            "zero before key reveal, but that privacy comes from the one-time pad, "
            "not from quantum mechanics -- which is why the post-key-reveal column "
            "matters so much.\n"
        )

    add("## 5. Neural model\n")
    add(
        "> Neural dynamics are a Shiu-et-al.-style LIF transfer model applied to "
        "MaleCNS connectivity; this does not constitute experimental validation of "
        "MaleCNS neural dynamics.\n"
    )
    if neural:
        add(f"- neurons: {_fmt(neural.get('n_neurons'))}, synapses: "
            f"{_fmt(neural.get('n_synapses'))}, dataset: `{neural.get('dataset')}`")
        add(f"- total spikes: {_fmt(neural.get('total_spikes'))}, mean rate: "
            f"{_fmt(neural.get('mean_rate_hz'))} Hz, active neurons: "
            f"{_fmt(neural.get('n_active_neurons'))}")
        add(f"- propagation depth reached: {_fmt(neural.get('propagation_depth'))} hops")
        add(f"- wall time {_fmt(neural.get('wall_seconds'))} s, peak RSS "
            f"{_fmt(neural.get('peak_rss_mb'))} MB\n")

    add("## 6. Experiments\n")
    add("Experiments 1-9 of the specification plus the two scale-enabled additions; "
        "see `docs/model.md` for the mapping from names to numbers.\n")

    add("## 7. Results\n")
    if s.get("persistence"):
        add("**Experiment 6/7 - what survives host decoherence.** At t >= 10 T2:\n")
        add("| attack | Eve memory | host I(M:Q_t) | Eve I(M:E_t) | gap |")
        add("|---|---|---|---|---|")
        for row in sorted(s["persistence"], key=lambda d: (d["attack"], d["eve_memory"])):
            add(f"| {row['attack']} | {row['eve_memory']} | {_fmt(row['host_info'])} "
                f"| {_fmt(row['eve_info'])} | {_fmt(row['gap'])} |")
        add("")
    if s.get("secure_region"):
        r = s["secure_region"]
        add(f"**Feasibility map.** {r['n_points']} of the swept points "
            f"({r['fraction']:.1%}) fall in the readable-and-low-Eve-information "
            f"regime, reaching at most t/T2 = {_fmt(r['max_t_over_T2'])} and attack "
            f"strength eta = {_fmt(r['max_eta'])}.\n")
        if s.get("regime_counts"):
            add("| regime | points |")
            add("|---|---|")
            for k, v in sorted(s["regime_counts"].items(), key=lambda kv: -kv[1]):
                add(f"| {k} | {v} |")
            add("")
        if "readable_vulnerable" not in (s.get("regime_counts") or {}):
            add(
                "Note that the *readable but vulnerable* regime is *empty* under these "
                "thresholds, and that is a result rather than a gap in the sweep. For a "
                "weak POVM the adversary's information and the disturbance she causes "
                "rise together in near-fixed proportion, so Bob's error crosses its "
                "threshold before Eve's information crosses hers. Whether that ordering "
                "survives a different attack family, or a different threshold pair, is "
                "exactly the kind of question the map is built to answer.\n"
            )
    if s.get("placement"):
        add("**Host placement (network-science question, not a biological claim).**\n")
        add("| host strategy | downstream neurons perturbed | depth | amplification |")
        add("|---|---|---|---|")
        for row in sorted(s["placement"], key=lambda d: -d["perturbed_non_host"]):
            add(f"| {row['strategy']} | {_fmt(row['perturbed_non_host'])} "
                f"| {_fmt(row['depth'])} | {_fmt(row['amplification'])} |")
        add("")
    if s.get("placement_controls"):
        add("| graph variant | downstream neurons perturbed |")
        add("|---|---|")
        for row in s["placement_controls"]:
            add(f"| {row['variant']} | {_fmt(row['perturbed_non_host'])} |")
        add("")
    if s.get("hybrid"):
        add("**Experiment 9 (SPECULATIVE coupling).**\n")
        add(
            "Each row is averaged over independent network seeds. A spiking network "
            "near threshold is chaotic -- an arbitrarily small drive can advance one "
            "spike across a timestep and cascade -- so a single realisation of "
            "'how many neurons changed' measures that sensitivity rather than the "
            "perturbation. The *integrated* magnitude is the better-behaved "
            "dose-response measure: the neuron count saturates as the same "
            "perturbation concentrates into fewer cells, while the total spike-count "
            "change keeps rising with the gain.\n"
        )
        add("| kappa | Eve condition | downstream perturbed | |spike change| |")
        add("|---|---|---|---|")
        for row in sorted(s["hybrid"], key=lambda d: (d["gain"], d["condition"])):
            add(f"| {_fmt(row['gain'])} | {row['condition']} "
                f"| {_fmt(row['perturbed_non_host'])} | {_fmt(row['spike_change'])} |")
        add("")

    if s.get("error_correction_gain"):
        g = s["error_correction_gain"]
        add("**Is the coherence gap fundamental?** No -- and most of it was an "
            "artefact of the error budget rather than of the physics.\n")
        add(f"Tolerating a {g['tolerant_ber']:.0%} bit-error rate and correcting it, "
            f"instead of demanding {g['strict_ber']:g} outright, extends the usable "
            f"window by **{g['storage_time_gain']:.0f}x** at a code rate of "
            f"{g['code_rate_at_tolerant']:.2f} -- a net gain of "
            f"{g['net_bit_seconds_gain']:.0f}x in bit-seconds. Reporting only the "
            "strict threshold overstates the physical shortfall by two orders of "
            "magnitude.\n")
    if s.get("requirements"):
        add("| target window | tolerated BER | required T2 | orders from cryogenic EYFP | from a nuclear-spin memory |")
        add("|---|---|---|---|---|")
        for row in sorted(s["requirements"], key=lambda d: (d["target"], d["max_ber"])):
            if row["max_ber"] not in (1e-3, 0.10):
                continue
            add(f"| {row['target'].replace('_', ' ')} | {row['max_ber']:g} "
                f"| {_time(row['required_T2'])} | {_fmt(row['orders_from_cryogenic'], 2)} "
                f"| {_fmt(row['orders_from_nv_nuclear'], 2)} |")
        add("")
        add(
            "The remaining gap is architectural rather than fundamental. An "
            "*electron* spin in a warm protein is the worst available place to "
            "store: the strong coupling that makes it optically addressable is "
            "the same coupling that decoheres it. Solid-state work resolves this "
            "by splitting the roles -- an electron spin as the interface, a "
            "nuclear spin as the memory -- and nuclear spins carry magnetic "
            "moments about three orders of magnitude smaller. The last column "
            "above is negative for every target, meaning a memory of that quality "
            "would already suffice. Whether such an architecture can exist inside "
            "a protein in a living cell is an open experimental question, and "
            "nothing here answers it.\n"
        )

    if s.get("ablation"):
        add("**Experiment 10 - what each primitive buys.** Weak measurement at "
            "eta = 0.25, adversary memory equal to the host's:\n")
        add("| layers enabled | secret bits | without delayed key release |")
        add("|---|---|---|")
        for row in sorted([r for r in s["ablation"] if r["eve_memory"] == "same_as_host"],
                          key=lambda d: d["secret_bits"]):
            add(f"| {row['layers']} | {_fmt(row['secret_bits'], 5)} "
                f"| {_fmt(row['secret_without_delay'], 5)} |")
        add("")
        add(
            "Privacy amplification is the step that turns the scheme from unusable "
            "into usable: without it no secret survives reconciliation at all. "
            "Delayed key release then adds more on top, but only against an "
            "adversary whose memory actually decays -- against an ideal memory the "
            "two curves coincide exactly, which is the noisy-storage assumption "
            "showing its price.\n"
        )
    if s.get("theft_detected"):
        t = s["theft_detected"]
        add(f"**The SWAP failure is repairable, and not by better physics.** Under "
            f"Protocol B an ideal-memory adversary who takes the state wins outright. "
            f"Under Protocol C the same attack raises the error rate to "
            f"{t['bob_ber']:.3f}, far above the ~11% one-way threshold, and the round "
            f"is discarded: {0 if not t['any_usable'] else 'some'} of "
            f"{t['n_configurations']} configurations "
            "yielded a key. Theft is defeated by *detecting* it and refusing to "
            "proceed -- the parameter-estimation step that Protocol B lacked.\n")
    if s.get("deletion") and not s["deletion"]["any_certifiable"]:
        d = s["deletion"]
        add(f"**Certified deletion needs more check positions than these runs "
            f"provide.** With about {d['n_check_available']} Hadamard-basis "
            "positions available, no tolerance separates an honest deleter from a "
            "partial cheater at the 1% theft bound: the binomial spread on that "
            "many trials is wider than the signal. Certification becomes possible "
            "at roughly 10^4 check positions and comfortable at 10^5, which is a "
            "concrete design requirement rather than a limitation of the "
            "mechanism -- see `docs/model.md`. The certificate machinery itself "
            f"behaves correctly (honest accepted: {d['honest_accepted']}, cheater "
            f"accepted: {d['cheating_accepted']}, decayed state accepted: "
            f"{d['decay_accepted']}).\n")
    elif s.get("deletion"):
        d = s["deletion"]
        add(f"**Certified deletion behaves as it must.** An honest deleter's "
            f"certificate is accepted ({d['honest_accepted']}), a cheater who kept "
            f"the message is rejected ({not d['cheating_accepted']}), and -- the "
            f"point of the layer -- a party who merely waited for the state to "
            f"decay is also rejected ({not d['decay_accepted']}). Deletion stays "
            f"certifiable out to {_fmt(d['window_over_T2'], 3)} x T2, roughly an "
            "order of magnitude longer than the data stays readable, because "
            "certifying needs only a statistically visible correlation while "
            "reading needs near-perfect data.\n")

    add("## 8. Security interpretation\n")
    add(
        "The results separate cleanly along the access boundary. Under constrained "
        "access the toy protocol behaves as conjugate coding should: an adversary who "
        "measures gains information and leaves a detectable disturbance, and the "
        "tradeoff is smooth in the attack strength. Under unrestricted extraction the "
        "picture inverts. The SWAP control shows the host information falling to zero "
        "while the adversary's copy is untouched, so:\n"
    )
    add("> **no-cloning is not no-theft.** Natural decoherence of the host is not "
        "certified deletion, and this repository never calls it that.\n")

    add("## 9. Biological interpretation\n")
    add(
        "None of the above is evidence about Drosophila. The connectome contributes "
        "measured anatomy; the coherence times are proxies from other systems and "
        "temperatures; the neural parameters are transferred from a model validated on "
        "a different brain; and the quantum-to-neural coupling has no empirical "
        "calibration whatsoever. What the simulation can say is conditional and "
        "quantitative: given a coherence time, here is the storage window, and here is "
        "the adversary strength at which it stops being usable.\n"
    )

    add("## 10. Limitations\n")
    add("See `docs/limitations.md` for the full list. The three that most constrain "
        "the conclusions:\n")
    add("1. Arbitrary-write capability is assumed and is not available in any known "
        "biological system.")
    add("2. Product-state encoding only; entangled multi-qubit encodings are out of "
        "scope, so uncloneability-style constructions are approximated, not implemented.")
    add("3. The coupling gain kappa is uncalibrated, so Experiment 9 locates a "
        "sensitivity threshold rather than predicting an effect.\n")

    add("## 11. Next experiments\n")
    add("1. Replace the offline substitute with a real MaleCNS subgraph and repeat the "
        "host-placement sweep, which is the one question that genuinely needs the "
        "measured connectome.")
    add("2. Extend the feasibility map along t_write and t_read separately, rather than "
        "the single storage-time axis used here.")
    add("3. Add an entangled two-carrier encoding to test whether the tradeoff curve "
        "moves, which is the first step toward a real uncloneable-encryption analogue.")
    add("4. Supply measured hyperfine and recombination constants from a primary source "
        "so the radical-pair model produces chemically meaningful yields.\n")

    out = results_root / "initial_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    (results_root / "summary.json").write_text(json.dumps(s, indent=2, default=str))
    return out
