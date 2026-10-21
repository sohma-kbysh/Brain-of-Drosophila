"""Publication-quality figures (spec sections 27, 28).

Design rules applied throughout:

* Categorical hues are assigned in a fixed, validated order and never cycled.
  The seven-slot order below passes the adjacent-pair CVD and normal-vision
  gates on a light surface.
* Evidence class is encoded by LINE STYLE as well as by legend text -- solid for
  measured, dashed for proxy, dotted for speculative. This does double duty: it
  is the scientific labelling the spec demands, and it is the secondary encoding
  that keeps series distinguishable without relying on colour alone.
* Never a dual-axis chart. Two measures of different scale get two panels.
* Every figure writes its data to a CSV beside the PNG, which is the table view
  that discharges the sub-3:1 contrast relief obligation for the lighter hues.
* A figure built from speculative parameters, a synthetic graph, or the
  speculative coupling is stamped as such on the figure itself, not only in the
  caption -- a PNG travels away from its README.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

# --- palette (validated; see the dataviz reference instance) ----------------
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7")
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_LINE = "#c3c2b7"
SURFACE = "#fcfcfb"

#: Status palette, reserved for the feasibility regimes. Never used as series.
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

#: Evidence class -> line style. The secondary encoding.
STATUS_LINESTYLE = {
    "measured": "-",
    "proxy": "--",
    "speculative": ":",
    "transferred": "-.",
    "convention": "-",
}

SPECULATIVE_BANNER = "SPECULATIVE / NOT EXPERIMENTALLY DEMONSTRATED"
TRANSFER_CAVEAT = (
    "Neural dynamics: Shiu-et-al.-style LIF transfer model applied to MaleCNS\n"
    "connectivity. Not experimental validation of MaleCNS neural dynamics."
)


def apply_style() -> None:
    """Recessive grid and axes, ink-coloured text, thin marks."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS_LINE,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK_PRIMARY,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.6,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": INK_SECONDARY,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.0,
        "font.size": 9.5,
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })
    for spine in ("top", "right"):
        plt.rcParams[f"axes.spines.{spine}"] = False


def _series_color(i: int) -> str:
    """Fixed-order assignment. A ninth series folds into 'Other' upstream."""
    if i >= len(SERIES):
        raise ValueError(
            f"series index {i} exceeds the {len(SERIES)}-slot categorical palette; "
            "fold extra series into 'Other' or facet into small multiples "
            "rather than generating a new hue"
        )
    return SERIES[i]


def _format_time(seconds: float) -> str:
    """Human-readable duration. Spans the full range the sweeps produce.

    Usable storage windows reach into picoseconds at the in-vivo proxy
    coherence times, so the ladder runs all the way down rather than falling
    back to scientific notation there.
    """
    if seconds == 0:
        return "0 s"
    for scale, unit in ((1.0, "s"), (1e-3, "ms"), (1e-6, "us"),
                        (1e-9, "ns"), (1e-12, "ps"), (1e-15, "fs")):
        if seconds >= scale:
            value = seconds / scale
            return f"{value:.3g} {unit}" if value < 100 else f"{value:.0f} {unit}"
    return f"{seconds:.3g} s"


def _stamp(fig, text: str, color: str = "#d03b3b") -> None:
    """Put a provenance stamp on the figure itself."""
    fig.text(0.995, 0.005, text, ha="right", va="bottom", fontsize=7.5,
             color=color, style="italic")


def _save(fig, path: Path, data: pd.DataFrame | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    if data is not None:
        data.to_csv(path.with_suffix(".csv"), index=False)
    return path


def _label_last_point(ax, x: Sequence[float], y: Sequence[float], text: str,
                      color: str, min_gap_frac: float = 0.045) -> None:
    """Direct label at the end of a line, for four or fewer series.

    Skips the label when it would collide with one already placed on this axis:
    overlapping direct labels are worse than none, and the legend still carries
    identity. Placed labels are tracked on the axis itself.
    """
    if len(x) == 0:
        return
    placed = getattr(ax, "_bqf_labels", [])
    lo, hi = ax.get_ylim()
    span = (hi - lo) or 1.0
    if any(abs(y[-1] - prev) < min_gap_frac * span for prev in placed):
        return
    ax.annotate(text, xy=(x[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                color=color, fontsize=8, va="center", fontweight="bold")
    ax._bqf_labels = [*placed, float(y[-1])]


def _speculative_present(df: pd.DataFrame) -> bool:
    return bool(df.get("is_speculative", pd.Series(dtype=bool)).any())


# ---------------------------------------------------------------------------
# Figure 1 -- system architecture
# ---------------------------------------------------------------------------
def figure_architecture(path: str | Path) -> Path:
    """Schematic of the layered model. Not a data chart."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.grid(False)

    def box(x, y, w, h, label, face, edge, note=""):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.12", linewidth=1.2,
            facecolor=face, edgecolor=edge, zorder=2,
        ))
        ax.text(x + w / 2, y + h / 2 + (0.16 if note else 0), label, ha="center",
                va="center", fontsize=9, color=INK_PRIMARY, fontweight="bold", zorder=3)
        if note:
            ax.text(x + w / 2, y + h / 2 - 0.22, note, ha="center", va="center",
                    fontsize=7.2, color=INK_SECONDARY, style="italic", zorder=3)

    def arrow(x1, y1, x2, y2, color=INK_MUTED, style="-|>", dashed=False):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
            linewidth=1.3, color=color, zorder=1,
            linestyle="--" if dashed else "-", shrinkA=2, shrinkB=2,
        ))

    box(3.1, 9.6, 3.8, 0.9, "Alice", "#eaf2fd", SERIES[0], "writes conjugate-coded carriers")
    arrow(5.0, 9.6, 5.0, 8.9)
    box(2.4, 7.9, 5.2, 1.0, "Molecular quantum subsystem", "#eaf2fd", SERIES[0],
        "finite coherence T1, T2  (rho_i)")
    arrow(5.0, 7.9, 5.0, 7.2)
    box(2.4, 6.2, 5.2, 1.0, "Host neuron (classical LIF)", "#eefaf5", SERIES[2],
        "one cell of the connectome")
    arrow(5.0, 6.2, 5.0, 5.5)
    box(1.6, 4.4, 6.8, 1.1, "MaleCNS classical connectome", "#eefaf5", SERIES[2],
        "measured anatomy and connectivity only")

    box(0.15, 7.9, 1.9, 1.0, "Eve", "#fdecea", STATUS["critical"], "measure / couple / SWAP")
    arrow(2.05, 8.4, 2.4, 8.4, color=STATUS["critical"])
    box(8.0, 7.9, 1.85, 1.0, "Bob", "#eaf2fd", SERIES[0], "reads with key")
    arrow(7.6, 8.4, 8.0, 8.4, color=SERIES[0])

    ax.text(5.0, 3.85, "optional, speculative coupling (default OFF)",
            ha="center", fontsize=8, color=STATUS["critical"], fontweight="bold")
    arrow(8.6, 7.9, 8.6, 3.2, color=STATUS["critical"], dashed=True)
    ax.text(8.75, 5.5, "y = Tr[M rho]\n-> dg = kappa (y - y0)", fontsize=7.4,
            color=STATUS["critical"], va="center")
    box(1.6, 2.2, 6.8, 1.0, "Downstream neural response", "#fdecea", STATUS["critical"],
        "tamper_alarm mode only")

    ax.text(5.0, 1.5, "molecular quantum subsystem  <  selected neuron  <  MaleCNS connectome",
            ha="center", fontsize=8.5, color=INK_SECONDARY, style="italic")
    ax.text(5.0, 0.95,
            "The fly brain is NOT modelled as a quantum computer. Neurons are not qubits\n"
            "and synapses are not entanglement.",
            ha="center", fontsize=7.8, color=INK_MUTED)
    ax.set_title("Figure 1  System architecture", loc="left", pad=12)
    return _save(fig, Path(path))


# ---------------------------------------------------------------------------
# Figure 2 -- Bob recovery vs normalised storage time
# ---------------------------------------------------------------------------
def figure_bob_recovery(df: pd.DataFrame, path: str | Path,
                        usable_ber: float = 1e-3) -> Path:
    """Legitimate readout, in normalised and in absolute time.

    Three panels, never a dual axis. The first two use t/T2 as the spec asks;
    in those units the curves for every T2 collapse onto one another exactly,
    because the channel depends on time only through t/T2. That collapse is the
    result, not a plotting bug -- so the third panel plots absolute seconds,
    where the coherence regimes separate by six orders of magnitude and the
    reader can see how long the memory actually lasts.
    """
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.2))
    t2_values = sorted(df["T2"].unique())
    n_series = len(t2_values)

    for i, t2 in enumerate(t2_values):
        sub = df[df["T2"] == t2].sort_values("t_over_T2")
        color = _series_color(i)
        status = str(sub["T2_status"].iloc[0])
        style = STATUS_LINESTYLE.get(status, "-")
        label = f"T2 = {_format_time(t2)} ({status})"
        axes[0].plot(sub["t_over_T2"], sub["bob_ber"], color=color,
                     linestyle=style, label=label)
        axes[1].plot(sub["t_over_T2"], sub["bob_exact_recovery"], color=color,
                     linestyle=style, label=label)
        # t = 0 has no place on a log axis, and clipping it to the floor draws a
        # spurious vertical stroke. Drop it rather than distort it.
        finite = sub[(sub["t_seconds"] > 0) & (sub["bob_ber"] > 0)]
        seconds = finite["t_seconds"].to_numpy()
        axes[2].plot(seconds, finite["bob_ber"], color=color,
                     linestyle=style, marker="o", markersize=3.5, label=label)
        if n_series <= 4:
            _label_last_point(axes[2], seconds, finite["bob_ber"].to_numpy(),
                              _format_time(t2), color)

    axes[0].axhline(0.5, color=INK_MUTED, linewidth=0.9, linestyle=(0, (1, 3)))
    axes[0].text(0.99, 0.5, "random guessing (BER = 1/2)", fontsize=7.5, ha="right",
                 va="bottom", color=INK_MUTED, transform=axes[0].get_yaxis_transform())
    axes[0].set_ylabel("Bob bit-error rate")
    axes[0].set_title("Per-bit error (normalised)", loc="left")
    axes[0].set_ylim(-0.02, max(0.56, float(df["bob_ber"].max()) * 1.08))
    axes[0].annotate("all T2 collapse\nonto one curve", xy=(0.55, 0.28),
                     xycoords="axes fraction", fontsize=8, color=INK_MUTED, style="italic")

    axes[1].set_ylabel("Pr[exact message recovery]")
    axes[1].set_title(
        f"Whole-message recovery (n = {int(df['n_carriers'].iloc[0])})", loc="left"
    )
    axes[1].set_ylim(-0.02, 1.02)

    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].axhline(usable_ber, color=STATUS["critical"], linewidth=1.0, linestyle="--")
    axes[2].text(0.01, usable_ber * 1.3, f"usable threshold BER = {usable_ber:g}",
                 fontsize=7.5, color=STATUS["critical"],
                 transform=axes[2].get_yaxis_transform())
    axes[2].set_ylabel("Bob bit-error rate")
    axes[2].set_title("Per-bit error (absolute time)", loc="left")
    axes[2].set_ylim(1e-6, 1.0)

    for ax, xlabel in zip(axes, ("storage time  t / T2", "storage time  t / T2",
                                 "storage time  t  (seconds)")):
        ax.set_xlabel(xlabel)
    axes[1].legend(loc="upper right", title="coherence time", title_fontsize=8)

    fig.suptitle("Figure 2  Legitimate readout against storage time",
                 x=0.005, ha="left", fontsize=11.5, fontweight="bold", color=INK_PRIMARY)
    if _speculative_present(df):
        _stamp(fig, f"dotted lines: {SPECULATIVE_BANNER}")
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    return _save(fig, Path(path), df)


# ---------------------------------------------------------------------------
# Figure 3 -- information vs disturbance
# ---------------------------------------------------------------------------
def figure_information_disturbance(df: pd.DataFrame, path: str | Path) -> Path:
    """Eve's information against the disturbance she causes Bob."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    attacks = sorted(df["attack"].unique())

    for i, attack in enumerate(attacks):
        sub = df[df["attack"] == attack].sort_values("eta")
        color = _series_color(i)
        axes[0].plot(sub["eta"], sub["eve_info_after_key"], color=color,
                     marker="o", label=f"{attack} - Eve information")
        axes[0].plot(sub["eta"], sub["bob_disturbance_ber"], color=color,
                     marker="s", linestyle="--", label=f"{attack} - Bob disturbance")
        axes[1].plot(sub["bob_disturbance_ber"], sub["eve_info_after_key"],
                     color=color, marker="o", label=attack)
        if len(attacks) <= 4:
            _label_last_point(axes[1], sub["bob_disturbance_ber"].to_numpy(),
                              sub["eve_info_after_key"].to_numpy(), attack, color)

    axes[0].set_xlabel("attack strength  eta")
    axes[0].set_ylabel("bits  /  induced error rate")
    axes[0].set_title("Both quantities against attack strength", loc="left")
    axes[0].legend(loc="upper left")

    axes[1].set_xlabel("Bob disturbance (induced bit-error rate)")
    axes[1].set_ylabel("Eve information after key reveal (bits)")
    axes[1].set_title("The tradeoff itself", loc="left")
    axes[1].legend(loc="lower right", title="attack", title_fontsize=8)

    fig.suptitle("Figure 3  Information-disturbance tradeoff",
                 x=0.005, ha="left", fontsize=11.5, fontweight="bold", color=INK_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, Path(path), df)


# ---------------------------------------------------------------------------
# Figure 4 -- host self-decay vs Eve retained information
# ---------------------------------------------------------------------------
def figure_self_decay_vs_eve(df: pd.DataFrame, path: str | Path) -> Path:
    """The central negative result of the project.

    Host information falls to zero while Eve's does not. Natural decoherence is
    not certified deletion.

    Both curves use the POST-KEY-REVEAL measure. The pre-reveal host quantity is
    identically zero under the masked protocol whatever the physics -- that is
    the one-time pad, not decoherence -- so plotting it would show a flat line
    and hide the effect. What decays is how much of the encoded value a
    key-holder could still recover from the host.

    Curves are averaged over T2 because the x-axis is already normalised by T2,
    so in these units the trajectories coincide up to finite-carrier sampling.
    """
    apply_style()
    attacks = sorted(df["attack"].unique())
    host_col = "host_info_after_key" if "host_info_after_key" in df.columns else "host_info_at_read"
    fig, axes = plt.subplots(1, len(attacks), figsize=(4.6 * len(attacks), 4.4),
                             sharey=True, squeeze=False)

    exported = []
    for panel, attack in enumerate(attacks):
        ax = axes[0][panel]
        sub = (
            df[df["attack"] == attack]
            .groupby("t_over_T2", as_index=False)[[host_col, "eve_info_after_key"]]
            .mean()
            .sort_values("t_over_T2")
        )
        exported.append(sub.assign(attack=attack))
        x = sub["t_over_T2"].to_numpy()
        host = sub[host_col].to_numpy()
        eve = sub["eve_info_after_key"].to_numpy()

        ax.fill_between(x, host, eve, where=eve >= host,
                        color=STATUS["critical"], alpha=0.12, linewidth=0,
                        label="Eve retains what the host has lost")
        ax.plot(x, host, color=SERIES[0], marker="o", label="host  I(M:Q_t)")
        ax.plot(x, eve, color=SERIES[1], marker="s", linestyle="--",
                label="Eve  I(M:E_t) after key reveal")
        ax.set_xscale("log")
        ax.set_xlabel("storage time  t / T2")
        ax.set_title(f"attack: {attack}", loc="left")
        _label_last_point(ax, x, host, "host", SERIES[0])
        _label_last_point(ax, x, eve, "Eve", SERIES[1])
        if panel == 0:
            ax.set_ylabel("recoverable information about the message (bits)")
            ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Figure 4  Host self-decay does not erase what Eve already took",
        x=0.005, ha="left", fontsize=11.5, fontweight="bold", color=INK_PRIMARY,
    )
    _stamp(fig, "shaded: Eve retains information the host has lost -- "
                "natural decoherence is NOT certified deletion", STATUS["critical"])
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    return _save(fig, Path(path), pd.concat(exported, ignore_index=True))


# ---------------------------------------------------------------------------
# Figure 5 -- physical regime comparison
# ---------------------------------------------------------------------------
def figure_regime_comparison(df: pd.DataFrame, path: str | Path,
                             usable_ber: float = 1e-3) -> Path:
    """Usable storage window per coherence regime, on one log axis."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8.6, 4.6))

    rows = []
    for t2 in sorted(df["T2"].unique()):
        sub = df[df["T2"] == t2].sort_values("t_over_T2")
        ok = sub[sub["bob_ber"] <= usable_ber]
        window_ratio = float(ok["t_over_T2"].max()) if not ok.empty else 0.0
        rows.append({
            "T2": t2,
            "T2_status": str(sub["T2_status"].iloc[0]),
            "T2_regime": str(sub["T2_regime"].iloc[0]),
            "usable_window_s": window_ratio * t2,
            "usable_window_over_T2": window_ratio,
        })
    summary = pd.DataFrame(rows)

    status_order = ["measured", "proxy", "speculative"]
    colors = {"measured": SERIES[2], "proxy": SERIES[0], "speculative": SERIES[3]}
    x = np.arange(len(summary))
    bars = ax.bar(
        x, np.maximum(summary["usable_window_s"], 1e-15),
        color=[colors.get(s, SERIES[4]) for s in summary["T2_status"]],
        width=0.62, edgecolor=SURFACE, linewidth=2.0,
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([_format_time(t) for t in summary["T2"]])
    ax.set_xlabel("coherence time T2")
    ax.set_ylabel(f"usable storage window (s), BER <= {usable_ber:g}")
    ax.set_title(
        "Figure 5  Usable storage window by physical regime", loc="left", pad=10
    )
    for rect, row in zip(bars, summary.itertuples()):
        ax.annotate(_format_time(row.usable_window_s) if row.usable_window_s > 0 else "none",
                    xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=8, color=INK_SECONDARY, fontweight="bold")
    ax.legend(
        handles=[Patch(facecolor=colors[s], label=s) for s in status_order
                 if s in set(summary["T2_status"])],
        loc="upper left", title="evidence class", title_fontsize=8,
    )
    if (summary["T2_status"] == "speculative").any():
        _stamp(fig, SPECULATIVE_BANNER)
    return _save(fig, Path(path), summary)


# ---------------------------------------------------------------------------
# Figure 6 -- connectome subgraph with hosts highlighted
# ---------------------------------------------------------------------------
def figure_connectome(graph, host_indices: Iterable[int], path: str | Path,
                      max_nodes: int = 400) -> Path:
    """Spring layout of a subgraph, hosts marked. Visualisation only."""
    import networkx as nx

    apply_style()
    idx = np.asarray(list(host_indices), dtype=np.int64)
    if graph.n_neurons > max_nodes:
        keep = np.unique(np.concatenate([idx, graph.k_hop(idx, hops=1)]))[:max_nodes]
        graph = graph.subgraph(keep, "visualisation subset")
        idx = np.array([i for i, b in enumerate(graph.body_ids)
                        if b in set(graph.body_ids[:len(idx)])], dtype=np.int64)
        idx = np.arange(min(len(idx), graph.n_neurons))

    g = graph.to_networkx()
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    ax.set_axis_off()
    ax.grid(False)
    pos = nx.spring_layout(g, seed=7, k=1.4 / max(1.0, np.sqrt(g.number_of_nodes())))

    host_bodies = set(graph.body_ids[idx].tolist()) if idx.size else set()
    node_colors = [SERIES[1] if b in host_bodies else SERIES[0] for b in g.nodes()]
    node_sizes = [70 if b in host_bodies else 16 for b in g.nodes()]

    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=GRIDLINE, width=0.5,
                           arrows=False, alpha=0.7)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=node_colors, node_size=node_sizes,
                           linewidths=0.8, edgecolors=SURFACE)

    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="", color=SERIES[1], markersize=9,
               label=f"hypothetical quantum hosts (n = {len(host_bodies)})"),
        Line2D([], [], marker="o", linestyle="", color=SERIES[0], markersize=6,
               label=f"other neurons (n = {g.number_of_nodes() - len(host_bodies)})"),
    ], loc="upper left")
    label = "SYNTHETIC graph -- not MaleCNS" if graph.is_synthetic else graph.dataset
    ax.set_title(f"Figure 6  Connectome subgraph and host placement\n{label}",
                 loc="left", pad=10)
    _stamp(fig, "Host placement is a hypothetical engineering scenario, "
                "not a biological identification.", INK_MUTED)
    return _save(fig, Path(path))


# ---------------------------------------------------------------------------
# Figure 7 -- downstream neural response with and without Eve
# ---------------------------------------------------------------------------
def figure_downstream_response(df: pd.DataFrame, path: str | Path) -> Path:
    """Perturbed-neuron count and integrated spike change, by attack condition."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
    conditions = [c for c in ("none", "weak", "strong") if c in set(df["attack_condition"])]
    gains = sorted(df["speculative_coupling_gain"].unique())

    for i, cond in enumerate(conditions):
        sub = df[df["attack_condition"] == cond].groupby(
            "speculative_coupling_gain", as_index=False
        )[["n_perturbed_non_host", "absolute_spike_count_change"]].mean()
        color = _series_color(i)
        for ax, col in zip(axes, ("n_perturbed_non_host", "absolute_spike_count_change")):
            ax.plot(np.maximum(sub["speculative_coupling_gain"], 1e-4), sub[col],
                    color=color, marker="o", label=f"Eve: {cond}")
            _label_last_point(ax, np.maximum(sub["speculative_coupling_gain"], 1e-4).to_numpy(),
                              sub[col].to_numpy(), cond, color)

    for ax, ylabel, title in zip(
        axes,
        ("downstream neurons with changed spike count", "total |change| in spike count"),
        ("Spatial extent (saturates)", "Integrated magnitude (dose-response)"),
    ):
        ax.set_xscale("log")
        ax.set_xlabel("speculative coupling gain  kappa")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        # kappa = 0 has no place on a log axis; it is drawn at the left edge.
        ax.annotate("leftmost point\nis kappa = 0", xy=(1e-4, 0), xytext=(3, 14),
                    textcoords="offset points", fontsize=7.2, color=INK_MUTED,
                    style="italic")
    axes[0].legend(loc="upper left", title="attack condition", title_fontsize=8)
    axes[0].annotate(
        "extent saturates as the same perturbation\nconcentrates into fewer neurons",
        xy=(0.36, 0.62), xycoords="axes fraction", fontsize=7.6,
        color=INK_MUTED, style="italic",
    )

    fig.suptitle("Figure 7  Downstream neural response with and without Eve",
                 x=0.005, ha="left", fontsize=11.5, fontweight="bold", color=INK_PRIMARY)
    _stamp(fig, f"{SPECULATIVE_BANNER} -- quantum-to-neural coupling has no empirical calibration.\n"
                + TRANSFER_CAVEAT, STATUS["critical"])
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    return _save(fig, Path(path), df)


# ---------------------------------------------------------------------------
# Feasibility phase diagram
# ---------------------------------------------------------------------------
REGIME_COLORS = {
    "readable_low_eve_info": STATUS["good"],
    "readable_vulnerable": STATUS["serious"],
    "self_erased": STATUS["warning"],
    "unreadable_even_for_bob": "#b8b6ad",
}
REGIME_LABELS = {
    "readable_low_eve_info": "readable, low Eve information",
    "readable_vulnerable": "readable but vulnerable",
    "self_erased": "self-erased",
    "unreadable_even_for_bob": "unreadable even for Bob",
}


def figure_feasibility_map(df: pd.DataFrame, path: str | Path) -> Path:
    """Regime classification over (T2, absolute storage time), faceted by eta.

    The x-axis is ABSOLUTE seconds, not t/T2. In normalised units the channel
    depends on time only through t/T2, so every T2 row would be identical and
    the map would carry no information about coherence time at all. Plotting
    real seconds is what turns it into a feasibility statement: for this T2, the
    memory is usable for this long.

    Regimes are a status encoding, never a magnitude ramp, and each is named in
    the legend so colour never carries the meaning alone.
    """
    apply_style()
    etas = sorted(df["eta"].unique())
    if len(etas) > 3:
        etas = [etas[0], etas[len(etas) // 2], etas[-1]]
    fig, axes = plt.subplots(1, len(etas), figsize=(4.5 * len(etas), 4.6),
                             sharey=True, squeeze=False)

    t2_values = sorted(df["T2"].unique())
    order = list(REGIME_COLORS)
    lookup = {r: i for i, r in enumerate(order)}
    cmap = matplotlib.colors.ListedColormap([REGIME_COLORS[r] for r in order])

    for panel, eta in enumerate(etas):
        ax = axes[0][panel]
        sub = df[df["eta"] == eta]
        for row_index, t2 in enumerate(t2_values):
            line = sub[sub["T2"] == t2].sort_values("t_read_seconds")
            line = line[line["t_read_seconds"] > 0]
            if line.empty:
                continue
            seconds = line["t_read_seconds"].to_numpy()
            codes = np.array([lookup.get(r, np.nan) for r in line["regime"]])
            # Draw each row as a strip of coloured segments along real time.
            edges = np.sqrt(seconds[:-1] * seconds[1:])
            edges = np.concatenate([[seconds[0] ** 2 / edges[0]], edges,
                                    [seconds[-1] ** 2 / edges[-1]]])
            for k, code in enumerate(codes):
                if np.isnan(code):
                    continue
                ax.axhspan(row_index - 0.42, row_index + 0.42,
                           xmin=0, xmax=1, facecolor="none")
                ax.fill_betweenx([row_index - 0.42, row_index + 0.42],
                                 edges[k], edges[k + 1],
                                 color=cmap(int(code)), linewidth=0)
        ax.set_xscale("log")
        ax.set_yticks(range(len(t2_values)))
        ax.set_yticklabels([_format_time(v) for v in t2_values])
        ax.set_ylim(-0.5, len(t2_values) - 0.5)
        ax.set_xlabel("storage time  t  (seconds)")
        ax.set_title(f"attack strength eta = {eta:g}", loc="left")
        ax.grid(False)
        if panel == 0:
            ax.set_ylabel("coherence time T2")

    thresholds = (
        f"regime thresholds are configurable modelling choices, not universal "
        f"cryptographic standards:  BER <= {df['threshold_max_bob_ber'].iloc[0]:g},  "
        f"I(M:E) <= {df['threshold_max_eve_info'].iloc[0]:g}"
    )
    fig.legend(handles=[Patch(facecolor=REGIME_COLORS[r], label=REGIME_LABELS[r])
                        for r in order],
               loc="lower center", ncol=4, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Figure 8  Cryptographic feasibility map",
                 x=0.005, ha="left", fontsize=11.5, fontweight="bold", color=INK_PRIMARY)
    fig.text(0.5, 0.075, thresholds, ha="center", fontsize=7.5, color=INK_MUTED,
             style="italic")
    fig.tight_layout(rect=(0, 0.13, 1, 0.93))
    return _save(fig, Path(path), df)


def figure_raw_vs_masked(df: pd.DataFrame, path: str | Path) -> Path:
    """Protocol A versus B: a secret basis alone is not encryption."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    protocols = sorted(df["protocol"].unique())
    attacks = sorted(df["attack"].unique())
    width = 0.8 / max(1, len(protocols))
    x = np.arange(len(attacks))

    table = []
    for i, protocol in enumerate(protocols):
        values = []
        for attack in attacks:
            sub = df[(df["protocol"] == protocol) & (df["attack"] == attack)]
            values.append(float(sub["eve_info_before_key"].max()) if not sub.empty else 0.0)
            table.append({"protocol": protocol, "attack": attack, "eve_info_before_key": values[-1]})
        bars = ax.bar(x + i * width - 0.4 + width / 2, values, width=width * 0.9,
                      color=_series_color(i), edgecolor=SURFACE, linewidth=2.0,
                      label=protocol)
        for rect, v in zip(bars, values):
            ax.annotate(f"{v:.3f}", xy=(rect.get_x() + rect.get_width() / 2, v),
                        xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=8, color=INK_SECONDARY)

    ax.set_xticks(x)
    ax.set_xticklabels(attacks, rotation=20, ha="right")
    ax.set_ylabel("Eve information BEFORE key reveal (bits)")
    ax.set_xlabel("attack")
    ax.set_title("Figure 9  A secret basis alone is not encryption", loc="left", pad=10)
    ax.legend(loc="upper left", title="protocol", title_fontsize=8)
    _stamp(fig, "raw_bb84 leaks the message with no key; masked_conjugate leaks "
                "exactly zero before key reveal.", INK_MUTED)
    return _save(fig, Path(path), pd.DataFrame(table))


# ---------------------------------------------------------------------------
# Figure 10 -- which primitive repairs which failure
# ---------------------------------------------------------------------------
def figure_protocol_ablation(df: pd.DataFrame, path: str | Path) -> Path:
    """Secret bits against attack strength, one line per layer set.

    Reads as a ladder: each added primitive lifts the curve, and the gap between
    two adjacent lines is what that primitive bought.
    """
    apply_style()
    memories = [m for m in ("same_as_host", "better", "ideal")
                if m in set(df["eve_memory_model"])]
    fig, axes = plt.subplots(1, len(memories), figsize=(4.7 * len(memories), 4.4),
                             sharey=True, squeeze=False)

    layer_order = sorted(df["layer_set"].unique(), key=lambda s: df.loc[
        df["layer_set"] == s, "layer_index"].iloc[0])

    for panel, memory in enumerate(memories):
        ax = axes[0][panel]
        subset = df[(df["eve_memory_model"] == memory) & (df["attack"] == "weak_measurement")]
        for i, layer in enumerate(layer_order):
            line = (subset[subset["layer_set"] == layer]
                    .groupby("eta", as_index=False)["secret_bits"].max()
                    .sort_values("eta"))
            if line.empty:
                continue
            # Layers that add nothing produce EXACTLY coincident curves -- which
            # is the finding, not a glitch. Tapering the linewidth lets the
            # hidden ones show through instead of vanishing under the last drawn.
            ax.plot(line["eta"], line["secret_bits"], color=_series_color(i),
                    marker="o", markersize=4.5, linewidth=4.0 - 0.6 * i,
                    solid_capstyle="round", label=layer)
        ax.set_xlabel("attack strength  eta")
        ax.set_title(f"Eve memory: {memory}", loc="left")
        if panel == 0:
            ax.set_ylabel("extractable secret (bits)")
            ax.legend(loc="lower left", fontsize=8, title="layers enabled",
                      title_fontsize=8)
        if memory == "ideal":
            ax.annotate("delayed key release adds\nnothing here: the curves\ncoincide",
                        xy=(0.30, 0.55), xycoords="axes fraction", fontsize=8,
                        color=INK_MUTED, style="italic")

    fig.suptitle("Figure 10  What each primitive buys",
                 x=0.005, ha="left", fontsize=11.5, fontweight="bold", color=INK_PRIMARY)
    _stamp(fig, "Toy composition, not a security proof: no formal finite-key or "
                "noisy-storage analysis is reproduced here.", INK_MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    return _save(fig, Path(path), df)


# ---------------------------------------------------------------------------
# Figure 11 -- the inverted feasibility question
# ---------------------------------------------------------------------------
def figure_coherence_requirement(df: pd.DataFrame, path: str | Path) -> Path:
    """Required T2 against the tolerated error rate, with reference systems.

    The point of the figure: the strict-threshold requirement sits two orders of
    magnitude above the error-corrected one, so quoting only the strict number
    overstates the physical gap.
    """
    from bioquantumfly.analysis.requirements import REFERENCE_COHERENCE_TIMES

    apply_style()
    fig, ax = plt.subplots(figsize=(8.8, 5.0))

    targets = sorted(df["target"].unique(), key=lambda t: df.loc[
        df["target"] == t, "window_seconds"].iloc[0])
    for i, target in enumerate(targets):
        line = (df[df["target"] == target]
                .groupby("max_ber", as_index=False)["required_T2_seconds"].first()
                .sort_values("max_ber"))
        ax.plot(line["max_ber"], line["required_T2_seconds"], color=_series_color(i),
                marker="o", label=target.replace("_", " "))
        _label_last_point(ax, line["max_ber"].to_numpy(),
                          line["required_T2_seconds"].to_numpy(),
                          target.replace("_", " "), _series_color(i))

    for name, (t2, note) in REFERENCE_COHERENCE_TIMES.items():
        measured = "measured" in note
        ax.axhline(t2, color=INK_MUTED if not measured else STATUS["good"],
                   linewidth=1.0, linestyle="--" if measured else (0, (1, 3)))
        # Reference labels go on the LEFT; the direct series labels occupy the
        # right edge, and the two collide otherwise.
        ax.text(0.012, t2, name.replace("_", " "), transform=ax.get_yaxis_transform(),
                ha="left", va="bottom", fontsize=7.2,
                color=INK_SECONDARY if measured else INK_MUTED)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("tolerated bit-error rate (error correction absorbs the rest)")
    ax.set_ylabel("required coherence time T2 (s)")
    ax.set_title("Figure 11  What would have to be true", loc="left", pad=10)
    ax.legend(loc="lower left", title="target storage window", title_fontsize=8)
    _stamp(fig, "dashed = measured coherence times (EYFP); dotted = context only "
                "(NV centres in diamond, NOT protein)", INK_MUTED)
    return _save(fig, Path(path), df)
