"""Fixed additional confirmation, then a bounded predictive-state pilot.

python -m bioquantumfly.operational.phase1_next confirmation --output <new-dir>
python -m bioquantumfly.operational.phase1_next prediction --confirmation <dir> --output <new-dir>
"""

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["MPLBACKEND"] = "Agg"

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from .lif import simulate_sequence
from .phase1 import seed_for, write_json
from .predictive import (words, words_through, fit_predictive, predict, fit_suffix_baseline,
                         predict_suffix, choose_candidate, probability_diagnostics,
                         probability_error_bounds, brier_improvement_bounds)
from .statistics import simultaneous_binomial, affine_rank_witness

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT/"configs/operational_phase1_next.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_config(config):
    """This is a fixed study, not an unbounded parameter-sweep interface."""
    if config.get("schema") != "operational-phase1-next-v1" or config.get("batch_shots") != 20000:
        raise ValueError("registered schema and batch limit required")
    a, d = config["confirmation"], config["prediction"]
    if (a != {"shots": 100000, "dt_ms": [.1, .05, .025],
              "preparations": ["00", "01", "10", "11"], "futures": ["", "0", "1"],
              "probability_tolerance": .03}
            or d["dt_ms"] != .05 or d["train_history_lengths"] != [2, 3, 4]
            or d["heldout_history_length"] != 5 or d["core_max_length"] != 2
            or d["evaluation_futures"] != ["0000", "0101", "1010", "1111"]
            or d["train_shots"] != 10000 or d["anchor_shots"] != 10000 or d["target_shots"] != 20000
            or d["state_dimensions"] != list(range(1, 8)) or d["ridge_strengths"] != [1e-6, 1e-4, .01]
            or d["rmse_tolerance"] != .03 or d["maximum_error_tolerance"] != .05
            or config["alpha"] != .05 or d["selection_alpha"] != .05):
        raise ValueError("configuration differs from the registered finite experiment")
    if any(type(s) is not int or not 0 <= s < 2**32 for s in (config["seed"], d["split_seed"])):
        raise ValueError("seeds must be uint32 integers")


def start_run(output, config, config_path, stage):
    import brian2
    output.mkdir(parents=True, exist_ok=False)
    sources = sorted((ROOT/"src/bioquantumfly/operational").glob("*.py")) + [
        ROOT/"src/bioquantumfly/neural/lif.py", ROOT/"src/bioquantumfly/params.py",
        ROOT/"docs/operational_gpt/phase1_next_protocol.md"]
    hashes = {}
    for path in sources:
        relative = path.relative_to(ROOT)
        snapshot = output/"source_snapshot"/relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        hashes[str(relative)] = sha(path)
    (output/"config.json").write_bytes(config_path.read_bytes())
    provenance = {"stage": stage, "started_utc": now(), "config": config,
                  "config_sha256": sha(config_path), "sources_sha256": hashes,
                  "versions": {"python": platform.python_version(), "numpy": np.__version__,
                               "scipy": scipy.__version__, "brian2": brian2.__version__},
                  "synthetic": True, "classical": True, "threads": 1, "workers": 1,
                  "status": "running"}
    write_json(output/"provenance.json", provenance)
    return provenance


def sample_table(config, output, prefix, histories, futures, shots, dt, coordinate):
    """Independent trajectories for every cell and batch, persisted in full."""
    outcomes = np.zeros((len(histories), len(futures), shots), dtype=bool)
    with (output/"batches.jsonl").open("a") as log:
        for i, history in enumerate(histories):
            print(f"{prefix}: preparation {i+1}/{len(histories)} ({history}), {len(futures)} tests, N={shots}", flush=True)
            for j, future in enumerate(futures):
                for batch, start in enumerate(range(0, shots, config["batch_shots"])):
                    size = min(config["batch_shots"], shots-start)
                    seed = seed_for(config["seed"], coordinate, i, j, batch)
                    result = simulate_sequence(history, future, size, seed=seed, dt_ms=dt)
                    outcomes[i, j, start:start+size] = result
                    log.write(json.dumps({"dataset": prefix, "history": history, "future": future,
                                          "dt_ms": dt, "batch": batch, "start": start, "shots": size,
                                          "seed": seed, "successes": int(result.sum())})+"\n")
                    log.flush()
    counts = outcomes.sum(axis=-1)
    np.savez_compressed(output/f"{prefix}_outcomes.npz", outcomes=outcomes,
                        histories=np.array(histories), futures=np.array(futures))
    with (output/f"{prefix}_counts.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["history", "future", "shots", "successes", "frequency"])
        for i, j in np.ndindex(counts.shape):
            writer.writerow([histories[i], futures[j], shots, counts[i,j], counts[i,j]/shots])
    return counts


def run_confirmation(config, output):
    a = config["confirmation"]
    rows = []
    family = len(a["dt_ms"])*len(a["preparations"])*len(a["futures"])
    for g, dt in enumerate(a["dt_ms"]):
        counts = sample_table(config, output, f"dt_{dt}", a["preparations"], a["futures"],
                              a["shots"], dt, 100+g)
        lo, hi = simultaneous_binomial(counts, a["shots"], alpha=config["alpha"], family_size=family)
        rows.append({"dt_ms": dt, "counts": counts, "probabilities": counts/a["shots"],
                     "lower": lo, "upper": hi, "rank_witness": affine_rank_witness(lo, hi)})
    comparisons = []
    for first, second in combinations(rows, 2):
        lo, hi = second["lower"]-first["upper"], second["upper"]-first["lower"]
        largest = float(np.max(np.maximum(abs(lo), abs(hi))))
        comparisons.append({"dt_pair_ms": [first["dt_ms"], second["dt_ms"]],
                            "lower": lo, "upper": hi, "largest_difference_bound": largest,
                            "largest_point_difference": float(np.max(abs(second["probabilities"]-first["probabilities"]))),
                            "within_tolerance": largest <= a["probability_tolerance"]})
    return {"stage": "confirmation", "records": rows, "comparisons": comparisons,
            "shots_per_cell": a["shots"], "total_trial_pairs": family*a["shots"],
            "gate_pass": all(r["rank_witness"]["certified_rank_at_least_3"] for r in rows)
                         and all(c["within_tolerance"] for c in comparisons),
            "continuous_time_limit_established": False, "relative_tomography_established": False}


def prediction_split(config):
    d = config["prediction"]
    train = [h for length in d["train_history_lengths"] for h in words(length)]
    heldout = np.array(words(d["heldout_history_length"]))
    np.random.default_rng(d["split_seed"]).shuffle(heldout)
    mid = len(heldout)//2
    return train, heldout[:mid].tolist(), heldout[mid:].tolist()


def run_prediction(config, output):
    d = config["prediction"]
    train_h, val_h, test_h = prediction_split(config)
    core, train_f = words_through(d["core_max_length"]), words_through(d["core_max_length"]+1)
    target_f = d["evaluation_futures"]
    write_json(output/"split.json", {"train": train_h, "validation": val_h, "test": test_h,
                                     "core": core, "training_futures": train_f, "target_futures": target_f})
    train_counts = sample_table(config, output, "train", train_h, train_f, d["train_shots"], d["dt_ms"], 200)
    train = train_counts/d["train_shots"]
    models = [fit_predictive(train, train_f, core, k, ridge)
              for k in d["state_dimensions"] for ridge in d["ridge_strengths"]]
    immediate_models = [fit_predictive(train, train_f, [""], 1, ridge) for ridge in d["ridge_strengths"]]
    suffix_model = fit_suffix_baseline(train_h, train[:, train_f.index("")])
    val_anchors = sample_table(config, output, "validation_anchors", val_h, core, d["anchor_shots"], d["dt_ms"], 201)
    val_x = val_anchors/d["anchor_shots"]
    val_predictions = [predict(model, val_x, target_f) for model in models]
    val_immediate = [predict(model, val_x[:, :1], target_f) for model in immediate_models]
    # Forecasts conditional on independently measured anchors, before validation labels.
    write_json(output/"validation_predictions.json", {"models": val_predictions, "immediate": val_immediate})
    val_counts = sample_table(config, output, "validation_targets", val_h, target_f, d["target_shots"], d["dt_ms"], 202)
    chosen, scores = choose_candidate(models, val_predictions, val_counts, d["target_shots"], alpha=d["selection_alpha"]/2)
    immediate_chosen, immediate_scores = choose_candidate(immediate_models, val_immediate, val_counts,
                                                        d["target_shots"], alpha=d["selection_alpha"]/2)
    selection = {"selected_index": chosen, "selected_immediate_index": immediate_chosen,
                 "scores": scores, "immediate_scores": immediate_scores}
    write_json(output/"selection.json", selection)
    write_json(output/"models.json", {"models": models, "immediate": immediate_models, "suffix": suffix_model})
    print(f"1D: validation selected model index={chosen}, immediate index={immediate_chosen}", flush=True)
    test_anchors = sample_table(config, output, "test_anchors", test_h, core, d["anchor_shots"], d["dt_ms"], 203)
    test_x = test_anchors/d["anchor_shots"]
    forecasts = [predict(model, test_x, target_f) for model in models]
    immediate_forecasts = [predict(model, test_x[:, :1], target_f) for model in immediate_models]
    suffix_forecast = predict_suffix(suffix_model, test_h, target_f)
    write_json(output/"test_predictions.json", {"models": forecasts, "immediate": immediate_forecasts,
                                               "suffix": suffix_forecast})
    lock_paths = ["models.json", "selection.json", "test_predictions.json", "test_anchors_counts.csv"]
    lock = {"locked_utc": now(), "files_sha256": {p: sha(output/p) for p in lock_paths},
            "test_targets_exist_at_lock": (output/"test_targets_counts.csv").exists()}
    assert not lock["test_targets_exist_at_lock"]
    write_json(output/"prediction_lock.json", lock)
    print("1D: models and predictions locked; collecting final evaluation labels", flush=True)
    test_counts = sample_table(config, output, "test_targets", test_h, target_f, d["target_shots"], d["dt_ms"], 204)
    assert all(sha(output/p) == h for p, h in lock["files_sha256"].items())
    lo, hi = simultaneous_binomial(test_counts, d["target_shots"], alpha=config["alpha"])

    def evaluate(forecast):
        frequencies = test_counts/d["target_shots"]
        metrics = probability_error_bounds(forecast, lo, hi)
        metrics.update({"probabilities": probability_diagnostics(forecast),
                        "rmse_to_test_frequencies": float(np.sqrt(np.mean((forecast-frequencies)**2))),
                        "observed_brier": float(np.mean(forecast**2-2*forecast*frequencies+frequencies))})
        return metrics

    all_metrics = [{"index": i, "dimension": model["dimension"], "ridge": model["ridge"], **evaluate(forecast)}
                   for i, (model, forecast) in enumerate(zip(models, forecasts))]
    immediate_metrics = [evaluate(forecast) for forecast in immediate_forecasts]
    improvement, selected = None, None
    if chosen is not None:
        selected = all_metrics[chosen]
        if immediate_chosen is not None:
            improvement = brier_improvement_bounds(immediate_forecasts[immediate_chosen], forecasts[chosen], lo, hi)
    gate = (selected is not None and immediate_chosen is not None and selected["dimension"] < len(core)
            and selected["probabilities"]["valid"] and selected["rmse_upper"] <= d["rmse_tolerance"]
            and selected["maximum_error_upper"] <= d["maximum_error_tolerance"] and improvement[0] > 0)
    trial_pairs = (len(train_h)*len(train_f)*d["train_shots"]
                   +(len(val_h)+len(test_h))*(len(core)*d["anchor_shots"]+len(target_f)*d["target_shots"]))
    return {"stage": "prediction", "selection": selection, "selected": selected, "all_model_metrics": all_metrics,
            "immediate_metrics": immediate_metrics, "suffix_metrics": evaluate(suffix_forecast),
            "improvement_over_immediate": improvement,
            "improvement_over_suffix": brier_improvement_bounds(suffix_forecast, forecasts[chosen], lo, hi) if chosen is not None else None,
            "test_counts": test_counts, "test_lower": lo, "test_upper": hi,
            "test_target_collection_finished_utc": now(), "total_trial_pairs": trial_pairs,
            "anchor_shots_per_new_preparation": len(core)*d["anchor_shots"],
            "gate_pass": bool(gate), "relative_tomography_established": False,
            "globally_valid_GPT_established": False, "minimal_dimension_established": False,
            "scope": "fixed prepared ensembles, four open-loop future sequences; no observation-conditioned filtering"}


def artifacts(result, output):
    import matplotlib.pyplot as plt
    if result["stage"] == "confirmation":
        fig, ax = plt.subplots(figsize=(7, 4))
        rows = result["comparisons"]
        ax.bar(range(len(rows)), [r["largest_difference_bound"] for r in rows], label="Simultaneous upper bound")
        ax.scatter(range(len(rows)), [r["largest_point_difference"] for r in rows], color="black", label="Point estimate", zorder=3)
        ax.set_xticks(range(len(rows)), [" vs ".join(map(str, r["dt_pair_ms"])) for r in rows])
        ax.axhline(.03, color="red", linestyle="--", label="Registered tolerance")
        ax.set(xlabel="Time steps (ms)", ylabel="Largest absolute probability difference", title="1A: independent fixed-shot confirmation")
        ax.legend()
        lines = ["# 1A追加確認", "", f"判定: {'PASS' if result['gate_pass'] else '未通過'}。各セル10万試行。", "",
                 "| dt比較 (ms) | 最大推定差 | 同時区間からの最大上界 | ±0.03内 |", "|---|---|---|---|"]
        for row in rows:
            lines.append(f"| {row['dt_pair_ms']} | {row['largest_point_difference']:.6f} | {row['largest_difference_bound']:.6f} | {row['within_tolerance']} |")
        lines += ["", "| dt (ms) | rank≥3の証拠 | 行列式区間 |", "|---|---|---|"]
        for row in result["records"]:
            w = row["rank_witness"]
            lines.append(f"| {row['dt_ms']} | {w['certified_rank_at_least_3']} | {w['determinant_interval']} |")
        lines += ["", "入力の離散化を含む確率の安定性。連続時間極限・LIF親の相対的完全性は未確立。既存runの結果は変更していない。"]
    else:
        fig, ax = plt.subplots(figsize=(7, 4))
        rows = result["all_model_metrics"]
        for ridge in sorted({row["ridge"] for row in rows}):
            part = [row for row in rows if row["ridge"] == ridge]
            ax.plot([row["dimension"] for row in part], [row["rmse_upper"] for row in part], "o-", label=f"ridge={ridge:g}")
        ax.axhline(.03, color="red", linestyle="--", label="Registered RMSE tolerance")
        if result["selected"] is not None:
            s = result["selected"]
            ax.scatter([s["dimension"]], [s["rmse_upper"]], marker="*", s=200, color="black", label="Chosen using validation", zorder=3)
        ax.set(xlabel="Predictive coordinates (excluding unit)", ylabel="Simultaneous upper bound on probability RMSE", title="1D: fixed held-out histories and 4-step futures")
        ax.legend(fontsize=8)
        lines = ["# 1D：有限予測圧縮pilot", "", f"判定: {'PASS' if result['gate_pass'] else '未通過'}。", "",
                 f"検証で選択したモデル: `{result['selected']}`", "",
                 f"即時状態モデルに対する期待Brier改善区間: `{result['improvement_over_immediate']}`。",
                 f"入力末尾2binモデルに対する改善区間: `{result['improvement_over_suffix']}`。", "",
                 "| 状態座標数 | ridge | 確率RMSE上界 | 最大誤差上界 | 確率範囲内 |", "|---|---|---|---|---|"]
        for row in rows:
            lines.append(f"| {row['dimension']} | {row['ridge']:g} | {row['rmse_upper']:.6f} | {row['maximum_error_upper']:.6f} | {row['probabilities']['valid']} |")
        lines += ["", "全候補の評価値は開示するが、最終評価でモデルを選び直していない。最終ラベル取得前のモデル・予測hashはprediction_lock.jsonに保存した。",
                  "検査対象は16準備×4未来列。anchor測定を条件にした有限予測であり、全未来十分性、最小次元、GPTの大域的妥当性、LIF親の相対的完全性は未確立。",
                  f"新しい準備一個のanchor費用は{result['anchor_shots_per_new_preparation']:,}試行。総試行ペア数は{result['total_trial_pairs']:,}。"]
    fig.tight_layout()
    fig.savefig(output/"overview.png", dpi=170)
    fig.savefig(output/"overview.pdf")
    plt.close(fig)
    lines += ["", "全て合成古典系。生物の計算資源・優位の結果ではない。", "", "![結果図](overview.png)", "",
              f"[手法点検と実行前仕様]({os.path.relpath(ROOT/'docs/operational_gpt/phase1_next_protocol.md', output)})"]
    (output/"report.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["confirmation", "prediction"])
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--confirmation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    if args.stage == "prediction":
        if args.confirmation is None:
            parser.error("prediction requires --confirmation with a completed passing run")
        previous = json.loads((args.confirmation/"summary.json").read_text())
        provenance = json.loads((args.confirmation/"provenance.json").read_text())
        if (previous.get("stage") != "confirmation" or not previous.get("gate_pass")
                or provenance.get("status") != "completed" or provenance["config_sha256"] != sha(args.config)
                or provenance["sources_sha256"]["src/bioquantumfly/operational/lif.py"] != sha(ROOT/"src/bioquantumfly/operational/lif.py")):
            parser.error("passing confirmation with matching configuration and simulator is required")
    output = args.output.resolve()
    provenance = start_run(output, config, args.config, args.stage)
    started = time.monotonic()
    try:
        result = run_confirmation(config, output) if args.stage == "confirmation" else run_prediction(config, output)
        result["elapsed_seconds"] = time.monotonic()-started
        write_json(output/"summary.json", result)
        artifacts(result, output)
        provenance.update({"status": "completed", "finished_utc": now(), "gate_pass": result["gate_pass"]})
        print(f"Completed {args.stage}; gate={result['gate_pass']}; {result['elapsed_seconds']:.1f} s", flush=True)
    except Exception as error:
        provenance.update({"status": "failed", "finished_utc": now(), "error": repr(error)})
        raise
    finally:
        write_json(output/"provenance.json", provenance)


if __name__ == "__main__":
    main()
