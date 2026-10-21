"""Exploratory resource estimates from fixed Phase 1D models and saved trials.

No new neural trajectories, model fitting or selection. Access budgets count
repeated preparations/terminal measurements, not a free readout of a state.
"""

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["MPLBACKEND"] = "Agg"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .phase1 import write_json
from .predictive import predict, probability_diagnostics, probability_error_bounds, brier_improvement_bounds
from .statistics import simultaneous_binomial

ROOT = Path(__file__).resolve().parents[3]


def allocation(budget, probes):
    if type(budget) is not int or type(probes) is not int or probes < 1 or budget < probes:
        raise ValueError("integer budget must allow at least one trial per probe")
    return [budget//probes+int(i < budget%probes) for i in range(probes)]


def anchor_frequencies(outcomes, budget):
    if outcomes.ndim != 3:
        raise ValueError("expected preparation x probe x shot outcomes")
    sizes = allocation(budget, outcomes.shape[1])
    if max(sizes) > outcomes.shape[2]:
        raise ValueError("requested access budget exceeds saved outcomes")
    return np.column_stack([outcomes[:,j,:n].mean(axis=-1) for j,n in enumerate(sizes)]), sizes


def arithmetic_profile(model, future_length=4):
    """Specified float64 payload / compact affine arithmetic, not RAM timing.

    Dense serialization keeps unit columns. The compact estimate removes
    fixed unit columns and represents z by its k nonconstant coordinates.
    Initialization includes both mean and basis; the effect is kept even
    though it can be derived from that encoder in this particular model.
    """
    k, m, actions = model["dimension"], len(model["core"]), len(model["updates"])
    dense = sum(np.asarray(v).size for v in model["updates"].values())
    dense += sum(np.asarray(model[key]).size for key in ("mean", "basis", "effect"))
    dynamic = actions*k*(k+1)+(k+1)
    encoder = m+m*k
    return {"coordinates": k, "unit_included_dimension": k+1,
            "state_float64_bytes_compact": 8*k,
            "model_float64_bytes_dense_arrays": 8*dense,
            "model_float64_bytes_compact_with_encoder": 8*(dynamic+encoder),
            "dynamic_parameters_including_effect": dynamic,
            "update_multiplications": k*k, "update_additions": k*k,
            "readout_multiplications": k, "readout_additions": k,
            "one_future_arithmetic_ops_compact": future_length*2*k*k+2*k,
            "scope": "numeric payload only; excludes Python/JSON overhead, simulator, acquisition, fitting and physical energy"}


def order_profile(counts, shots, histories, futures):
    lo, hi = simultaneous_binomial(counts, shots, family_size=counts.size)
    first, second = futures.index("01"), futures.index("10")
    lower, upper = lo[:,first]-hi[:,second], hi[:,first]-lo[:,second]
    abs_lower = np.maximum(np.maximum(lower, -upper), 0)
    abs_upper = np.maximum(np.abs(lower), np.abs(upper))
    estimate = counts[:,first]/shots-counts[:,second]/shots
    i = int(np.argmax(abs_lower))
    return {"difference": "p(future=01)-p(future=10)", "family_size": counts.size,
            "maximum_absolute_gap_interval": [float(abs_lower.max()), float(abs_upper.max())],
            "witness_history": histories[i], "witness_point_difference": float(estimate[i]),
            "witness_difference_interval": [float(lower[i]),float(upper[i])],
            "histories_with_zero_excluded": int(np.count_nonzero(abs_lower > 0)),
            "all_differences": estimate, "lower": lower, "upper": upper,
            "interpretation": "observed order sensitivity; not a stand-alone contextuality or capacity certificate"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(source, output):
    output.mkdir(parents=True, exist_ok=False)
    names = ["models.json", "selection.json", "prediction_lock.json", "test_anchors_outcomes.npz",
             "test_targets_outcomes.npz", "train_outcomes.npz", "config.json", "split.json"]
    provenance = {"analysis": "exploratory reuse, no new simulations or model selection",
                  "started_utc": datetime.now(timezone.utc).isoformat(), "source_directory": str(source),
                  "inputs_sha256": {name: digest(source/name) for name in names}, "equal_access_budgets": [200,2000,10000],
                  "reference_operational_budget": 70000, "alpha_each_family": .05}
    files = [Path(__file__), ROOT/"src/bioquantumfly/operational/predictive.py",
             ROOT/"src/bioquantumfly/operational/statistics.py", ROOT/"src/bioquantumfly/operational/phase1.py",
             ROOT/"docs/operational_gpt/resource_priority.md"]
    provenance["sources_sha256"] = {}
    for path in files:
        relative = path.relative_to(ROOT)
        dest = output/"source_snapshot"/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(path.read_bytes())
        provenance["sources_sha256"][str(relative)] = digest(path)
    write_json(output/"provenance.json",provenance)
    lock = json.loads((source/"prediction_lock.json").read_text())
    for name, value in lock["files_sha256"].items():
        if digest(source/name) != value:
            raise ValueError("source prediction lock changed")
    models = json.loads((source/"models.json").read_text())
    selection = json.loads((source/"selection.json").read_text())
    op = models["models"][selection["selected_index"]]
    baseline = models["immediate"][selection["selected_immediate_index"]]
    anchors = np.load(source/"test_anchors_outcomes.npz")["outcomes"]
    target_file = np.load(source/"test_targets_outcomes.npz")
    targets = target_file["outcomes"]
    futures = target_file["futures"].tolist()
    counts, shots = targets.sum(axis=-1), targets.shape[-1]
    lo, hi = simultaneous_binomial(counts, shots)

    def metrics(p):
        return {**probability_error_bounds(p,lo,hi), "probabilities": probability_diagnostics(p),
                "rmse_to_frequencies": float(np.sqrt(np.mean((p-counts/shots)**2)))}

    rows = []
    for budget in provenance["equal_access_budgets"]:
        x, sizes = anchor_frequencies(anchors, budget)
        b, bsizes = anchor_frequencies(anchors[:,:1], budget)
        op_prediction, base_prediction = predict(op,x,futures), predict(baseline,b,futures)
        rows.append({"total_access_shots_per_preparation": budget, "operational_allocation": sizes,
                     "baseline_allocation": bsizes, "operational": metrics(op_prediction),
                     "baseline": metrics(base_prediction),
                     "expected_brier_improvement_interval": brier_improvement_bounds(base_prediction,op_prediction,lo,hi),
                     "operational_predictions": op_prediction, "baseline_predictions": base_prediction})
    original, _ = anchor_frequencies(anchors, 70000)
    reference = metrics(predict(op,original,futures))
    train = np.load(source/"train_outcomes.npz")
    order = order_profile(train["outcomes"].sum(axis=-1), train["outcomes"].shape[-1],
                          train["histories"].tolist(),train["futures"].tolist())
    result = {"model_refitted": False, "new_neural_trials": 0, "matched_access": rows,
              "reference_operational_70000": reference, "order_sensitivity": order,
              "operational_implementation": arithmetic_profile(op), "baseline_implementation": arithmetic_profile(baseline),
              "memory_and_nonlinear_task_capacity": "not_measured",
              "scope": "conditional on saved anchor samples; reused evaluation data, exploratory budget profile"}
    write_json(output/"summary.json",result)
    lines = ["# GPT-like挙動と資源推定：既存データの探索的集計", "",
             "モデルの再学習・再選択や、新しい神経試行は行っていない。元の最終評価データを再利用した資源プロファイルである。", "",
             "## 入力順序への依存", "",
             f"28履歴×15テストの同時95%領域から得た最大順序差の絶対値の区間: {order['maximum_absolute_gap_interval']}。",
             f"代表履歴 `{order['witness_history']}` の01−10の推定差: {order['witness_point_difference']:.6f}、区間: {order['witness_difference_interval']}。",
             f"零を除外できた履歴数: {order['histories_with_zero_excluded']}/28。入力順序への感度として扱い、単独で文脈性や計算容量を認定しない。", "",
             "## 同じアクセス予算での比較", "",
             "| 準備ごとの総測定数 | 操作的モデル：確率RMSE上限 | 即時モデル：確率RMSE上限 | 期待Brier改善区間 |", "|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['total_access_shots_per_preparation']} | {row['operational']['rmse_upper']:.6f} | {row['baseline']['rmse_upper']:.6f} | {row['expected_brier_improvement_interval']} |")
    lines += ["", "正の改善区間は操作的モデルの改善を示す。ゼロを含む場合は未確認。比較する二方式でモデル記憶・更新の費用まで等しくした実験ではない。",
              f"元の操作的モデルの7万測定時のRMSE上限は{reference['rmse_upper']:.6f}。同じ7万測定の即時モデルを新たに取得してはいない。", "",
              "予算間は保存outcomeの入れ子の部分列で、独立反復ではない。区間は固定したanchor試料での64評価確率についてのもので、再測定による平均性能や新しい履歴全体への保証ではない。", "",
              "## 数値表現の費用", "", "| 項目 | 操作的モデル | 即時モデル |", "|---|---|---|"]
    for key in ["coordinates","state_float64_bytes_compact","model_float64_bytes_dense_arrays",
                "model_float64_bytes_compact_with_encoder","update_multiplications","update_additions","one_future_arithmetic_ops_compact"]:
        lines.append(f"| {key} | {result['operational_implementation'][key]} | {result['baseline_implementation'][key]} |")
    lines += ["", "float64を選び固定unit座標を省いた場合のpayload・アフィン演算数の見積もりである。実際のPython/JSON占有量、学習、準備、シミュレータ、物理回路の記憶・消費電力を含まない。",
              "遅延記憶・非線形課題の容量とnoise toleranceは今後の主測定対象で、今回は未測定。", "",
              "![アクセス予算と誤差](access_budget.png)"]
    (output/"report.md").write_text("\n".join(lines)+"\n")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7,4))
    for model, label in [("operational","4-coordinate model"),("baseline","Instantaneous model")]:
        ax.plot([r["total_access_shots_per_preparation"] for r in rows], [r[model]["rmse_upper"] for r in rows], "o-", label=label)
    ax.scatter([70000],[reference["rmse_upper"]],marker="*",s=120,label="Original 4-coordinate budget")
    ax.set(xscale="log",xlabel="Total access shots per preparation",ylabel="Upper bound on probability RMSE",title="Exploratory resource profile: fixed models")
    ax.legend();fig.tight_layout();fig.savefig(output/"access_budget.png",dpi=170);fig.savefig(output/"access_budget.pdf");plt.close(fig)
    for name, value in provenance["inputs_sha256"].items():
        assert digest(source/name) == value
    provenance["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output/"provenance.json",provenance)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=ROOT/"results/operational_gpt/phase1d_prediction_v1")
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    result = run(args.source.resolve(),args.output.resolve())
    print(json.dumps({"order_gap_interval":result["order_sensitivity"]["maximum_absolute_gap_interval"],
                      "budgets":[{"shots":r["total_access_shots_per_preparation"],
                                  "operational_rmse_upper":r["operational"]["rmse_upper"],
                                  "baseline_rmse_upper":r["baseline"]["rmse_upper"],
                                  "improvement":r["expected_brier_improvement_interval"]} for r in result["matched_access"]]},indent=2))


if __name__ == "__main__":
    main()
