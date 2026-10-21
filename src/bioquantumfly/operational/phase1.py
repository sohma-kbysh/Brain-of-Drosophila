"""Run the three bounded, synthetic Phase 1 experiments.

Usage: python -m bioquantumfly.operational.phase1 --output <new-directory>
"""

import os

# This CLI owns its process; set limits before importing numeric libraries.
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["MPLBACKEND"] = "Agg"

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from .embedding import embedding_lp, pom_success_bounds, replacement_robustness
from .statistics import affine_rank_witness, factor_probabilities, select_dimension, simultaneous_binomial
from .toys import compression_toy, preparation_relations, rational_matrix, relative_tomography, shadow_toy

ROOT = Path(__file__).resolve().parents[3]


def seed_for(seed, *coordinates):
    return int(np.random.SeedSequence([seed, *coordinates]).generate_state(1)[0])


def write_json(path, content):
    def convert(value):
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        raise TypeError(type(value).__name__)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, default=convert, allow_nan=False)+"\n")


def write_table(path, counts, shots, lo, hi):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["preparation", "effect", "shots", "successes", "frequency", "lower", "upper"])
        for i, j in np.ndindex(counts.shape):
            writer.writerow([i, j, shots, int(counts[i,j]), counts[i,j]/shots, lo[i,j], hi[i,j]])


def run_a(config, output):
    from .lif import simulate_protocol
    levels, grids = config["shot_levels"], config["lif_dt_ms"]
    n = max(levels)
    family = len(levels)*len(grids)*12
    records, interval_by_grid = [], {}
    for g, dt in enumerate(grids):
        outcomes = np.zeros((4, 3, n), dtype=bool)
        seeds = np.zeros((4, 3), dtype=np.uint32)
        for i, prep in enumerate(config["lif_preparations"]):
            for j, future in enumerate(config["lif_futures"]):
                seeds[i,j] = seed_for(config["seed"], 1, g, i, j)
                print(f"1A: dt={dt} ms, P={prep}, future={future or 'immediate'}, N={n}", flush=True)
                outcomes[i,j] = simulate_protocol(prep, future, n, seed=seeds[i,j], dt_ms=dt)
        np.savez_compressed(output/f"1a_dt_{dt}_outcomes.npz", outcomes=outcomes, seeds=seeds)
        for shots in levels:
            counts = outcomes[..., :shots].sum(axis=-1)
            lo, hi = simultaneous_binomial(counts, shots, alpha=config["alpha"], family_size=family)
            witness = affine_rank_witness(lo, hi)
            record = {"dt_ms": dt, "shots": shots, "counts": counts, "probabilities": counts/shots,
                      "lower": lo, "upper": hi, "witness": witness,
                      "singular_values_with_unit": np.linalg.svd(np.c_[np.ones(4), counts/shots], compute_uv=False)}
            records.append(record)
            write_table(output/f"1a_dt_{dt}_n_{shots}.csv", counts, shots, lo, hi)
            if shots == n: interval_by_grid[dt] = (lo, hi)
    stability = {"status": "not_checked"}
    if len(grids) == 2:
        l0, h0 = interval_by_grid[grids[0]]; l1, h1 = interval_by_grid[grids[1]]
        low, high = l1-h0, h1-l0
        bound = float(np.max(np.maximum(np.abs(low), np.abs(high))))
        stability = {"status": "within_registered_tolerance" if bound <= config["dt_probability_tolerance"]
                     else "equivalence_not_established", "largest_absolute_difference_bound": bound,
                     "tolerance": config["dt_probability_tolerance"], "lower": low, "upper": high}
    necessity = all(r["witness"]["certified_rank_at_least_3"] for r in records if r["shots"] == n)
    return {"records": records, "instantaneous_scalar_insufficient": necessity,
            "relative_tomography": "not_established_for_LIF_parent", "dt_stability": stability,
            "gate_pass": necessity and stability["status"] == "within_registered_tolerance",
            "interpretation": "finite-interface predictive sufficiency only; no contextuality inference"}


def run_b(config, output):
    toy = compression_toy(); d = toy["probabilities"]
    audit = relative_tomography(toy["states"], np.c_[np.ones(6), toy["effects"]])
    algebra = {"lift_collect_residual": float(np.max(np.abs(toy["lift"]@toy["collect"]-np.eye(3)))),
               "dynamic_intertwining_residual": float(np.max(np.abs(toy["lift"]@toy["kernel"]-
                                                                     toy["coarse_kernel"]@toy["lift"])))}
    records = []
    for shots in config["shot_levels"]:
        rng = np.random.default_rng(seed_for(config["seed"], 2, shots))
        train, validation, test = [rng.binomial(shots, d) for _ in range(3)]
        selection_options = ({"validation_shots": shots, "alpha": config["rank_selection_alpha"]}
                             if config["rank_selection_rule"] == "paired_validation_hoeffding" else {})
        selected, scores = select_dimension(train/shots, validation/shots, **selection_options)
        states, effects = factor_probabilities(train/shots, selected)
        prediction = states@effects
        # Fixed, invertible unit-preserving gauge; probabilities, not coordinates, compared.
        gauge = np.eye(selected)
        if selected > 1:
            gauge[0,1] = .3; gauge[1,1] = 2
        transformed = (states@gauge)@np.linalg.solve(gauge, effects)
        choices = []
        calibration_rng = np.random.default_rng(seed_for(config["seed"], 20, shots))
        for _ in range(config["calibration_replicates"]):
            f1, f2 = [calibration_rng.binomial(shots, d)/shots for _ in range(2)]
            choices.append(select_dimension(f1, f2, **selection_options)[0])
        correct = choices.count(3); repeats = len(choices)
        lower, upper = simultaneous_binomial(np.array([correct]), repeats, alpha=.05)
        records.append({"shots": shots, "selected_linear_dimension": selected, "selected_d_op": selected-1,
                        "candidate_scores": scores, "states": states, "effects": effects,
                        "train_counts": train, "validation_counts": validation, "test_counts": test,
                        "reconstructed_probabilities": prediction,
                        "rmse_to_known_probabilities": float(np.sqrt(np.mean((prediction-d)**2))),
                        "test_rmse": float(np.sqrt(np.mean((prediction-test/shots)**2))),
                        "gauge_probability_residual": float(np.max(np.abs(transformed-prediction))),
                        "selected_dimension_counts": {str(k): choices.count(k) for k in sorted(set(choices))},
                        "correct_selection_fraction": correct/repeats,
                        "correct_selection_fraction_95ci": [float(lower[0]), float(upper[0])]})
        lo, hi = simultaneous_binomial(train, shots, alpha=config["alpha"],
                                      family_size=d.size*len(config["shot_levels"]))
        write_table(output/f"1b_train_n_{shots}.csv", train, shots, lo, hi)
        print(f"1B: N={shots}, selected d={selected}, correct calibration={correct}/{repeats}", flush=True)
    last = records[-1]
    return {"oracle": toy, "relative_tomography_audit": audit, "compression_algebra": algebra,
            "known_linear_dimension": 3, "known_d_op": 2, "records": records,
            "gate_pass": audit["relative_tomography"] and last["selected_linear_dimension"] == 3
                         and last["correct_selection_fraction"] >= .9
                         and last["rmse_to_known_probabilities"] <= .02
                         and max(algebra.values()) < 1e-12,
            "interpretation": "calibrated finite toy; data-driven SVD model selection, not universal tomography validation"}


def run_c(config, output):
    records, oracles = [], []
    # All two visibility conditions, two interfaces and shot levels in one family.
    family = len(config["shadow_visibilities"])*len(config["shot_levels"])*(7*3+7*2)
    for vindex, visibility in enumerate(config["shadow_visibilities"]):
        toy = shadow_toy(visibility)
        for cindex, case in enumerate(("adequate", "hidden")):
            d = toy[f"{case}_probabilities"]; e = toy[f"{case}_effects"]
            audit = relative_tomography(toy["states"], np.c_[np.ones(4), e])
            exact_pairing = rational_matrix(toy["states"]) * rational_matrix(np.c_[np.ones(4), e])
            basis = toy["states"] if case == "adequate" else np.array(exact_pairing.tolist(), dtype=object)
            relations = preparation_relations(basis)
            exact_lp = embedding_lp(d, d, relations)
            robustness = replacement_robustness(d, relations)
            oracles.append({"visibility": visibility, "case": case, "states": toy["states"],
                            "effects": e, "probabilities": d, "relations": relations,
                            "relation_source": "known_parent" if case == "adequate" else "known_restricted_quotient",
                            "relative_tomography_audit": audit, "exact_lp": exact_lp,
                            "noise_robustness": robustness})
            rng = np.random.default_rng(seed_for(config["seed"], 3, vindex, cindex))
            observations = rng.random((*d.shape, max(config["shot_levels"]))) < d[..., None]
            np.savez_compressed(output/f"1c_{case}_visibility_{visibility}_outcomes.npz", outcomes=observations)
            for shots in config["shot_levels"]:
                counts = observations[..., :shots].sum(axis=-1)
                lo, hi = simultaneous_binomial(counts, shots, alpha=config["alpha"], family_size=family)
                lp = embedding_lp(lo, hi, relations)
                pom = pom_success_bounds(lo, hi)
                certified = case == "hidden" and pom[0] > .75+1e-12 and lp["status"] == "infeasible_numeric"
                label = ("nonembeddable_shadow_parent_attribution_blocked" if certified else
                         "noncontextual_model_compatible" if lp["status"] == "feasible" else "undetermined")
                records.append({"visibility": visibility, "case": case, "shots": shots, "counts": counts,
                                "lower": lo, "upper": hi, "lp": lp, "pom_success_interval": pom,
                                "pom_bound_applicable": case == "hidden", "classification": label,
                                "parent_nonclassicality": False})
                write_table(output/f"1c_{case}_visibility_{visibility}_n_{shots}.csv", counts, shots, lo, hi)
                print(f"1C: {case}, visibility={visibility}, N={shots}: {label}", flush=True)
    # Negative control: incorrectly force the parity preparation identity on the adequate table.
    sharp = shadow_toy(1)
    wrong = preparation_relations(np.array((rational_matrix(sharp["states"]) *
                                           rational_matrix(np.c_[np.ones(4), sharp["hidden_effects"]])).tolist(), dtype=object))
    misuse = embedding_lp(sharp["adequate_probabilities"], sharp["adequate_probabilities"], wrong)
    gates = []
    for r in oracles:
        negative = r["case"] == "hidden" and r["visibility"] > .5
        expected_r = 1-.5/r["visibility"] if negative else 0
        gates.append(r["relative_tomography_audit"]["relative_tomography"] == (r["case"] == "adequate"))
        gates.append(r["exact_lp"]["status"] == ("infeasible_numeric" if negative else "feasible"))
        gates.append(abs(r["noise_robustness"]["replacement_probability"]-expected_r) < 1e-8)
    for r in records:
        if r["shots"] != max(config["shot_levels"]): continue
        expected = "nonembeddable_shadow_parent_attribution_blocked" if r["case"] == "hidden" and r["visibility"] > .5 else "noncontextual_model_compatible"
        gates.append(r["classification"] == expected)
    return {"oracles": oracles, "records": records, "deliberately_wrong_identity_control": misuse,
            "gate_pass": all(gates) and misuse["status"] == "scenario_inconsistent",
            "interpretation": "relative tomography must be audited against the parent; hidden-interface contextuality is not physical nonclassicality"}


def figures(result, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    a = result["1A"]
    if a["records"]:
        matrix = a["records"][-1]["probabilities"]
        axes[0].imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        axes[0].set_xticks(range(3), ["now", "wait", "input"])
        axes[0].set_yticks(range(4), ["00", "01", "10", "11"])
        for (i,j), value in np.ndenumerate(matrix): axes[0].text(j,i,f"{value:.3f}",ha="center",va="center",color="white")
    axes[0].set_title("1A: future spike probabilities")
    b = result["1B"]["records"]
    axes[1].plot([r["shots"] for r in b], [r["rmse_to_known_probabilities"] for r in b], "o-")
    axes[1].set(xscale="log", yscale="log", xlabel="Shots per setting", ylabel="RMSE to exact table", title="1B: probability reconstruction")
    for case, visibility, label in [("hidden",1.,"Sharp shadow"),("hidden",.4,"Weak shadow")]:
        rows=[r for r in result["1C"]["records"] if r["case"]==case and r["visibility"]==visibility]
        if rows:
            midpoint=np.array([sum(r["pom_success_interval"])/2 for r in rows])
            error=np.array([(r["pom_success_interval"][1]-r["pom_success_interval"][0])/2 for r in rows])
            axes[2].errorbar([r["shots"] for r in rows],midpoint,yerr=error,fmt="o-",label=label)
    axes[2].axhline(.75,ls="--",color="black",label="Noncontextual bound")
    axes[2].set(xscale="log",xlabel="Shots per setting",ylabel="POM interval",title="1C: shadow controls")
    axes[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output/"phase1_overview.png",dpi=170); fig.savefig(output/"phase1_overview.pdf")
    plt.close(fig)


def report(result, output):
    a,b,c=result["1A"],result["1B"],result["1C"]
    lines=["# Phase 1 実行結果", "", "全て合成の古典系。生体の文脈性・計算容量・計算優位を測定した結果ではない。", "",
           "| 段階 | gate | 判定対象 |", "|---|---|---|"]
    for name, r, meaning in [("1A",a,"即時確率一個の予測不足＋指定dt安定性"),
                            ("1B",b,"既知の操作的dimensionと確率の再構成"),
                            ("1C",c,"親に対する相対的完全性とshadow誤帰属の検出")]:
        lines.append(f"| {name} | {'PASS' if r['gate_pass'] else '未通過'} | {meaning} |")
    lines += ["", "## 1A", "", "| dt (ms) | N/セル | rank≥3の区間証拠 | 行列式区間 |", "|---|---|---|---|"]
    for r in a["records"]:
        w=r["witness"]
        lines.append(f"| {r['dt_ms']} | {r['shots']} | {w['certified_rank_at_least_3']} | {w.get('determinant_interval')} |")
    lines += ["", f"dt安定性: `{a['dt_stability']}`。", "即時に等しい準備を有限標本から厳密認定したものではなく、全混合に対するaffine十分性を検査した。LIF親系のrelative tomographyは未確立。", "",
              "## 1B", "", "親微視状態数6、準備span=効果span=pairing rank=3、正しい d_op=2。", "",
              "| N | 選択した線形d | d_op | oracle確率RMSE | 正しいdの選択率（100反復既定） |", "|---|---|---|---|---|"]
    for r in b["records"]:
        lines.append(f"| {r['shots']} | {r['selected_linear_dimension']} | {r['selected_d_op']} | {r['rmse_to_known_probabilities']:.6f} | {r['correct_selection_fraction']:.2%} |")
    lines += ["", "SVDの選択は有限標本のモデル選択で、正確なdimensionの統計的証明ではない。信頼区間・全候補score・S/E・独立test計数は summary.json に保存した。", "",
              "## 1C", "", "| case | visibility | 親に相対的な完全性 | 真の線形rank | LP | 置換noise閾値 |", "|---|---|---|---|---|---|"]
    for r in c["oracles"]:
        au=r["relative_tomography_audit"]
        lines.append(f"| {r['case']} | {r['visibility']} | {au['relative_tomography']} | {au['pairing_rank']} | {r['exact_lp']['status']} | {r['noise_robustness']['replacement_probability']:.6f} |")
    lines += ["", "鋭いhidden caseの非埋込性はPOMの3/4境界による区間証拠で確認する。LP表示のみを証明としない。商の文脈性は数学的に実在するが、親装置は同じ古典二ビット系であり非古典性へ帰属しない。弱いsquareは埋込可能で、形状だけでは判定できない。", "",
              f"故意の等式誤適用: `{c['deliberately_wrong_identity_control']['status']}`。", "",
              "## 到達点", "",
              "三段階が通過しても、生物学的interfaceの根拠・未知親系のrelative tomography・計算容量への追加説明力は未確立。小規模実connectomeへの移行設計は別途必要で、本実行はデータ取得を行わない。", "",
              "再現設定とソースhashは provenance.json、全数値と証明候補は summary.json、計数と区間はCSV、元outcomesはNPZに保存。", "",
              "![Phase 1 overview](phase1_overview.png)", "",
              f"方法・基準: [事前仕様]({os.path.relpath(ROOT/'docs/operational_gpt/phase1_protocol.md', output)})。"]
    # Avoid dumping array-valued stability bounds as repr in prose.
    lines=[line if not line.startswith("dt安定性:") else
           f"dt安定性: {a['dt_stability']['status']}。全セル差の絶対値上界最大={a['dt_stability'].get('largest_absolute_difference_bound', '未測定')}。" for line in lines]
    (output/"report.md").write_text("\n".join(lines)+"\n")


def validate_config(config):
    levels=config["shot_levels"]
    if not levels or levels != sorted(set(levels)) or any(type(n) is not int or not 1 <= n <= 20_000 for n in levels):
        raise ValueError("shot levels must be increasing positive integers <=20000")
    if config["lif_dt_ms"] != [.1, .05] or config["lif_preparations"] != ["00","01","10","11"] or config["lif_futures"] != ["","0","1"]:
        raise ValueError("registered LIF design must be preserved")
    fixed={"lif_bin_ms":10,"lif_readout_ms":2,"lif_rate_hz":100,
           "lif_recurrent_synapse_count":200,"lif_stimulus_drive_factor":1.2}
    if any(config.get(k)!=v for k,v in fixed.items()):
        raise ValueError("LIF constants differ from the implemented registered protocol")
    if type(config["calibration_replicates"]) is not int or not 1 <= config["calibration_replicates"] <= 500 or config["shadow_visibilities"] != [1.0,.4]:
        raise ValueError("bounded calibration and registered shadow controls required")
    if not 0 < config["alpha"] < 1 or not 0 < config["dt_probability_tolerance"] < 1:
        raise ValueError("invalid alpha or dt tolerance")
    if config.get("rank_selection_rule") not in ("raw_validation_minimum", "paired_validation_hoeffding"):
        raise ValueError("explicit rank selection rule required")
    if not 0 < config.get("rank_selection_alpha", 0) < 1:
        raise ValueError("invalid rank selection alpha")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/operational_phase1.json")
    parser.add_argument("--output",type=Path,required=True,help="New directory; existing output is never overwritten")
    args=parser.parse_args()
    config=json.loads(args.config.read_text()); validate_config(config)
    args.output.mkdir(parents=True,exist_ok=False)
    import brian2
    sources=[*Path(__file__).parent.glob("*.py"),ROOT/"src/bioquantumfly/neural/lif.py",
             ROOT/"docs/operational_gpt/phase1_protocol.md"]
    provenance={"config":config,"config_sha256":hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "sources_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                "versions":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"brian2":brian2.__version__},
                "synthetic":True,"quantum_coupling":False,"connectome_download":False,
                "workers":1,"started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    write_json(args.output/"provenance.json",provenance)
    write_json(args.output/"config.json",config)
    for source in sources:
        target=args.output/"source_snapshot"/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(source.read_bytes())
    started=time.perf_counter()
    result={"1A":run_a(config,args.output),"1B":run_b(config,args.output),"1C":run_c(config,args.output)}
    result["elapsed_seconds"]=time.perf_counter()-started
    write_json(args.output/"summary.json",result)
    figures(result,args.output); report(result,args.output)
    print(json.dumps({k:v["gate_pass"] for k,v in result.items() if isinstance(v,dict)},ensure_ascii=False),flush=True)
    print(f"Saved to {args.output}; {result['elapsed_seconds']:.1f}s",flush=True)


if __name__ == "__main__":
    main()
