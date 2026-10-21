"""Registered common-control discovery and independent resource confirmation."""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time
import numpy as np

from .resource_benchmark import BITS, CostLedger, gate_words, metrics, reference_fixtures, transformed_bits
from .resource_lif import sample
from .statistics import simultaneous_binomial

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/operational_resource_experiment_v1.json"
PROTOCOL = ROOT / "docs/operational_gpt/resource_experiment_v1_protocol.md"


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed(*parts):
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], "little")


def now():
    return datetime.now(timezone.utc).isoformat()


def validate(config):
    if (config["schema"] != "operational-resource-experiment-v1"
            or config["neuron_counts"] != [2, 8, 16]
            or config["search_seeds"] != [913101, 913102, 913103, 913104, 913105]
            or config["preparation_candidates"] != 64 or config["gate_candidates"] != 64
            or config["discovery_shots_per_cell"] != 200
            or config["validation_shots_per_cell"] != 2000
            or config["confirmation_shots_per_cell"] != 20000
            or config["dt_ms"] != .1 or config["confirmation_max_gate_length"] != 4
            or config["alpha"] != .05):
        raise ValueError("not the registered experiment")


def source_paths():
    names = ("resource_benchmark.py", "resource_lif.cpp", "resource_lif.py",
             "resource_experiment.py", "statistics.py")
    return [Path(__file__).parent/name for name in names] + [CONFIG, PROTOCOL]


def initialize(output, config):
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for path in source_paths():
        relative = path.relative_to(ROOT)
        dest = output/"source_snapshot"/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        hashes[str(relative)] = digest(path)
    write(output/"config.json", config)
    write(output/"provenance.json", {
        "started_utc": now(), "status": "discovery_running", "sources": hashes,
        "config_sha256": digest(CONFIG), "python": platform.python_version(),
        "numpy": np.__version__, "compiler": subprocess.check_output(
            ["g++", "--version"], text=True).splitlines()[0],
        "model": "synthetic_classical_lif", "measured_connectome": False,
        "search_method": "common_random_search_control",
        "reset_physical_latency_and_energy": None,
        "single_monitor_enforcement_cost": None,
        "implementation_timing_pilot_trials_not_reused": 6000})
    refs = reference_fixtures()
    assert all(abs(r["lp_success"]-r["analytic_success"]) < 1e-10
               for r in refs["ontic_tv_lp"])
    write(output/"reference_fixtures.json", refs)


def evaluate(candidate, words, shots, config, directory, name, ledger,
             coordinate, *, dt=None):
    """Persist every sampled outcome and per-trial proxies, including failures."""
    path = directory/(name+".npz")
    if path.exists():
        raise FileExistsError(path)
    shape = (len(words), 4, 2, shots)
    outcome = np.zeros(shape, dtype=np.uint8)
    activity = np.zeros(shape, dtype=np.uint32)
    background = np.zeros(shape, dtype=np.uint32)
    drive = np.zeros(shape)
    records = []
    for wi, word in enumerate(words):
        for x in range(4):
            for query in range(2):
                cell_seed = seed(*coordinate, name, word, x, query)
                ledger.reserve(shots)
                out, cost = sample(candidate["neurons"], candidate["preparations"][x],
                                   candidate["gates"], candidate["gain"], word, query, shots,
                                   seed=cell_seed, dt_ms=dt or config["dt_ms"])
                outcome[wi, x, query] = out
                for dest, key in ((activity, "activity"), (background, "noise"), (drive, "drive")):
                    dest[wi, x, query] = cost.pop(key)
                ledger.record(**cost)
                records.append({"artifact": path.name, "word": word, "preparation": x,
                                "query": query, "seed": cell_seed, "count": int(out.sum()),
                                "dt_ms": dt or config["dt_ms"], **cost})
    np.savez_compressed(path, outcome=outcome, activity=activity, background=background,
                        drive=drive, words=np.array(words))
    with (directory/"cells.jsonl").open("a") as f:
        for record in records:
            f.write(json.dumps(record)+"\n")
    write(directory/"budget.json", ledger.to_dict())
    return outcome.mean(axis=-1)


def table_score(table, word=""):
    r = metrics(table, table, word)
    return r["success_interval"][0], r["parity_gap_interval"][0]


def prep_candidates(config, n, search_seed):
    rng = np.random.default_rng(seed(search_seed, n, "preparations"))
    result = []
    for index in range(config["preparation_candidates"]):
        if index == 0:
            preps, gain = 1.6*BITS, 2.
        elif index < 32:
            low, high = rng.uniform(0, .3, 2), rng.uniform(1.2, 2.2, 2)
            preps = np.clip(low+(high-low)*BITS+rng.normal(0, .15, (4, 2)), 0, 3)
            gain = rng.uniform(0, 8)
        else:
            preps, gain = rng.uniform(0, 3, (4, 2)), rng.uniform(0, 8)
        result.append({"neurons": n, "preparations": preps.tolist(), "gain": float(gain),
                       "gates": np.zeros((2, 2, 2)).tolist(), "preparation_candidate": index})
    return result


def discovery(output):
    config = json.loads(CONFIG.read_text()); validate(config)
    initialize(output, config)
    records = []
    for n in config["neuron_counts"]:
        for search_seed in config["search_seeds"]:
            directory = output/f"search_n{n}_seed{search_seed}"
            directory.mkdir()
            ledger = CostLedger(config["max_physical_trials_per_search_run"])
            candidates = prep_candidates(config, n, search_seed)
            train = []
            for index, candidate in enumerate(candidates):
                p = evaluate(candidate, [""], 200, config, directory,
                             f"prep_train_{index:03}", ledger, (search_seed, n))
                s, d = table_score(p[0]); train.append(s-2*d)
            finalists = sorted(range(len(train)), key=lambda i: (-train[i], i))[:4]
            validation = {}
            for index in finalists:
                p = evaluate(candidates[index], [""], 2000, config, directory,
                             f"prep_validation_{index:03}", ledger, (search_seed, n))
                s, d = table_score(p[0]); validation[index] = (s-2*d, s, d)
            winner = max(finalists, key=lambda i: (validation[i][0], -i))
            selected = candidates[winner]
            print(f"n={n} seed={search_seed} preparation S={validation[winner][1]:.4f} "
                  f"delta={validation[winner][2]:.4f}", flush=True)
            rng = np.random.default_rng(seed(search_seed, n, "gates"))
            gate_candidates = [dict(selected, gates=(np.zeros((2, 2, 2)) if i == 0
                               else rng.uniform(-3, 3, (2, 2, 2))).tolist(),
                                    gate_candidate=i) for i in range(64)]
            gate_train = []
            for index, candidate in enumerate(gate_candidates):
                p = evaluate(candidate, ["F", "W"], 200, config, directory,
                             f"gate_train_{index:03}", ledger, (search_seed, n))
                vals = [table_score(p[j], word) for j, word in enumerate(("F", "W"))]
                gate_train.append(min(v[0] for v in vals)-2*max(v[1] for v in vals))
            finalists_g = sorted(range(64), key=lambda i: (-gate_train[i], i))[:4]
            validation_g = {}
            for index in finalists_g:
                p = evaluate(gate_candidates[index], ["F", "W"], 2000, config, directory,
                             f"gate_validation_{index:03}", ledger, (search_seed, n))
                vals = [table_score(p[j], word) for j, word in enumerate(("F", "W"))]
                validation_g[index] = {"score": min(v[0] for v in vals)-2*max(v[1] for v in vals),
                                       "success": [v[0] for v in vals], "parity": [v[1] for v in vals]}
            best_gate = max(finalists_g, key=lambda i: (validation_g[i]["score"], -i))
            best = gate_candidates[best_gate]
            sg = validation_g[best_gate]
            selection_score = (min(validation[winner][1], *sg["success"])
                               -2*max(validation[winner][2], *sg["parity"]))
            record = {"search_seed": search_seed, "candidate": best,
                      "preparation_train_scores": train, "preparation_finalists": finalists,
                      "preparation_validation": validation,
                      "gate_train_scores": gate_train, "gate_finalists": finalists_g,
                      "gate_validation": validation_g,
                      "selection_score": selection_score, "ledger": ledger.to_dict()}
            assert ledger.physical_trials == config["expected_physical_trials_per_search_run"]
            write(directory/"selection.json", record)
            records.append(record)
            print(f"n={n} seed={search_seed} gate S={sg['success']} "
                  f"trials={ledger.physical_trials}", flush=True)
    chosen = []
    for n in config["neuron_counts"]:
        pool = [r for r in records if r["candidate"]["neurons"] == n]
        chosen.append(max(pool, key=lambda r: (r["selection_score"], -r["search_seed"])))
    lock = {"locked_utc": now(), "config_sha256": digest(CONFIG),
            "total_search_trials": sum(r["ledger"]["physical_trials"] for r in records),
            "selection_rule": "max validation min(S,S_F,S_W)-2 max(delta,delta_F,delta_W)",
            "devices": [{"candidate": r["candidate"], "search_seed": r["search_seed"],
                         "selection_score": r["selection_score"]} for r in chosen]}
    assert lock["total_search_trials"] <= config["max_campaign_search_trials"]
    write(output/"selection_lock.json", lock)
    write(output/"discovery_summary.json", {
        "runs": len(records), "total_trials": lock["total_search_trials"],
        "selected": lock["devices"], "method_comparison_performed": False,
        "ledgers": [r["ledger"] for r in records]})
    provenance = json.loads((output/"provenance.json").read_text())
    provenance.update(status="discovery_complete", discovery_completed_utc=now(),
                      selection_lock_sha256=digest(output/"selection_lock.json"))
    write(output/"provenance.json", provenance)


def interval_table(path, shots, family):
    with np.load(path) as archive:
        counts = archive["outcome"].sum(axis=-1)
    lo, hi = simultaneous_binomial(counts, shots, family_size=family)
    return counts/shots, lo, hi


def confirmation(output):
    config = json.loads((output/"config.json").read_text()); validate(config)
    provenance = json.loads((output/"provenance.json").read_text())
    for path in source_paths():
        if digest(path) != provenance["sources"][str(path.relative_to(ROOT))]:
            raise RuntimeError("source changed after discovery; create a new version")
    lock_hash = digest(output/"selection_lock.json")
    if lock_hash != provenance["selection_lock_sha256"]:
        raise RuntimeError("selection lock changed")
    lock = json.loads((output/"selection_lock.json").read_text())
    folder = output/"confirmation"; folder.mkdir(exist_ok=False)
    words = gate_words(4)
    waits = ["I"*n for n in (1, 2, 3, 4)]
    family = len(lock["devices"])*(8*(len(words)+len(waits)+2*5)+16)
    assert family == 1128
    write(folder/"confirmation_lock.json", {
        "created_utc": now(), "selection_lock_sha256": lock_hash,
        "shots": 20000, "words": words, "waits": waits, "family_size": family,
        "dt_audit_words": config["dt_audit_words"], "alpha": .05})
    ledger = CostLedger(config["max_campaign_confirmation_trials"])
    summaries = []
    for item in lock["devices"]:
        candidate = item["candidate"]; n = candidate["neurons"]
        directory = folder/f"n{n}"; directory.mkdir()
        print(f"confirmation n={n}: all 31 gate words, N=20000", flush=True)
        evaluate(candidate, words, 20000, config, directory, "words", ledger,
                 (config["confirmation_seed"], n))
        evaluate(candidate, waits, 20000, config, directory, "waits", ledger,
                 (config["confirmation_seed"], n))
        p, lo, hi = interval_table(directory/"words.npz", 20000, family)
        pw, low, high = interval_table(directory/"waits.npz", 20000, family)
        results = [metrics(lo[i], hi[i], word) for i, word in enumerate(words)]
        point = [metrics(p[i], p[i], word) for i, word in enumerate(words)]
        errors = []
        for i, word in enumerate(words[1:], 1):
            perm = transformed_bits(word) @ np.array([2, 1])
            lref, href = low[len(word)-1, perm], high[len(word)-1, perm]
            errors.append(float(np.maximum(np.abs(lo[i]-href), np.abs(hi[i]-lref)).max()))
        depth4 = [i for i, w in enumerate(words) if len(w) == 4]
        depth4_ci = np.mean([results[i]["success_interval"] for i in depth4], axis=0).tolist()
        each_lo = min(results[i]["success_interval"][0] for i in depth4)
        dt_results = {}
        for dt in config["dt_audit_values_ms"]:
            print(f"confirmation n={n}: dt={dt}, five prescribed words", flush=True)
            tag = f"dt_{dt}"
            evaluate(candidate, config["dt_audit_words"], 20000, config, directory,
                     tag, ledger, (config["confirmation_seed"], n), dt=dt)
            pd, ld, hd = interval_table(directory/(tag+".npz"), 20000, family)
            idx = [words.index(w) for w in config["dt_audit_words"]]
            differences = np.maximum(np.abs(lo[idx]-hd), np.abs(hi[idx]-ld))
            dt_results[str(dt)] = {
                "maximum_probability_difference_upper": float(differences.max()),
                "maximum_point_difference": float(np.abs(p[idx]-pd).max()),
                "scope": "five_prespecified_words_only"}
        joint = np.zeros((4, 20000), dtype=np.uint8)
        joint_activity = np.zeros_like(joint, dtype=np.uint32)
        joint_background = np.zeros_like(joint, dtype=np.uint32)
        joint_drive = np.zeros_like(joint, dtype=float)
        for x in range(4):
            ledger.reserve(20000)
            out, cost = sample(n, candidate["preparations"][x], candidate["gates"],
                               candidate["gain"], "", 2, 20000,
                               seed=seed(config["confirmation_seed"], n, "joint", x))
            joint[x] = out
            joint_activity[x] = cost.pop("activity")
            joint_background[x] = cost.pop("noise")
            joint_drive[x] = cost.pop("drive")
            ledger.record(**cost)
            with (directory/"cells.jsonl").open("a") as f:
                f.write(json.dumps({"artifact": "joint.npz", "preparation": x, "query": 2,
                    "seed": seed(config["confirmation_seed"], n, "joint", x),
                    "counts": np.bincount(out, minlength=4).tolist(), **cost})+"\n")
        np.savez_compressed(directory/"joint.npz", outcome=joint, activity=joint_activity,
                            background=joint_background, drive=joint_drive)
        jc = np.array([np.bincount(row, minlength=4) for row in joint])
        jl, jh = simultaneous_binomial(jc, 20000, family_size=family)
        dlow = (jl[0]+jl[3]-jh[1]-jh[2])/2
        dhigh = (jh[0]+jh[3]-jl[1]-jl[2])/2
        tvlo = .5*np.maximum(0, np.maximum(dlow, -dhigh)).sum()
        tvhi = min(1., .5*np.maximum(np.abs(dlow), np.abs(dhigh)).sum())
        full_delta = max(r["parity_gap_interval"][1] for r in results)
        operations_pass = depth4_ci[0] >= .8 and each_lo >= .77 and max(errors) <= .05
        summary = {
            "neurons": n, "candidate": candidate, "static": results[0],
            "static_point": point[0],
            "full_gate_interface_parity_gap_upper": full_delta,
            "full_gate_interface_parity_gap_lower": max(r["parity_gap_interval"][0] for r in results),
            "static_success_and_full_interface_parity_pass":
                results[0]["success_interval"][0] >= .8 and full_delta <= .02,
            "depth4_success_interval": depth4_ci,
            "depth4_each_word_min_success_lower": each_lo,
            "maximum_composition_error_upper": max(errors),
            "operations_pass": operations_pass,
            "joint_readout_parity_tv_interval": [float(tvlo), float(tvhi)],
            "joint_optimal_parity_success_interval": [.5+.5*float(tvlo), .5+.5*float(tvhi)],
            "dt_sensitivity": dt_results,
            "words": [{"word": w, "intervals": r, "point": pt}
                      for w, r, pt in zip(words, results, point)],
            "wait_success_intervals": [metrics(low[i], high[i])["success_interval"]
                                       for i in range(4)],
        }
        write(directory/"summary.json", summary); summaries.append(summary)
        write(folder/"budget.json", ledger.to_dict())
        print(f"n={n}: static={results[0]['success_interval']}, depth4={depth4_ci}, "
              f"operation_pass={operations_pass}", flush=True)
    assert ledger.physical_trials == 21840000
    write(folder/"summary.json", {
        "completed_utc": now(), "family_size": family, "alpha": .05,
        "devices": summaries, "ledger": ledger.to_dict(),
        "selection_lock_sha256": lock_hash,
        "biological_resource_established": False,
        "gpt_design_efficiency_comparison_performed": False})
    provenance.update(status="confirmation_complete", completed_utc=now())
    write(output/"provenance.json", provenance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("discovery", "confirmation"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    (discovery if args.stage == "discovery" else confirmation)(args.output)


if __name__ == "__main__":
    main()
