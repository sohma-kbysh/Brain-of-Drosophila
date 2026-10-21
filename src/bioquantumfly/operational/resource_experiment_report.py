"""Read-only audit of registered outcomes, then reproducible summary artifacts."""
import os
os.environ["MPLBACKEND"] = "Agg"
import argparse
from collections import defaultdict
from itertools import product
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(output):
    if read(output/"provenance.json")["status"] != "confirmation_complete":
        raise RuntimeError("confirmation is not complete")
    seeds, artifacts, totals = set(), [], defaultdict(float)
    groups = {"search": defaultdict(float), "confirmation": defaultdict(float)}
    by_device = defaultdict(lambda: defaultdict(float))
    cell_count = 0
    for log in sorted(output.rglob("cells.jsonl")):
        group = "confirmation" if "confirmation" in log.parts else "search"
        by_file = defaultdict(list)
        for line in log.read_text().splitlines():
            record = json.loads(line)
            if record["seed"] in seeds:
                raise AssertionError("duplicate cell seed")
            seeds.add(record["seed"]); by_file[record["artifact"]].append(record)
        for filename, records in by_file.items():
            path = log.parent/filename
            with np.load(path) as a:
                outcome = a["outcome"]
                # NpzFile does not cache decompression: load each array once.
                activity, background, drive = a["activity"], a["background"], a["drive"]
                all_words = a["words"].tolist() if "words" in a else None
                expected = len(records)*records[0]["trials"]
                assert outcome.size == expected
                for r in records:
                    idx = ((all_words.index(r["word"]), r["preparation"], r["query"])
                           if all_words is not None else (r["preparation"],))
                    o = outcome[idx]
                    assert len(o) == r["trials"]
                    if r["query"] == 2:
                        assert np.bincount(o, minlength=4).tolist() == r["counts"]
                    else:
                        assert np.all(o <= 1) and int(o.sum()) == r["count"]
                    assert int(activity[idx].sum()) == r["spikes"]
                    assert int(background[idx].sum()) == r["noise_events"]
                    assert np.isclose(drive[idx].sum(), r["absolute_drive_neuron_ms"],
                                      rtol=1e-12, atol=1e-8)
                    costs = {
                        "trials": r["trials"], "spikes": r["spikes"],
                        "noise_events": r["noise_events"],
                        "drive": r["absolute_drive_neuron_ms"],
                        "neuron_time_ms": r["trials"]*r["neurons"]*r["duration_ms"],
                        "readout_bits": r["trials"]*r["readout_bits_per_trial"],
                        "readout_neuron_channels": r["trials"]*(r["neurons"]//2)
                                                   *r["readout_bits_per_trial"],
                        "kernel_seconds": r["simulator_seconds"]}
                    for key, value in costs.items():
                        totals[key] += value; groups[group][key] += value
                        by_device[(group, r["neurons"])][key] += value
                    cell_count += 1
            artifacts.append({"path": str(path.relative_to(output)), "sha256": sha(path),
                              "trials": expected, "bytes": path.stat().st_size})
    discovery = read(output/"discovery_summary.json")
    confirmation = read(output/"confirmation/summary.json")
    assert groups["search"]["trials"] == discovery["total_trials"] == 7488000
    assert groups["confirmation"]["trials"] == confirmation["ledger"]["physical_trials"] == 21840000
    assert totals["trials"] == 29328000
    assert cell_count == len(seeds) == 25572
    fields = {"spikes": "emitted_spikes", "noise_events": "external_noise_events",
              "drive": "absolute_drive_neuron_ms", "readout_bits": "readout_bits",
              "neuron_time_ms": "neuron_time_ms", "kernel_seconds": "simulator_seconds"}
    for group, ledgers in (
            ("search", discovery["ledgers"]), ("confirmation", [confirmation["ledger"]])):
        for key, ledger_key in fields.items():
            assert np.isclose(groups[group][key], sum(row[ledger_key] for row in ledgers),
                              rtol=1e-10, atol=1e-6), (group, key)
    prov = read(output/"provenance.json")
    for relative, expected_hash in prov["sources"].items():
        assert sha(output/"source_snapshot"/relative) == expected_hash
    lock_hash = sha(output/"selection_lock.json")
    assert lock_hash == prov["selection_lock_sha256"] == confirmation["selection_lock_sha256"]
    result = {
        "status": "passed", "unique_cell_seeds": len(seeds), "sampled_cells": cell_count,
        "artifacts_verified": len(artifacts), "totals": dict(totals),
        "groups": {k: dict(v) for k, v in groups.items()},
        "by_device": [{"stage": k[0], "neurons": k[1], **v} for k, v in sorted(by_device.items())],
        "counts_and_per_trial_cost_arrays_verified": True,
        "snapshot_and_selection_lock_verified": True,
        "missing_physical_costs": ["reset_latency_and_energy", "controller_hardware_energy",
                                   "single_monitor_enforcement", "live_biological_measurements"],
    }
    write(output/"artifact_manifest.json", artifacts)
    write(output/"audit.json", result)
    return result


def make_figure(output):
    import matplotlib.pyplot as plt
    summary = read(output/"confirmation/summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    for device in summary["devices"]:
        n = device["neurons"]
        means, lows, highs = [], [], []
        for depth in range(5):
            entries = [row for row in device["words"] if len(row["word"]) == depth]
            means.append(np.mean([r["point"]["success_interval"][0] for r in entries]))
            interval = np.mean([r["intervals"]["success_interval"] for r in entries], axis=0)
            lows.append(interval[0]); highs.append(interval[1])
        means = np.array(means)
        axes[0].errorbar(range(5), means, yerr=[means-np.array(lows), np.array(highs)-means],
                         marker="o", capsize=3, label=f"{n} neurons")
    axes[0].axhline(.75, color="gray", linestyle="--", label="NC POM reference (conditional)")
    axes[0].axhline(.8, color="black", linestyle=":", label="Engineering target")
    axes[0].set(xlabel="Gate sequence length", ylabel="Bit retrieval success",
                ylim=(.45, 1.03), xticks=range(5), title="Generating a state vs. operating on it")
    axes[0].legend(fontsize=8)
    for offset, key, label, color in (
            (-.14, "selected", "One pool, no gates", "C0"),
            (0., "future", "One pool, with gates", "C2"),
            (.14, "joint", "Both pool bits, no gates", "C3")):
        for index, d in enumerate(summary["devices"]):
            if key == "selected":
                ci = .5+.5*np.array(d["static"]["parity_gap_interval"])
            elif key == "future":
                ci = .5+.5*np.array([d["full_gate_interface_parity_gap_lower"],
                                     d["full_gate_interface_parity_gap_upper"]])
            else:
                ci = np.array(d["joint_optimal_parity_success_interval"])
            axes[1].errorbar(index+offset, ci.mean(), yerr=[[ci.mean()-ci[0]], [ci[1]-ci.mean()]],
                             marker="o", capsize=4, color=color,
                             label=label if index == 0 else None)
    axes[1].set(xticks=range(3), xticklabels=["2", "8", "16"], xlabel="Neurons",
                ylabel="Optimal parity-guessing success", ylim=(.45, 1.03),
                title="Opening the observation interface")
    axes[1].axhline(.5, linestyle=":", color="gray")
    axes[1].legend(fontsize=9)
    fig.suptitle("Synthetic classical LIF: fixed-sample confirmation, simultaneous 95% intervals")
    fig.savefig(output/"resource_results.png", dpi=180)
    fig.savefig(output/"resource_results.pdf")
    plt.close(fig)


def summary_table(output):
    summary = read(output/"confirmation/summary.json")
    rows = []
    for d in summary["devices"]:
        base = output/"confirmation"/f"n{d['neurons']}"
        with np.load(base/"words.npz") as data:
            words = data["words"].tolist()
            counts = data["outcome"].sum(-1)
            primary_costs = {}
            for word in ("", "FWFW"):
                idx = words.index(word)
                primary_costs[word or "static"] = {
                    "spikes_per_use": float(data["activity"][idx].mean()),
                    "drive_per_use": float(data["drive"][idx].mean()),
                    "background_events_per_use": float(data["background"][idx].mean()),
                    "nominal_latency_ms_excluding_reset": 19+10*len(word),
                    "monitored_neuron_channels": d["neurons"]//2,
                    "output_bits": 1, "copies": 1}
        rows.append({
            "neurons": d["neurons"], "static_success": d["static"]["success_interval"],
            "static_parity_gap": d["static"]["parity_gap_interval"],
            "static_candidate_pass": d["static"]["finite_interface_candidate_pass"],
            "full_interface_parity_gap": [d["full_gate_interface_parity_gap_lower"],
                                           d["full_gate_interface_parity_gap_upper"]],
            "depth4_success": d["depth4_success_interval"],
            "operations_pass": d["operations_pass"],
            "composition_error_upper": d["maximum_composition_error_upper"],
            "joint_parity_success": d["joint_optimal_parity_success_interval"],
            "ontic_parity_tv_lower": d["static"]["required_ontic_parity_tv_lower"],
            "witness_replacement_noise": d["static"]["conditional_witness_noise_interval"],
            "costs": primary_costs,
            "design_parameter_float64_payload_bytes": 17*8,
            "search_trials_including_all_five_seeds": 2496000,
            "dt_sensitivity": d["dt_sensitivity"],
            "wait_success": d["wait_success_intervals"]})
    write(output/"compact_results.json", rows)
    return rows


def unit_minor_interval(lower, upper):
    """Exhaust all 4096 corners of a 4-row, 3-effect confidence box."""
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    if (lo.shape != (4, 3) or hi.shape != lo.shape or np.any(lo > hi)
            or not np.all(np.isfinite(lo+hi))):
        raise ValueError("expected a finite 4 by 3 interval table")
    bits = np.array(list(product((False, True), repeat=12)))
    corners = np.where(bits, hi.ravel(), lo.ravel()).reshape(-1, 4, 3)
    matrices = np.concatenate([np.ones((len(corners), 4, 1)), corners], axis=2)
    determinants = np.linalg.det(matrices)
    return [float(determinants.min()), float(determinants.max())]


def rank_diagnostic(output):
    from .resource_experiment import interval_table
    from .statistics import affine_rank_witness
    path = output/"confirmation/n8/words.npz"
    p, lo, hi = interval_table(path, 20000, 1128)
    with np.load(path) as a:
        words = a["words"].tolist()
    i = words.index("WWW")
    determinant = unit_minor_interval(np.c_[lo[0], lo[i, :, 1]],
                                     np.c_[hi[0], hi[i, :, 1]])
    static = affine_rank_witness(lo[0], hi[0])
    result = {
        "status": "post_confirmation_diagnostic_no_new_trials",
        "neurons": 8, "initial_preparations": 4,
        "static_rank_with_unit": 3 if static["certified_rank_at_least_3"] else None,
        "static_minor": static,
        "added_effect": "WWW_then_second_pool_readout",
        "expanded_unit_minor_interval": determinant,
        "expanded_rank_with_unit": 4 if max(determinant[0], -determinant[1]) > 1e-10 else None,
        "confidence_scope": "derived from the same simultaneous 1128-probability box",
        "claim_scope": "four initial preparations and their mixtures, not all neural states",
    }
    write(output/"rank_diagnostic.json", result)
    return result


def make_markdown(output, rows, audited):
    root = Path(__file__).resolve().parents[3]
    relative = output.resolve().relative_to(root)
    link = "../../"+str(relative)
    interval = lambda values: "–".join(f"{100*x:.3f}" for x in values)
    table = "\n".join(
        f"| {r['neurons']} | {interval(r['static_success'])}% | "
        f"{interval(r['static_parity_gap'])} pp | "
        f"{'合格' if r['static_candidate_pass'] else '未達'} | "
        f"{interval(r['depth4_success'])}% | 未達 |" for r in rows)
    opening = "\n".join(
        f"| {r['neurons']} | {interval(r['full_interface_parity_gap'])} pp | "
        f"{interval(r['joint_parity_success'])}% |" for r in rows)
    cost_table = "\n".join(
        f"| {r['neurons']} | {r['costs']['static']['spikes_per_use']:.4f} | "
        f"{r['costs']['static']['drive_per_use']:.3f} | "
        f"{r['costs']['static']['monitored_neuron_channels']} | 19 ms |" for r in rows)
    wait_table = "\n".join(
        f"| {r['neurons']} | "+ " | ".join(interval(ci)+"%" for ci in r["wait_success"])+" |"
        for r in rows)
    dt_table = "\n".join(
        f"| {r['neurons']} | {dt} | {v['maximum_point_difference']*100:.3f} pp | "
        f"{v['maximum_probability_difference_upper']*100:.3f} pp | "
        f"{'合格' if v['maximum_probability_difference_upper'] <= .01 else '未確定'} |"
        for r in rows for dt, v in r["dt_sensitivity"].items())
    resource_table = "\n".join(
        f"| {r['neurons']} | {r['ontic_parity_tv_lower']:.6f} | "
        f"{interval(r['witness_replacement_noise'])}% |" for r in rows)
    summary = read(output/"confirmation/summary.json")
    worst = max(summary["devices"][1]["words"],
                key=lambda r: r["intervals"]["parity_gap_interval"][0])
    rank = read(output/"rank_diagnostic.json")
    text = f"""# 1E・初回2Aの実験結果：状態生成は成功、演算利用は未達

2026-09-13。[実行前仕様](resource_experiment_v1_protocol.md)に従った共通ランダム探索と独立確認を完了した。**8・16ニューロンの合成古典LIFで、指定された静的interfaceのGPT-likeな候補特性を生成できた。ただし今回の刺激・探索では操作して使える資源にできず、効率的な生物演算資源の構成は未達である。**

実測コネクトームや生体での結果ではない。GPTによる設計と通常の最適化の比較も未実施であり、本結果をその比較の成功・失敗として解釈しない。親計画の45 runの3方式比較に先立つ、15 runの共通対照である。

## 試行と独立性

探索7,488,000試行、独立確認21,840,000試行、合計**29,328,000試行**。探索は各サイズ5 seed、各run499,200試行で上限500,000を守った。サイズごとに検証データから一つの装置を選び、selection lockを保存してから確認を開始した。

確認では各cell20,000試行を固定し、途中の成績による増減を行っていない。**20,000は成功率を測るための反復数であり、装置が一回の答えを出すのに20,000コピーを使う方式ではない。** 主方式は1準備から選択された1ビットを読む。joint対照も1準備から取得するが、両poolの2ビットを読む。

以下の区間は1,128確率を同時95%で覆う領域から計算した。実装時間の測定用パイロット6,000試行とunit testは、上記の科学的な探索・確認データに含めず、候補選択にも再利用していない。

## 主結果

| ニューロン数 | 静的な成功率の区間 | 静的parity差の区間 | 静的候補基準 | 4段操作の平均成功率 | 操作基準 |
|---|---:|---:|---|---:|---|
{table}

静的候補基準は成功率の下限80%以上、指定効果のparity差の上限2ポイント以下。操作基準は4段平均の下限80%以上、各列下限77%以上、応答誤差上限5ポイント以下。2ニューロンは静的成功率を満たしたが、parity差の上限が基準内に収まらなかった。「parity差が必ず2ポイントを超える」と確定したわけではない。

8・16ニューロンの静的成功率は理想qubitのPOM参照値約85.36%も超える。しかし、これは制限された観測で2ビット古典記憶が作るshadowと整合する。量子的・物理的な非古典性や、古典計算に対する優位の証拠とはしない。[POMの参照値](https://arxiv.org/pdf/0805.1463)

![生成・操作・観測を広げた場合の比較]({link}/resource_results.png)

[図のPDF]({link}/resource_results.pdf)、[数値の要約]({link}/compact_results.json)。

## 観測と操作を広げるとどうなるか

| ニューロン数 | 全31操作列を含むparity差 | 静的な両pool読出しでparityを当てる成功率 |
|---|---:|---:|
{opening}

8ニューロンでは、操作列 **{worst['word']}** を追加した効果だけでもparity漏れが大きくなる。したがって、静的な二つの読出しで近似的に成立する準備同値性を、操作を含む理論へそのまま延長できない。

8・16ニューロンでは、両poolの出力を読むとparityをほぼ完全に推定できる。静的な高得点は、隠れた古典情報を持つモデルによって説明可能である。単一monitor制限の強制装置と、その費用は今回実装していない。

確認後の追加解析では、8ニューロンの確率表について、単位列を含む線形rankが静的な3から、WWW後の第二pool読出しを加えると4になることを確認した。4×4 minorの区間は[{rank['expanded_unit_minor_interval'][0]:.6f}, {rank['expanded_unit_minor_interval'][1]:.6f}]で零を含まない。対応する規格化状態のaffine次元は2から3へ増える。

この解析は事後に選んだ診断だが、新規データは使わず、元の同時信頼領域内の4,096隅を調べた。初期4準備の混合に限れば、追加効果によって四つの頂点がaffine独立になる。全神経状態や変換の閉性まで同定したという意味ではない。[rank診断の数値]({link}/rank_diagnostic.json)

## 失敗の切り分け：保持時間も不足している

同じ準備の後、刺激操作を入れずに待つ対照は次の通り。追加待機は基礎手続き19 msに加わる。

| ニューロン数 | 追加10 ms | 追加20 ms | 追加30 ms | 追加40 ms |
|---|---:|---:|---:|---:|
{wait_table}

操作を行わなくても40 ms後にはほぼ偶然水準まで落ちる。従って、今回の演算失敗を操作器だけに帰せない。静的な成功率とparity差で準備を選ぶ現在の探索では、長く保持できる状態を選ぶ条件が欠けていた。

これは、指定した電流範囲・pool構造・64候補のランダム探索での結果である。神経回路一般で保持や反転・交換が不可能だという証明ではない。

今回の操作器は、固定readoutの下で神経回路へ刺激を加える方式である。外部controllerによる読出しラベルの入替などは別方式であり、今回の反転・交換の成功として数えていない。

## resource estimation

| ニューロン数 | 古典的再現に必要な隠れたparity TVの下限 | 指定witnessを消す出力置換ノイズ |
|---|---:|---:|
{resource_table}

TV下限はmax(0,4S_lower−3)。右列は出力を公平なcoinへ置換した場合のPOM参照witnessの閾値であり、完全なcontextuality robustnessでも、回路が同率の生物学的ノイズに耐えるという意味でもない。近似準備同値性を厳密な同値性と仮定した非文脈性の認定は行っていない。

単発の静的な利用費用は、確認データで次の通り。

| ニューロン数 | 全回路の平均spike数／利用 | 絶対駆動量／利用 | monitorが見るニューロン数 | 公称時間（resetを除く） |
|---|---:|---:|---:|---:|
{cost_table}

絶対駆動量の単位は閾値差で規格化した入力×neuron×msであり、jouleではない。1利用の出力は1ビットだが、poolの多数決に必要な内部観測チャネルを無料の1チャネルと数えない。

各選択装置の探索には、不採用の4 seedも含めて2,496,000試行を使用した。3サイズ全体では7,488,000試行。固定した設計パラメータは準備8・操作8・gain 1の17実数であり、float64配列を選んだ場合の値だけの保存量は136 byte。実機の記憶容量・Pythonの全占有量ではない。

4段操作の公称時間は59 msだが、性能が基準に達しないため、「成功した演算1回の効率」として比較できない。2ビット古典方式の解析対照は静的・全31操作列で成功率1を達成する。実機エネルギー、reset時間、観測制限の強制費用、外部controller費用には未測定項目があるため、生体対電子回路のエネルギー優位は未判定である。

## 数値感度

dt=0.1 msを基準に、事前指定の5列を比較した。

| ニューロン数 | 比較dt（ms） | 観測された最大確率差 | 最大差の信頼上限 | 1ポイント以内の同等性 |
|---|---:|---:|---:|---|
{dt_table}

信頼上限を観測された差と混同しない。この監査は5列だけを対象としており、合格した場合も全31列の数値安定性を保証しない。本結果はまず指定の離散LIFモデルに対するものとして報告する。

## 検証と保存物

数理基準、parity漏れ、試行予算、queryによる内部発展の不変性、操作順の真理値表、rank区間、入力範囲、独立Brian2モデルとの照合を含む23項目のtestが通過した。Brian2との比較はノイズなし4条件であり、全確率分布の一致を保証したという意味ではない。

元outcomeとtrialごとの費用配列を全て保存し、**{audited['artifacts_verified']:,}個のNPZ、{audited['sampled_cells']:,}個のcell・相異なるseed、29,328,000個のoutcome**について、保存count・spike・背景event・駆動量・ledgerと照合した。source snapshotとselection lockのhashも確認した。

- [元データと実行記録]({link}/provenance.json)
- [探索後の固定候補]({link}/selection_lock.json)
- [確認結果の全数値]({link}/confirmation/summary.json)
- [費用と元データの監査]({link}/audit.json)
- [全データファイルのhash]({link}/artifact_manifest.json)
- [実装](../../src/bioquantumfly/operational/resource_experiment.py)
- [独立した集計・監査コード](../../src/bioquantumfly/operational/resource_experiment_report.py)

## 次に実施する比較

次の版では、準備選択の時点から保持時間と操作可能性を目的関数に含める。GPTの状態–効果–変換を使う設計と、同じ情報・費用を持つ通常設計を比較し、静的な正解率の高い状態を選ぶだけで終わらないかを検証する。

今回の確認データは次の版の探索を考える材料には使えるが、新しい設計の確認データとして再利用しない。生物由来の資源を効率的に生成できるという最終結論には、設計方式の比較、実測結線への移行、実機費用と生体での確認が残る。
"""
    report = root/"docs/operational_gpt/resource_experiment_v1_results.md"
    report.write_text(text)
    write(output/"report_provenance.json", {
        "reporter_source_sha256": sha(__file__),
        "report_sha256": sha(report), "report": str(report.relative_to(root)),
        "input_summary_sha256": sha(output/"confirmation/summary.json"),
        "audit_sha256": sha(output/"audit.json")})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.output)
    rows = summary_table(args.output)
    make_figure(args.output)
    rank_diagnostic(args.output)
    make_markdown(args.output, rows, result)
    print(json.dumps({"audit": result, "results": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
