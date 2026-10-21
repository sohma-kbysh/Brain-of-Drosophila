# Phase 1 — 合成系の実装・校正結果

> 以下は初回run1/run2完了時の記録である。その後の手法点検、1A追加確認、1D予測実験は[追加実験の結果](phase1_next_results.md)を参照。本文中の「現在の実装との一致」などは本文作成時点の検証を指し、以後の拡張前のソースは各runのsnapshotに保存されている。

1A–1Cを実装し、独立した設定・seedによる二回の実行を保存した。**即時確率の予測不足、既知dimensionの再構成、意図的shadowの検出は確認できたが、改訂後の1Aのdt同等性gateは未通過である。** このため小規模実connectomeへの移行はまだ推奨しない。全て人工的な古典系であり、生物の文脈性・計算容量・優位の結果ではない。

## 結果

| 段階 | 改訂後の結果 | 限定 |
|---|---|---|
| 1A：予測十分性 | N=20,000でdt=0.1/0.05 msの両方にrank≥3の同時区間証拠 | 即時二値確率一個では、全混合に対するaffineな未来予測に不足。有限標本から厳密な同値準備を認定したわけではない |
| 1A：dt比較 | 全セルの差の絶対値上界の最大は0.03121 | 事前の±0.03内という同等性を未確認。数値差が0.03121あると実証した意味ではない |
| 1B：再構成 | 線形dimension3、規格化dimension2を全shot水準で選択 | この既知toyに対するモデル選択の校正 |
| 1B：反復校正 | 各Nの100反復で100/100が正しいdimensionを選択 | 各成功率の95%区間は約[0.9638,1]。常に成功する保証ではない |
| 1C：adequate | 親の状態span=効果span=pairing rank=4、埋込み可能 | 既知親に対するrelative tomography成立 |
| 1C：鋭いhidden | 親状態span4に対しpairing rank3、商は非埋込み、置換noise閾値0.5 | 親は同じ古典二bit記憶。非古典性の親への帰属を遮断 |
| 1C：弱いhidden | 正方形のまま埋込み可能、noise閾値0 | 非単体形状や測定の隠蔽だけでは文脈性を示さない |

1Aの代表的な小行列式区間は、N=20,000でdt=0.1 msが[-0.02715,-0.01378]、dt=0.05 msが[-0.02422,-0.01114]。どちらも零を含まない。改訂後seedでは200/2,000試行の区間ではrank≥3を確定できず、shot増加の意義が現れた。真の全未来dimensionやLIF親空間のrelative tomographyを確立したとは言わない。

1Bのoracle確率表への再構成RMSEは、N=200で0.022567、2,000で0.006017、20,000で0.002588だった。親は6微視状態だが、許容準備・効果はそれぞれ3次元spanを持ち相互に完全である。3カテゴリーへの圧縮と未来遷移の整合を代数的に検査した。全微視自由度を観測せずにrelative tomographyが成立する正例である。

1Cの鋭いhidden caseでは、N=20,000のPOM成功率の同時下界が0.999548で、非文脈境界0.75を超えた。商に追加したparity混合同値は既知モデルから厳密に導いた。単なるLPのinfeasible表示だけを証拠にはしていない。同じ等式を故意にadequate caseへ誤適用すると、準備等式とデータ自体の不整合として検出した。これを文脈性陽性に数えない。

## 初回の失敗と改訂を残す

初回seed 20260911のrun1では、単純なvalidation誤差最小による次元選択を使った。1Bの100反復での正答率は96%、95%、96%だったが、代表N=20,000では真の3を4と選び、gateが不合格になった。初回結果・計数・コードのsnapshotを削除せず保存した。

改訂v2は、候補間の検証損失差を同時Hoeffding区間で比較し、最良候補より明らかに悪いと示されない最小dimensionを選ぶ。正解dimensionを推定器へ渡していない。変更を固定して別seed 20260912で再実行したところ、上記の1B校正結果を得た。これは初回の問題を受けた方法開発とその別データでの確認であり、当初から同じ方法で成功したようには報告しない。[改訂の導出](phase1_protocol.md)

初回1Aのdt同等性は通過したが、改訂後seedでは上限が3ポイントをわずかに超えた。都合のよいseedのみを採用せず、現状を未確認とする。次は刺激や閾値を変更するのではなく、dt差の精度を目的とする独立の追加shot設計と誤差予算を先に固定する必要がある。0.03121を見て同等性幅を広げない。

## 再現方法と成果物

リポジトリルートから、既存のプロジェクト環境で実行する。新規依存は不要で、CPU1 worker、数値ライブラリのthread数1とする。

```bash
/home/sohma/conda-envs/bioquantumfly/bin/python -m bioquantumfly.operational.phase1 \
  --config configs/operational_phase1.json \
  --output results/operational_gpt/phase1_reproduction

/home/sohma/conda-envs/bioquantumfly/bin/python -m pytest \
  tests/test_operational_phase1.py tests/test_neural.py tests/test_brian2_smoke.py -q
```

出力先が既に存在すれば停止し、過去結果を上書きしない。二回の実行はいずれも数値実験部分約10.2秒で完了した。この値は環境依存で、性能最適化の成果ではない。

上記の追加テストと既存神経関連テストは計42件すべて通過した。実装開始時の既存140ファイルとの比較では、意図したPhase 0文書4件以外に変更はなく、git indexの一致も確認した。両runのソースsnapshotは来歴に記録された全8ファイルのhashと一致し、run2は現在の実装とも一致する。

| 成果物 | 内容 |
|---|---|
| [設定](../../configs/operational_phase1.json) | boundedなN・dt・seed・判定方式 |
| [統計処理](../../src/bioquantumfly/operational/statistics.py) | 二項同時区間、小行列式証拠、SVDと次元選択 |
| [既知toy](../../src/bioquantumfly/operational/toys.py) | 親状態・効果、圧縮、厳密rank、relative tomography |
| [埋込み検査](../../src/bioquantumfly/operational/embedding.py) | 指定二値シナリオのLP、POM境界、置換noise |
| [LIF実験](../../src/bioquantumfly/operational/lif.py) | 独立二細胞試行と準備から未来への連続実行 |
| [ランナー](../../src/bioquantumfly/operational/phase1.py) | 三段階の実行、来歴、CSV/JSON/NPZ、PNG/PDF |
| [テスト](../../tests/test_operational_phase1.py) | exact正負例、統計被覆、圧縮、履歴保持、正しいdimが異なる対照 |
| [run1報告](../../results/operational_gpt/phase1_run1/report.md) | 初回の失敗を含む結果 |
| [run2報告](../../results/operational_gpt/phase1_run2/report.md) | 改訂後の全結果 |
| [run2数値](../../results/operational_gpt/phase1_run2/summary.json) | S/E、計数、confidence bounds、LP witness、校正率 |
| [run2来歴](../../results/operational_gpt/phase1_run2/provenance.json) | 設定・ソースhash・環境版・実行情報 |
| [run2図](../../results/operational_gpt/phase1_run2/phase1_overview.pdf) | 共有可能な静的研究図。元数値は同じ出力先のCSV/JSON |

各runは `source_snapshot/` と `config.json` を持ち、改訂前のソースも実行時hashとの一致を検査した。`results/` は既存リポジトリ方針によりgitで原則無視されるローカル生成物である。コード・設定・この要約から再生成できるが、共有時は必要なrunディレクトリを別途同梱する。

既存量子・暗号コード、神経本番API、gitのindexは変更していない。Phase 0文書には研究の優先順位と現在の実装への参照を追加し、当時の結論を履歴として保持した。MaleCNSデータ取得、実connectome実験、言語モデル・暗号・量子結合の追加は行っていない。
