# Phase 1追加実験 — 手法点検と実行結果

**手法の解釈と評価設計を修正したうえで、1A追加確認と1D予測圧縮pilotを実装・実行し、どちらも事前基準を通過した。** 1Aで未確認だった時間刻み間の確率安定性を確認でき、1Dでは短い未来テストから得た4座標の状態と入力更新によって、別の履歴で4入力先の発火確率を予測できた。

これは合成二ニューロン回路の有限予測実験である。生物の非古典資源、真の最小次元、一般GPTの再構成、計算優位を示した結果ではない。

## 手法の確認で修正した点

[手法点検](phase1_method_review.md)と[実行前仕様](phase1_next_protocol.md)に根拠文献と導出を記した。主な修正は次の通り。

- **dt比較の意味を限定。** Brian2の入力離散化も含む終端確率の比較とし、連続時間極限や同じ入力軌道の強収束とは区別した。
- **未来予測に更新を導入。** SVDだけで未知の効果を予測する構成を避け、短い列のデータで0/1入力ごとの更新を推定し、長い列をその合成で予測した。
- **最終評価を分離。** 学習28履歴、検証16履歴、評価16履歴を分け、検証で選択したモデルと最終予測を保存してから評価ラベルを取得した。
- **圧縮とGPT認定を区別。** 確率の範囲外をclipで隠さず、有限予測の誤差を評価した。LIF親の相対的完全性や大域的な状態・効果・更新の妥当性は未確立のままとした。

## 1A追加確認

旧run1/run2とは独立した新データを使った。各dtの4準備×3未来に各10万試行、計360万試行ペアを取得した。36確率を同時95%で覆う二項区間から全dt比較を評価した。

| dt比較（ms） | 差の最大推定値 | 差の絶対値の最大信頼上限 | 事前幅±3ポイント以内 |
|---|---:|---:|---|
| 0.1 / 0.05 | 0.576ポイント | 1.485ポイント | 通過 |
| 0.1 / 0.025 | 0.539ポイント | 1.355ポイント | 通過 |
| 0.05 / 0.025 | 0.425ポイント | 1.352ポイント | 通過 |

全dtでunit付きrank≥3の区間証拠も得た。したがって、指定した混合準備と未来に対し、即時発火確率一個によるaffine予測の不足を再確認した。旧結果の「差の上限3.12ポイント」は削除せず、新しい固定条件の確認実験として報告する。

これは最大32 msの指定プロトコルの安定性であり、後続の全時間範囲への保証ではない。数値実験時間は約84.7秒だった。

## 1D予測圧縮

学習用の準備履歴は長さ2～4、検証・評価履歴は長さ5とした。状態取得用の7個の短い未来テストと、その前に0または1を追加したテストから、状態と入力更新を学習した。検証・評価では `0000,0101,1010,1111` の4列を使った。これらの長い列のラベルは更新行列の回帰には使っていないが、検証履歴のラベルはモデル選択に使った。

検証で選択されたのは **4座標、unit込み線形次元5、ridge=10^-6** のモデルだった。これは7個のanchor確率からの圧縮であり、微視的変数からの圧縮率ではない。

最終評価は16履歴×4未来の64確率で行った。下表の信頼上限は、独立の評価二項計数を同時95%で覆う領域から得た。観測頻度へのRMSEと、真の確率への誤差の上限を区別している。

| 評価項目 | 選択した4座標モデル | 即時確率だけの更新モデル | 4座標モデルの事前基準 |
|---|---:|---:|---|
| 観測頻度へのRMSE | 0.007885 | 0.036912 | 参考値 |
| 真の確率へのRMSEの信頼上限 | **0.018478** | 0.047842 | ≤0.03：通過 |
| 最大確率誤差の信頼上限 | **0.038540** | 0.076941 | ≤0.05：通過 |
| 範囲外予測 | 0件 | 0件 | 0件：通過 |

即時モデルから4座標モデルへの期待Brier損失の改善区間は **[0.0004613, 0.0021347]** で、零を含まなかった。座標数4<7も満たし、有限予測圧縮pilotの全基準を通過した。

入力末尾2binの簡易比較モデルに対しても改善区間は正だった。ただし、これは弱い特定の比較モデルであり、その大きな差を一般的な計算優位の根拠としない。各モデルの生のBrier score、全候補、予測、計数は実行記録に保存した。

**4が最小次元と判明したわけではない。** 最終評価の曲線には、選択した4座標より誤差が小さい3座標候補もあった。評価後に選び直さず、検証で固定した4座標の結果を主結果として保持した。全候補の曲線を開示しており、低い次元でも指定誤差を満たし得る点を隠していない。

計900万試行ペア、数値実験時間約581.5秒だった。新しい準備のanchor取得には7テスト×1万=**7万試行**を使う。今回の結果は、その状態推定を行った集団についての予測であり、反復測定不要の一試行オンライン計算ではない。

## 到達点と次の判断

二ニューロンLIFについて「即時確率では不足する」に加え、「短い未来テストの予測から構成した状態を更新すると、別履歴で長い未来を指定精度で予測できる」という正例が得られた。研究の中心を予測圧縮に置く方針を支持する、限定された実装上の証拠である。

残る条件は具体的である。

1. **1Dの時間範囲の数値安定性。** 今回の1Dはdt=0.05 ms、最大92 msのモデルに対する結果。1Aの最大32 msの確認を外挿せず、使用した長いプロトコルについても確認する必要がある。
2. **予測範囲の拡張。** 選択したモデルを固定し、別の入力列・予測時間・anchor再測定で誤差を検査する。有限64確率での成功を全未来十分性へ拡張しない。
3. **容量との関係。** 更新モデル、単なる次元、状態–効果幾何の追加説明力を区別する。測定回数と読出し予算を揃える。今回、容量や文脈性の新しい神経実験は行っていない。

直後の研究作業は、固定モデルの時間刻み・予測範囲の拡張検証を設計すること。実connectome候補の設計検討は可能だが、今回の結果だけを理由に生物の操作的理論を同定済みとはしない。MaleCNSのデータ取得や大規模実行には進んでいない。

## 検証と再現

追加・既存神経関連の**計51テストが通過**した。既知の3状態古典制御系に対する未学習列の再現、過小次元の負例、入力順序、unit保存、混合・gauge不変性、区間誤差、バッチ集計、履歴分割を検査した。拡張したLIFと旧run2の保存ソースは、同一seedの8条件で試行ごとの出力が完全一致した。

両実行のソースsnapshotは各11ファイルのhashと一致する。全1,260万試行のNPZ、CSV計数、バッチ記録を照合し、952バッチのseedに重複がないことを確認した。最終評価前のモデル・予測固定記録も検証した。各出力先の`audit.json`に記録している。

```bash
/home/sohma/conda-envs/bioquantumfly/bin/python -m bioquantumfly.operational.phase1_next \
  confirmation --output results/operational_gpt/phase1a_confirmation_reproduction

/home/sohma/conda-envs/bioquantumfly/bin/python -m bioquantumfly.operational.phase1_next \
  prediction \
  --confirmation results/operational_gpt/phase1a_confirmation_reproduction \
  --output results/operational_gpt/phase1d_prediction_reproduction

/home/sohma/conda-envs/bioquantumfly/bin/python -m pytest \
  tests/test_operational_prediction.py tests/test_operational_phase1.py \
  tests/test_neural.py tests/test_brian2_smoke.py -q
```

出力先の上書きは禁止し、predictionは一致する設定とシミュレータで完了した確認実験の通過を要求する。CPU1 worker、数値thread1、既存環境の依存だけを使った。`results/`は既存git方針で無視されるローカル生成物なので、共有時は必要な実行ディレクトリも同梱する。

| 成果物 | リンク |
|---|---|
| 設定・実装 | [設定](../../configs/operational_phase1_next.json)、[予測モデル](../../src/bioquantumfly/operational/predictive.py)、[実験ランナー](../../src/bioquantumfly/operational/phase1_next.py) |
| 1A記録 | [報告](../../results/operational_gpt/phase1a_confirmation_v1/report.md)、[数値](../../results/operational_gpt/phase1a_confirmation_v1/summary.json)、[図PDF](../../results/operational_gpt/phase1a_confirmation_v1/overview.pdf) |
| 1D記録 | [報告](../../results/operational_gpt/phase1d_prediction_v1/report.md)、[数値](../../results/operational_gpt/phase1d_prediction_v1/summary.json)、[図PDF](../../results/operational_gpt/phase1d_prediction_v1/overview.pdf) |
| 予測の固定 | [モデル・予測hashと時刻](../../results/operational_gpt/phase1d_prediction_v1/prediction_lock.json) |
| 検証 | [追加テスト](../../tests/test_operational_prediction.py)、[1A監査](../../results/operational_gpt/phase1a_confirmation_v1/audit.json)、[1D監査](../../results/operational_gpt/phase1d_prediction_v1/audit.json) |

既存量子・暗号コード、神経本番API、git indexは維持した。既存の操作的LIFモジュールは短い旧APIを保ったまま長い入力列へ拡張し、旧結果の文書には本報告への参照だけを追記した。
