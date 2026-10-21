# Operational Geometry of Computation in a Biological Connectome

**仮題・構成案。論文貢献は未確定。** Phase 0 の数理整理を基に、将来、十分な証拠が得られた場合の構成を示す。神経の量子性、MaleCNS の一般化文脈性、計算上の優位を題名や abstract に先取りしない。

## Abstract の骨格

明示的に古典的な神経動力学について、準備・変換・測定を制限した操作的状態–効果 fragment を定義する。未来を含む同値性、親理論と shadow の区別、有限次元推定と統計的不確実性を扱う。将来の結果欄には、実際に評価した topology・局所モデル・I/O と課題、埋込みの可否または不定判定、幾何の追加説明力の有無を記す。現在は定義、既知結果の適用、小例、実験前仕様までであり、生物系の実験結果を置かない。

## 1. Introduction

connectome reservoir computing の既存成果から、入力履歴に対する可観測な状態と効果を同時に扱う意義へ進む。問いは「生物学的制約の下の操作的幾何が計算を説明するか」。神経 GPT の隣接研究、PSR、contextuality の既存資源論を紹介し、未接続の要素を明示する。新規性表を圧縮して、初めてでない要素を正確に位置づける。[06](06_prior_work.md)、[07](07_novelty_matrix.md)

## 2. Classical substrate and operational interface

$\lambda,\mu_P,K_\tau$ と $\omega=[P]$ を別々に定義する。許容 future tests、外部混合ラベル、reset、介入時間、terminal measurements/instruments を特定する。全将来同値性から変換が商に降りる短い証明と、有限 horizon の限定を記す。MaleCNS 適用時は解剖学と動力学の出典を分け、単一 black box として扱う。[01](01_operational_formalism.md)、[05](05_connectome_mapping.md)

## 3. Accessible geometry and inference

状態 cone、制限効果、unit、$D=SE$、gauge、rank の定義を置く。sampled fragment と completion、親 fragment と shadow の関係を示す。準備・効果の相対的トモグラフィー完全性をどの仮定から採るかを明記する。理論的に可能な dual effects を実装可能な測定と同一視しない。[02](02_gpt_geometry.md)

## 4. Classicality and no-go constraints

微視古典性、可模擬性、単体状態、単体埋込み、非文脈性の含意関係を整理する。非可換な古典操作、同じ square の弱効果での埋込み、鋭い効果での非埋込みを並べる。任意の多面体 GPT の classical shadow 実現は既知定理として引用し、本論文の新発見とはしない。state–effect 判定から変換全体への飛躍を排除する。[03](03_classicality_and_contextuality.md)

## 5. Sequential computation and candidate resource quantities

controlled state update と instrument による観測後更新を分け、PSR・RNN・reservoir との数学的対応を示す。fixed-readout IPC 上界、operational discriminability、指定 noise robustness を定義する。仮説 H1–H6 と task/free baseline/cost model を列挙し、相関と因果効果を分ける。[04](04_sequence_computation.md)、[08](08_hypotheses.md)

## 6. Methods for a later empirical study

ここは将来段階まで仕様のみとする。トポロジー×局所モデルの対照、I/O の独立選定、未来テストの分割、同時信頼領域と certificate、非有意と同等性の違い、dt/measurement bin sensitivity を示す。最初の合成 pilot と、その後の実測部分回路の gate を区別する。[09](09_future_experiment_spec.md)

## 7. Results：将来埋める欄

| 必要な証拠 | 掲載すべき結果 | 現在の状態 |
|---|---|---|
| 小さな数理例 | 有理数による embedding / bound / noncommutativity 検算 | Phase 0 で実行可能、経験的結果ではない |
| 統計校正 | 既知 null の偽陽性率と検出力 | 未実施 |
| 状態同定 | dimension の信頼範囲、未使用未来での適合/失敗 | 未実施 |
| 生物的 interface | 解剖・生理・観測制約の事前根拠 | 未選定 |
| 単体判定 | 証明済み等式と feasible/robust infeasible/不定の証明書 | 神経系では未実施 |
| 計算との関係 | 対照と costs を揃えた追加予測力、または null | 未実施 |
| 因果主張 | geometry を分離する介入または明示した同定仮定 | 未確立 |

結果が負でもこの構成を使える。埋込可能ならその witness、未判定なら同定を妨げた条件、追加説明力がなければその同等性境界を主要結果にする。探索した課題の都合のよい一部だけで abstract を構成しない。

## 8. Discussion

`How this research could be conceptually wrong` を中心に、恣意的 shadow、動力学転用の支配、既存 reservoir への還元、計算的無意味さを検討する。観測制約に相対的な理論が何を説明し、何を説明しないかを明記する。生体の物理的非古典性を結論にしない。[10](10_failure_modes.md)

## 図表の候補

| 図表 | 内容 | 必要データ |
|---|---|---|
| 図1 | 微視系→手続き→未来統計→商→state–effect fragment | 定義のみ、現行 Mermaid を利用可 |
| 図2 | 同じ square と弱/鋭い効果、対応する埋込み/不等式 | 厳密な人工例のみ |
| 図3 | future test matrix、rank と有限 horizon の関係 | pilot 後。PCA 図の代用にしない |
| 図4 | topology×local dynamics の factorial 比較 | 将来の独立対照データ |
| 図5 | 幾何指標と性能、通常指標を条件付けた予測 | 将来の held-out 結果。因果性を自動表示しない |
| 表1 | 先行研究・新規性 matrix | 一次文献 |
| 表2 | 仮説、棄却条件、null、未判定の定義 | 事前仕様 |

付録には、商への降下、square の証明、LP 入力等式、gauge 規約、信頼領域の保証、provenance と数値精度を置く。書誌情報は [references.bib](references.bib)、読者向け短縮版は [theory_summary](theory_summary.md) を参照する。
