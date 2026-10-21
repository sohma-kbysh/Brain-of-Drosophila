# 07 — 新規性の主張を制限する行列

評価日: 2026-09-11。[先行研究とアクセス範囲](06_prior_work.md)を前提とする。`plausibly novel` は今後検証する組合せの評価であり、完成した貢献の認定ではない。

| 主張候補 | 状態 | 根拠と必要な修正 |
|---|---|---|
| connectome を reservoir として使う | `already known` | [Suárez 2024](https://doi.org/10.1038/s41467-024-44900-4)。 |
| Drosophila connectome で時系列予測する | `already known` | [Costi 2025](https://doi.org/10.3390/biomimetics10050341)。MaleCNS への置換だけでは十分な概念的新規性にならない。 |
| connectome と biologically constrained LIF の組合せ | `already known` | [Shiu 2024](https://doi.org/10.1038/s41586-024-07763-9)。MaleCNS 移植の妥当性は別の検証。 |
| 神経ネットワークを ordered state/effect space と instruments で GPT 表現する | `already known` | [Khrennikov 2025](https://doi.org/10.1016/j.jmp.2025.102923) が直接先行。 |
| 未来の実験結果の予測を operational state にする | `already known` | [PSR / Singh 2004](https://arxiv.org/abs/1207.4167)。履歴の予測構造に computational mechanics の関連もある。 |
| $D=SE$ から theory-independent に状態と効果を再構成 | `already known` | [Mazurek 2021](https://doi.org/10.1103/PRXQuantum.2.020302)。 |
| dynamic GPT tomography と文脈性の変化を測る | `already known` | [Aloy 2026](https://doi.org/10.1038/s41467-026-69030-x)。 |
| accessible GPT fragment の simplex embedding を LP 判定 | `already known` | [Schmid 2021](https://doi.org/10.1103/PRXQuantum.2.010331)、[Selby 2023](https://doi.org/10.1103/PhysRevA.107.062203)、[Selby 2024](https://doi.org/10.1103/PhysRevLett.132.050202)。 |
| coarse-graining 下の classical simulation と非文脈 embedding の区別 | `already known` | [Müller–Garner 2023](https://doi.org/10.1103/PhysRevX.13.041001)、[Schmid 2025](https://doi.org/10.22331/q-2025-10-13-1880)。新定理と呼ばず、神経 interface の条件を明示する。 |
| neural manifold の幾何と computation の関係 | `already known` | [Cohen 2020](https://doi.org/10.1038/s41467-020-14578-5)。 |
| 実測 connectome から memory capacity を測る | `already known` | Suárez 2024、[Mijalkov 2025](https://doi.org/10.1038/s41467-025-57995-0)。 |
| topology、weight distribution、局所 dynamics、I/O を分離する | `already known` | conn2res と Costi に関連する分解・対照がある。厳密な本計画の factorial design は仕様上の選択。 |
| generalized contextuality が指定した情報処理課題の資源となる | `already known` | [Spekkens 2009 POM](https://doi.org/10.1103/PhysRevLett.102.010401)。任意の reservoir capacity に一般化できない。 |
| 生物 neural substrate で operational contextuality を問う提案 | `already known` | Sulis 2021 や [Staley 2026 preprint](https://www.preprints.org/manuscript/202605.0579) などに近接提案。ただし contextuality の定義・検査を分ける。 |
| 明示的な古典神経 dynamics に、予測状態、state/effect tomography、埋込解析を接続する | `known separately but not connected` | 各要素は既知。本レビューでは同じ神経実験仕様での接続を確認できなかった。この状態名は確認した corpus に限定する。 |
| measured connectome + biological dynamics + prepare–transform–measure + GPT reconstruction + simplex test + capacity の一貫した比較 | `plausibly novel` | 六要素を満たす直接研究は本調査では未確認。一般理論の新規性より、統制された統合研究としての候補。 |
| 上記六要素の組合せは全先行研究に存在しない | `uncertain — needs literature search` | 限定探索では不存在を証明できない。用語差、未索引論文、後続 preprint をさらに調べる。 |
| MaleCNS の operational geometry が matched reservoirs より特別である | `uncertain — needs literature search` | 文献探索だけでなく事前登録した比較実験が必要。方向も未確定。 |
| contextuality robustness が容量を因果的に高める | `uncertain — needs literature search` | 神経系での結果は未取得。task、free class、cost model、介入、交絡制御を必要とする。 |
| 状態図が non-simplex、または変換が非可換なので文脈的である | `incorrect / not defensible` | 効果・同値関係が不足。古典 stochastic instruments でも順序効果がある。 |
| LIF シミュレータが contextual なら生体は物理的に非古典 | `incorrect / not defensible` | シミュレーションは古典モデルを仮定しており、基礎物理の否定を導けない。 |
| rich geometry、高 rank、良い予測性能だけで量子／計算優位性がある | `incorrect / not defensible` | 競合クラス、コスト、task 制約および制御比較がない。 |

## 六要素の交差をどこまで調べたか

○ = 明示的に対象。△ = 近接した要素または限定的対象。— = 確認した研究では対象外。ここで「capacity」は通常の汎用的な neural computation 指標／task benchmark を指し、解剖学的な性差や qubit decoherence 自体を含めない。

| 研究 | 実測 connectome | 生物制約 dynamics | prepare–transform–measure | データ GPT 再構成 | simplex/contextuality | neural capacity |
|---|---|---|---|---|---|---|
| Schmid 2021 / Selby 2023, 2024 | — | — | △ 理論 | — | ○ | — |
| Mazurek 2021 | — | — | △ 光学 prepare–measure | ○ | ○ | — |
| Aloy 2026 | — | — | ○ 量子デバイス | ○ | ○ | — |
| Khrennikov 2025 | — | △ 抽象神経通信 | ○ instruments | — | △ 別種の効果 | — |
| Shiu 2024 | ○ | ○ | △ 生体刺激・readout | — | — | △ sensorimotor 検証 |
| Suárez 2024 | ○ | △ 選べる局所模型 | △ task I/O | — | — | ○ |
| Costi 2025 | ○ | △ ESN | △ task I/O | — | — | ○ |
| Mijalkov 2025 | ○ | △ reservoir | △ task I/O | — | — | ○ |
| Berg 2026 | ○ | — | — | — | — | — |
| Staley 2026 preprint | — | △ 培養神経を想定 | △ 理論仕様 | — | △ CbD 等の提案 | — |
| 本計画の将来の統合仕様 | 計画 | 計画 | 定義 | 仕様 | 仕様 | 仕様 |

本計画の最終行を「実施済み」の丸印で埋めない。Phase 0 が達成するのは定義と反証条件である。

## 現時点で使える文章

> 本研究は、実測 connectome と明示的な古典神経 dynamics を、制約された準備・介入・観測に対する operational theory として扱う研究計画を定式化する。既存の predictive state representation、GPT tomography、simplex-embeddability 理論および reservoir capacity 解析を組み合わせ、状態・効果の確率幾何が通常の reservoir 特徴量を超えて sequence computation を説明するかを問う。この六要素を一貫して扱う実証研究は本レビューの確認範囲では見つからなかったが、神経系への GPT 導入自体の先行研究は存在し、方法・現象の新規性は未確定である。

## 何が既知の方法の言い換えに終わるか

もし再構成量が通常の reservoir の可観測・可到達 dimension や線形 memory capacity と同じ情報しか持たず、制御比較で追加の予測力がないなら、GPT の呼称を付けることは説明上の貢献にならない。もし文脈性判定が観測を恣意的に隠したことだけで決まり、生物学的に正当化した interface 間で安定しないなら、connectome 固有の性質とも言えない。

逆に、正当化した複数 interface と matched controls の下で gauge に依存しない operational 量が容量の差を安定して予測し、介入でも事前指定した関係が保たれるなら、既知の各分野の間の接続として貢献候補が残る。これは現在の結果ではなく、[仮説](08_hypotheses.md)と[実験仕様](09_future_experiment_spec.md)で反証すべき条件である。simplex-embeddable という結果でも、dimension・discriminability・予測圧縮に説明力があれば研究価値はあるが、contextuality を資源とする主張は棄却する。
