# Phase 0 — 生物コネクトーム計算の操作的幾何

> **現行優先順位（2026-09-12）:** GPTの視点を使って、生物回路に有効な非古典的演算資源を生成・操作できるか、生成から利用までの費用が有利かを主課題にする。[最新実行計画](nonclassical_resource_execution_plan.md)を優先する。厳密なGPT整合性や親のRTC証明は、資源評価の開始条件にしない。以下はPhase 0と初期Phase 1の歴史的仕様である。

この文書群は、**古典的な神経動力学に対する準備・介入・観測の制限を、一般化確率理論（GenProbTheory）として定義すると、どの情報処理上の違いが説明できるか**を検討する。理論設計段階であり、生体の文脈性、MaleCNS の性能、計算上の優位性を実証したものではない。

Phase 1 では、主課題を**計算に十分な操作的状態の圧縮と、その状態–効果幾何の説明力**に置き、単体埋込み検査を副次課題とした。合成系の実装・校正は [Phase 1仕様](phase1_protocol.md)、[実行結果](phase1_results.md) を参照する。以下のPhase 0記録にある「未実行」は当時の状態であり、Phase 1の合成実験については新文書が現状を示す。

既存の BioQuantumFly の量子・暗号研究とは別の研究方向として共存する。本方向の出発点には量子メモリも量子神経結合も含めない。既存研究の妥当性を本文書群によって再判定することもしない。

## 読み方

| 文書 | 内容 |
|---|---|
| [最新実行計画](nonclassical_resource_execution_plan.md) | 資源生成・操作・効率、GPTによる設計の寄与、対照、探索上限、判断基準 |
| [1E・初回2Aの実験結果](resource_experiment_v1_results.md) | 合成LIFの29,328,000試行：静的候補特性の生成、操作と保持の未達、費用・shadow監査 |
| [resource_priority](resource_priority.md) | GPT-like挙動と資源推定を主評価にする方針 |
| [theory_summary](theory_summary.md) | 10 の問いに対する研究メモと Phase 1 判定 |
| [01_operational_formalism](01_operational_formalism.md) | 古典状態、手続き、未来を含む操作的同値性 |
| [02_gpt_geometry](02_gpt_geometry.md) | 有限次元化の条件、状態・効果、トモグラフィー、単体検査 |
| [03_classicality_and_contextuality](03_classicality_and_contextuality.md) | 古典性の区別、命題、正例・反例、shadow |
| [04_sequence_computation](04_sequence_computation.md) | 逐次計算、容量、資源の必要条件 |
| [05_connectome_mapping](05_connectome_mapping.md) | 既存コードとの対応、トポロジーと局所動力学の分離 |
| [06_prior_work](06_prior_work.md) | 一次文献の比較と検索範囲 |
| [07_novelty_matrix](07_novelty_matrix.md) | 主張ごとの新規性判定 |
| [08_hypotheses](08_hypotheses.md) | 反証可能な仮説と負の結果 |
| [09_future_experiment_spec](09_future_experiment_spec.md) | 事前統計仕様、最小パイロット、Phase 1 gate |
| [10_failure_modes](10_failure_modes.md) | 概念的な失敗と研究中止条件 |
| [glossary](glossary.md) | 用語・記号・主張の許容範囲 |
| [paper_outline](paper_outline.md) | 貢献未確定の論文構成案 |
| [references.bib](references.bib) | 書誌情報 |
| [theory_examples/README](../../theory_examples/README.md) | 小さな数理例の実行方法・適用範囲 |
| [operational_geometry_sanity.py](../../theory_examples/operational_geometry_sanity.py) | 標準ライブラリによる厳密有理数の検算 |

## 確立した事項と未確立事項

**定義**はこの研究が採用する対象・規約、**既知の結果**は引用文献の仮定の下の結果、**命題**はここに証明を示した限定的主張、**推論**はそれらのこの研究への適用、**仮説**は将来棄却できる経験的主張を意味する。候補新規性は文献探索に依存する暫定判定である。

| 水準 | 現段階で言えること | 言えないこと |
|---|---|---|
| 微視モデル | 状態分布は通常の Kolmogorov 確率、発展は古典的 Markov kernel | ニューロンが量子ビット、古典物理の破れ |
| 操作的モデル | 指定された未来実験で区別不能な準備を商にできる | 単発の発火率だけで全ての未来が予測できる |
| 有限データ | 指定した確率表の有限次元表現を定義・推定できる | 有限データで全未来の有限次元性を証明できる |
| 古典性検査 | 指定した状態・効果・等値関係について単体埋込みを問える | 非単体の図、非可換性だけで一般化文脈性を証明できる |
| 計算 | 予測、記憶、非線形課題の成績と幾何の関係を比較できる | 幾何の複雑さだけで計算資源・量子優位を主張できる |
| 生物学 | 実測結線と仮定した動力学を組み合わせる設計ができる | 結線のみから生体の実行状態・認知機能を再現したと言える |

特に、古典親理論の状態・効果を線形関係ごと制限する **fragment** と、見える確率だけで同一視し直す **operational quotient / shadow** は別物である。前者の埋込みを制限できても、後者で新たに課される等値関係をその埋込みが尊重するとは限らない。この区別の既存理論は [Schmid et al., Quantum 9, 1880 (2025)](https://doi.org/10.22331/q-2025-10-13-1880) を参照する。

## Phase 0 の境界

成果は数式、証明、文献比較、将来実験の仕様である。`theory_examples/` は古典有限確率表を有理数で検算するだけで、本番の神経シミュレータとは独立している。

MaleCNS 全データ取得、全 CNS シミュレーション、FlyGym、言語モデル学習、暗号実装、量子メモリとの再結合、hayate 性能最適化は将来段階に留める。Phase 0 の文書からそれらを実行する許可や、生物学的結論は生じない。

## リポジトリの出発点

調査時点で `master` にはコミットがなく、多数の既存ファイルが staged、`src/bioquantumfly/analysis/lifetime_extension.py` が untracked だった。既存 README、モデル・仮定・出典・環境文書、神経パラメータ・Brian2 構築・実験層を読み、追加文書の範囲を決めた。既存ファイルの内容、index、ブランチを保持する。既存文書の説明は実装の出発点を知るために用い、生物学的事実の根拠は一次文献で別途確認する。

完了時には既存123ファイルの内容ハッシュと index の不変を確認した。追加は本ディレクトリの15文書（BibTeX を含む）と `theory_examples/` の2ファイルである。小例の全厳密検算、文書内の相対リンク、数式区切り、表の列数、26件の BibTeX キー・括弧を確認した。神経シミュレーション、実データ取得、既存の実験再実行、依存追加、git commit は行っていない。
