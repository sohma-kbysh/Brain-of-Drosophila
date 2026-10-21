# 用語・記号・表現規則

## 科学的主張の語彙

| 語 | この研究での意味 | 使用規則 |
|---|---|---|
| GenProbTheory / operational GPT | Generalized Probabilistic Theory、一般化確率理論 | Generative Pre-trained Transformer と区別する。単独の GPT が曖昧なら使わない |
| 物理的非古典性 / physical nonclassicality | 仮定した古典物理理論では説明できない物理的証拠 | 現行 classical simulator に関しては主張しない |
| 一般化文脈性 / generalized contextuality | 指定した操作的理論に、その等値関係と混合を尊重する一般化非文脈 ontological model がないこと | prepare–measure か transformation までかを明記する。生体の事実とは現段階で言わない |
| 非単体幾何 / non-simplex geometry | 状態凸集合が単体ではないという affine 幾何の性質 | 図の見た目では確定しない。これだけで文脈性を結論しない |
| 単体埋込み不可能 / simplex non-embeddability | 状態・効果双方を、pairing・unit・指定等式を保って古典単体とその効果へ埋め込めないこと | 対象 fragment、finite ontic 等の条件、統計的確実性を併記 |
| 操作的に非古典的 | 上のどの判定を意味するか固定したときだけの略語 | 原則「指定した操作的 fragment は単体埋込み不可能」と書く |
| 古典的に模擬可能 / classically simulable | 古典アルゴリズムが指定精度・時間範囲で過程を実行/近似できること | efficient、exact、finite precision、sample generation の違いを明記 |
| 計算資源 / computational resource | 指定課題・free-resource class・操作・cost model の下で性能/達成可能性を支える性質 | 相関だけでは使わない。monotone または統制比較等の根拠を必要とする |
| 計算上の優位 / computational advantage | 指定 competitor class より、同じ費用と評価条件で課題成績が改善すること | 生物モデルが一つの課題を解けただけでは使わない |
| 非可換性 / noncommutativity | $T_AT_B\ne T_BT_A$ | 古典 stochastic maps にも存在。文脈性・量子性と別物 |
| quantum-like | 文献中の特定の数学的類似性を表す歴史的用語 | 本研究の説明的結論には使わない。引用時は非可換 instrument、順序効果等の具体的意味を記す |
| accessible fragment | 許容準備・効果・変換の部分集合とその関係 | 親理論の表現における fragment か、商後の操作的 fragment か区別 |
| shadow / operational quotient | 指定した pairing で見えない差を商にした理論 | 親 fragment と同じ古典性を持つとは自動的に仮定しない |
| relative tomographic completeness | 指定した親 fragment の状態・効果が、互いの span にある区別を漏れなく検出する条件 | 有限データを商にして得た「完全性」から親 fragment の条件を循環的に結論しない |
| operational equivalence | 全ての指定許容未来に対する確率の一致 | 近接や検定非有意は厳密等値ではない |
| baseline / control | 主張対象と異なる一要因または明示した要因群を持つ比較対象 | matched の内容を実測確認。関数名だけで一致を仮定しない |
| 証明・既知結果 | 仮定から論理的に従うもの | 既知結果は出典、独自の小命題は証明を付す |
| 推論・仮説・憶測 | 条件付き適用・検証可能主張・未定式化可能性 | 実証済みの語調で書かない |

## 記号

| 記号 | 意味 |
|---|---|
| $\lambda\in\Lambda$ | 完全な古典微視状態とその空間 |
| $\mu_P$ | 準備 $P$ が作る微視状態上の通常の確率分布 |
| $K_\tau$ | 古典微視発展の kernel |
| $\mathcal P,\mathcal T,\mathcal M$ | 許された準備・介入・測定手続き |
| $\mathcal Q,H$ | 許された未来テスト集合、有限 horizon |
| $P\simeq P'$ | 指定 $\mathcal Q$ に関する操作的同値性 |
| $\omega_P=[P]$ | 準備の操作的同値類。$\lambda$ と同一ではない |
| $V,C,\Omega_{\rm acc}$ | 操作的実ベクトル空間、非規格化状態 cone、規格化状態凸集合 |
| $e\in E_{\rm acc}\subset V^*$ | 許容 outcome の操作的効果 |
| $u$ | 単位効果。入力信号の記号には原則 $a,x$ を使う |
| $T_\tau$ | 商への降下条件を満たす介入の作用 |
| $D,F$ | 真の確率表、有限試行頻度表 |
| $S,E$ | $D=SE$ の状態行列・効果行列。$E_{\rm acc}$ という集合と区別 |
| $G$ | 文脈によりネットワーク topology または gauge 行列。混同がある箇所は $G_{\rm net},G_{\rm gauge}$ と展開 |
| $d,d_{\rm op}$ | GPT 線形 dimension、規格化状態の affine dimension $d-1$ |
| $\Delta_L$ | $L$ outcomes の古典確率単体。affine dimension は $L-1$ |
| $\iota,\kappa$ | 状態・効果の互換な単体埋込み |
| $\delta_E$ | 指定効果から定義した状態識別距離 |
| $R_{\rm ctx}^{\mathcal N}$ | 指定 noise family と等式の下での非文脈化閾値 |

用語の一次文献上の基礎は [Schmid et al. (2021)](https://doi.org/10.1103/PRXQuantum.2.010331)、[Selby et al. (2024)](https://doi.org/10.1103/PhysRevLett.132.050202)、fragment/shadow の区別は [Schmid et al. (2025)](https://doi.org/10.22331/q-2025-10-13-1880)。本研究の許容表現はこれらの用語を生物学的または計算量的結論へ拡張しないための規約である。
