# 反証可能な仮説と棄却条件

> **現在の主仮説（2026-09-12）：** GPTを使った設計により、生物回路で有効な非古典的資源を生成でき、その状態を操作して課題を解き、生成・利用の全費用において比較方式への利益を得られるか。[最新実行計画](nonclassical_resource_execution_plan.md)で、生成・操作・効率・設計手法の寄与を別々に判定する。以下の予測十分性と圧縮の仮説は、既存実験の背景と補助課題として保存する。

以下は測定済みの結果ではない。母集団、interface、horizon、効果量閾値、対照、統計モデルを先に固定して初めて検定命題になる。小さい pilot で広い生物仮説を棄却できるとはしない。仮説に都合のよい interface 探索と、その interface での確認検定を同じデータで行わない。

## 初期Phase 1の主仮説：予測十分性と操作的圧縮

2026-09-12のユーザー指示により、主評価を**GPT-likeな有効挙動とresource estimation**へ明示的に移す。[現行方針](resource_priority.md)の能力・性能対予算・頑健性を優先し、下記の厳密なモデル族による定義は補助的な参照とする。有限精度で有用な予測器や順序依存が得られる場合、全GPT条件やRTCの証明が未完でも資源評価を進める。

主仮説を「生物神経回路の計算に必要な予測を、操作的に再構成した小さい状態–効果空間で保持でき、その幾何が能力を説明する」に置く。単体埋込み可能性は副次的なH3である。H3が埋込可能でも、予測圧縮や説明力の研究は継続できる。

許容interface $\mathcal I$、未来horizon $H$、誤差 $\epsilon$、readout/cost制約を固定し、

$$
d_{\rm pred}(\epsilon,H,\mathcal I)
=\min\{\dim V:\sup_{P,q\in\mathcal Q_H}
|p(q\mid P)-e_q(\omega_P)|\le\epsilon\}
$$

を候補とする。最小化は許された混合を尊重し、有効な確率と状態–効果pairingを持つ指定モデル族の中で行う。閉じた逐次更新まで要求する場合は、その条件もモデル族に加える。有限データのfitはこの全集合上の最小値の証明ではなく、候補上界と予測誤差の検証を与えるに留まる。

評価は未使用の入力・未来テストと下流課題で行い、通常のPSR、同じ観測だけを使う予測器、微視状態oracleとの精度・費用を比較する。再準備とshot数を費用に含める。連続微視座標の個数 $\dim\Lambda$、分布空間の線形dimension、GPTの $\dim V$ は別の概念であり、その数の比だけを圧縮の実証としない。最小化が線形なGPT表現に限定されるなら、任意の非線形予測状態の最小dimensionを得たとは言わない。

この主仮説の負の結果は、低次元モデルが未使用未来で失敗する、同等の制約付き既存予測器より費用が高い、幾何が通常の指標以上に成績を説明しない、というものになる。1A–1Cはこの仮説の生物学的検証ではなく、手順の校正である。[Phase 1仕様](phase1_protocol.md)

## 仮説族

| ID | 主張の正確な形 | 主要な比較・指標 | 棄却または支持しない条件 |
|---|---|---|---|
| H0 | 現行 connectome/LIF 計算は明示的に古典アルゴリズムで実行可能 | kernel と既存実装の対応 | **これは前提事実であり、打倒する仮説ではない**。表現変更による物理的非古典性は結論できない |
| H1 | 同じ $N,G$, I/O、protocol/shot budget の下で、指定した豊かな局所動力学は指定 scalar baseline より大きい有限 horizon $d_{\rm op}$ を持つ | 不確実性込みの $\Delta d_H$、未使用未来への予測尤度 | 次元差が再現せず、事前の最小差を上限信頼限界も下回るなら、その比較に関する H1 を棄却 |
| H2 | MaleCNS と matched random ensemble の操作的幾何は事前の距離で異なる | $d_H$、固定 noise の $R$、$\delta_E$ packing 等の事前選択量 | 十分精度の同等性判定で差が実用的閾値内なら、指定効果の特殊性を棄却。方向は仮定しない |
| H3 | 指定 fragment と正当化した等値関係の下で単体埋込みの存在を決定する | feasible embedding または robust infeasibility certificate | 埋込可・非埋込・不定の三値を報告。**どちらの確定結果も科学的に受容する** |
| H4 | 幾何指標は $N$、次数、結合重み、動力学、活動量、通常の reservoir 指標を条件付けても held-out 課題成績を予測する | ベース回帰への out-of-sample 予測改善、介入による変化 | 事前最小改善を上限も下回る、または効果が通常指標で説明されるなら追加説明力を棄却 |
| H5 | トポロジーと局所動力学の richness の効果には指定尺度で interaction がある | 下記 factorial contrast | contrast の同等性区間が実用的零領域内なら、その課題・尺度で interaction 仮説を棄却 |
| H6 | MaleCNS に matched reservoir を超える特別な幾何・容量がない | H2/H4/H5 の負の結果を積極的にモデル化 | 十分な精度での null 結果を主要結果として採用。単なる $p>0.05$ は無差の証明ではない |

H1 の「scalar」は単一の発火率表現という曖昧な呼称ではなく、1 neuron あたり一つの連続内部変数を持つ具体的更新則に固定する。**scalar neuron も連続分布と豊かな future tests を持てば高 rank / 無限 rank になり得る。** したがって H1 は神経あたりの変数個数からの定理ではない。比較する topology、時間分解能、readout 非線形性、刺激量、実効 memory、学習自由度の違いを必ず併記する。

## 主張の強さを分ける

H4 の予測的関連について、network realization $b$、条件 $c$、課題 $s$ の成績を

$$
Y_{bcs}=\beta_0+\beta_q Q_{bc}+\beta_z^\top Z_{bc}
+\alpha_b+\gamma_s+\varepsilon_{bcs}
$$

とモデル化できる。$Z$ は network size、degree/strength、活動率、時定数、入力強度、通常の feature rank 等である。共線性で係数が識別できない場合は、符号の物語を作らず不定とする。複数の $Q$ を試すなら探索と確認を分け、多重性を扱う。

この観察的回帰だけでは「幾何が因果的に計算を改善する」と言えない。幾何量を変えつつ、他の情報経路と resource cost をどこまで一致させた介入かが必要である。トポロジー randomization や measurement noise は性能へ直接作用するので、それらを無条件に $Q$ の操作変数とは見なさない。排除制約が示せなければ、因果効果は intervention package 全体の効果として報告する。

H5 では $G\in\{\text{measured},\text{matched randomized}\}$、$L\in\{\text{richer},\text{scalar}\}$ に対し、loss を向きを揃えた成績 $C$ に変換して

$$
\Delta_{\rm int}
=(C_{G_m,L_r}-C_{G_m,L_s})
 -(C_{G_0,L_r}-C_{G_0,L_s}).
$$

interaction は選んだ尺度に依存する。seed の反復を独立個体数と数えず、rewiring realization、入力系列、simulator seed の階層を明示する。全 CNS 一個体の結線から種・性差の一般法則を推論しない。

## 事前に設定する負の主要結果

1. 未来テストを増やすと低次元の閉じたモデルが破れ続ける。有限 horizon の予測表は残せるが、簡潔な定常 operational GPT という動機を後退させる。
2. 正当な interface では埋込可能な範囲しか得られない。contextuality を必要資源とする強い主張を棄却する。幾何と容量の通常の関連は独立に検討できる。
3. 非埋込性が恣意的な readout の隠蔽・追加等値化だけで生じ、生物的な制限根拠や安定した計算関係がない。生物計算の説明として採用しない。
4. matched random reservoirs と事前閾値内で同等、または通常の reservoir / PSR 指標が全ての追加予測を吸収する。MaleCNS 特有の操作的計算幾何という動機を棄却する。
5. richer dynamics の効果だけで説明され、topology×richness interaction が同等性検定で実用的に零。H5 を棄却し、必要なら一般の動力学研究へ変更する。

最小の次段階 pilot は [09](09_future_experiment_spec.md) の一件である。これは H1–H6 の本確認実験ではなく、未来を含む定義が測定手順に落とせるかを確認する engineering/theory gate である。最終的な biological interface、matched ensemble と検出力設計が定まるまで、H1–H6 の MaleCNS 検定を開始しない。
