# 操作的状態による逐次計算

## 1. 入力駆動モデルと確率的出力

$\omega_t$ を入力 prefix $x_1,\ldots,x_t$ の処理直後の状態と定義する。この添字規約では

$$
\omega_{t+1}=T_{x_{t+1}}\omega_t,\qquad
\omega_t=T_{x_t}\cdots T_{x_1}\omega_0,
$$

$$
p(\widehat x_{t+1}=a\mid x_{\le t})=e_a(\omega_t),\quad
e_a\ge0,\quad\sum_a e_a=u.
$$

入力を処理する直前を $\omega_t$ と定義すれば $\omega_{t+1}=T_{x_t}\omega_t$ となり、予測 readout は更新後に置く。二つの規約を混ぜて一文字分の未来を漏洩させない。

この式は sequence predictor の構成を与える。出力分布を読み、サンプルしたトークンを外部入力として次に供給するなら、読み出し後の状態変化も含めてプロトコルを指定する。出力を生じながら内部状態が更新される一般の生成過程は instrument $B_a$ により

$$
p(a\mid\omega)=u(B_a\omega),\qquad
\omega'={B_a\omega\over u(B_a\omega)},\qquad
\sum_a B_a\ \text{は規格化保存}
$$

と表す。古典 HMM の観測後の Bayes 更新もこの形で、正規化後の状態更新は一般には非線形である。制御入力に対する線形 map と、観測による条件付けを区別する。

## 2. 既存モデルとの対応

| モデル | 微視または通常の内部表現 | この枠組みとの対応・限定 |
|---|---|---|
| 有限状態確率機械・HMM | $L$ 状態と非負遷移行列 | 全分布は $\Delta_L$。制限された予測商はこれより小さく、場合により shadow になる |
| RNN | $h_{t+1}=f(h_t,x_t)$ | $h$ 自体が一般には affine な GPT 状態ではない。$h$ 上の分布を線形に更新し、許容未来で商を取る |
| reservoir computer | 固定 recurrent dynamics、訓練する readout | 同じ prepare–transform–measure 記述が可能。幾何の追加分析が役立つかは未確定 |
| state-space model / PSR | 潜在状態または未来テストの条件付き確率 | PSR は特に直接の先行。history–test 表の rank と未来予測が既知の基本構造 |
| 生物再帰神経系 | 電位、シナプス、不応期、遅延、未観測変数 | 許された刺激履歴を準備とし、出力を確率効果にする。生物学的十分性は別の仮説 |
| 量子モデル | 密度行列、量子 instrument | 適切な選択なら数学的参照例として含むが、生体にその選択を採用する根拠にはならない |
| 一般の GenProbTheory | 順序線形空間、許容効果、正写像 | 上記を比較する共通言語。任意の GPT が生物学的に実装可能とは主張しない |

未来の実験予測ベクトルで状態を表し、その系の線形 dimension を history–test 行列で扱う発想は PSR に先行がある。[^1] 本研究の candidate contribution はこれを発見することではなく、実測結線と生物学的制約を持つ装置に対して、状態・効果の等値関係と非文脈表現の存在を計算課題とともに比較することである。

## 3. 非可換性と計算の順序

古典的 bit の分布 $p=(p_0,p_1)^\top$ に、$A$ を bit flip、$B$ を常に 0 に reset する操作として

$$
A=\begin{pmatrix}0&1\\1&0\end{pmatrix},\quad
B=\begin{pmatrix}1&1\\0&0\end{pmatrix},\quad
AB=\begin{pmatrix}0&0\\1&1\end{pmatrix}\ne
BA=\begin{pmatrix}1&1\\0&0\end{pmatrix}.
$$

全て非負で各列和が 1 の古典遷移である。したがって**命題 A：非可換性は物理的非古典性または一般化文脈性の十分条件ではない**。この反例で証明される。刺激順序、hysteresis、recurrence もそれだけでは witness にならない。

それでも順序差は $T_AT_B\omega$ と $T_BT_A\omega$ の許容効果での区別として計算に使える。演算子の差があっても到達状態や許容効果が差を見なければ、操作的に順序を読めない。このため将来観測量として

$$
\chi_{AB}=\sup_{\omega\in\Omega_{\rm acc},e\in E_{\rm acc}}
|e(T_AT_B\omega)-e(T_BT_A\omega)|
$$

を定義できる。これは gauge 不変で対象 interface に相対的であり、contextuality 指標ではない。

## 4. 計算能力の定義には readout 制約が要る

入力過程 $\nu$、課題 target $Y_t=f(X_{\le t})$、許容 readout class $\mathcal R$、学習データ数、計算コスト、loss を固定する。

$$
L^*(\mathfrak F;\nu,f,\mathcal R)=\inf_{r\in\mathcal R}
\mathbb E_\nu[\ell(Y_t,r(\omega_t))].
$$

任意精度の実数、全ての隠れ変数、任意に強力な非線形 readout を一方だけに許した比較はしない。複数回の同一 prefix 再準備から推定する確率ベクトルと、単一 online trajectory の readout は別の計算資源である。前者の shots と再実行時間を課題のコストに算入する。

**命題 C（限定された simulation no-go）。** 有限精度・有限 step の現行 LIF プログラムの出力分布は、その同じ古典プログラムを実行してサンプルできる。ゆえにその統計に GPT 座標を付けるだけでは、古典機を超える complexity-theoretic quantum supremacy は導けない。証明は既存プログラムが classical simulator の構成的 witness になっていることによる。全ての分布を厳密計算すること、任意サイズで効率的であること、最小の非文脈モデルが小さいことまでは主張していない。

**命題 E（資源主張の必要条件）。** 対象 $\mathfrak F$、課題族、free objects / free operations、コスト・readout class を固定しなければ「量 $Q$ が計算資源である」という比較命題は定まらない。操作の前順序を定めて $Q(\Phi\mathfrak F)\le Q(\mathfrak F)$ を証明するか、同条件の free baseline に対して課題達成可能性や loss の改善を示す必要がある。これは新しい資源定理ではなく、本研究でその語を用いるための方法論的必要条件である。相関だけで資源とは呼ばない。

## 5. 候補の定量指標

以下の観測量は未実装であり、互いに同じものではない。

| 量 | 定義と解釈 | 限定 |
|---|---|---|
| 操作的 dimension | $d_{\rm op}=\dim\operatorname{aff}\Omega_{\rm acc}$、有限表なら $\operatorname{rank}[\mathbf1,D]-1$ | 微視変数数・PCA dimension と異なる |
| 有効 rank | 固定した確率表設計での $\exp H(\sigma/\sum\sigma)$ | column 複製や設計で変わる記述量 |
| 状態・効果の幾何 | affine dependencies、exposed faces、$\delta_E$ による直径・packing 数 $N_\epsilon$ | 有限 sampling の凸包の角を生体の特異点と解釈しない |
| 単体埋込み | 指定等式の下で state–effect pairing を保つ古典埋込みの存在 | transformation 非文脈性の全検査ではない |
| $R_{\rm ctx}^{\mathcal N}$ | 指定 reference replacement noise による非文脈化の最小混合率 | noise family と等式を必ず併記 |
| 記憶容量 | 以下の $MC_K$ | 入力分布・lag 範囲・readout を固定 |
| 非線形情報処理容量 | 以下の $IPC_{\mathcal B}$ | orthonormal target と読み出し feature の選択に依存 |
| 予測 loss | $-\mathbb E\log p_\theta(Y_t\mid X_{\le t})$ または MSE | unseen sequences で評価、訓練・shots のコストを合わせる |
| 摂動頑健性 | 指定摂動 $\Pi_\epsilon$ 下での $\mathbb E[L(\Pi_\epsilon\mathfrak F)-L(\mathfrak F)]$ | 解剖誤差、入力雑音、パラメータ誤差を別々に定義 |
| 有効記憶時間 | 以下の $\tau_\epsilon$、または $MC$ の lag 減衰 | 混合性や exponential decay を仮定しない |
| 状態識別性 | $\delta_E(\omega,\omega')$ と有限 shot の識別誤差 | 同一 prefix の再準備数を揃える |
| 情報保持 | $I(X_{t-k};\omega_t)$ | 理想 ensemble と実際に読める outcome の情報は違う |

入力を中心化し分散を 1 に規格化、固定 feature $z_t\in\mathbb R^p$ の線形 readout を許す場合、

$$
C_k=1-\inf_{w,b}{\mathbb E[(X_{t-k}-w^\top z_t-b)^2]\over\operatorname{Var}(X_{t-k})},\qquad
MC_K=\sum_{k=1}^K C_k.
$$

母集団の最適化では定数予測も許し $0\le C_k\le1$。有限 test データの推定値が負になった場合、黙って零に切り上げて容量を過大評価しない。

入力履歴に関して中心化・正規直交した非線形 targets $\{f_\alpha\}_{\alpha\in\mathcal B}$ を事前選択し、同じ式の $X_{t-k}$ を $f_\alpha$ に置いた $C_\alpha$ で $IPC_{\mathcal B}=\sum_\alpha C_\alpha$ とする。Dambre et al. の容量はこのような target 空間への projection と関係する。[^2] **本研究への数学的適用：** 中心化 feature の線形包 $Z\subset L^2(\nu)$ の dimension が $r$ なら、Bessel の不等式から $\sum_\alpha\|\operatorname{Proj}_Z f_\alpha\|^2\le r$。従ってこの readout と正規直交性の下で $IPC\le r$。feature が $\omega$ の affine 座標なら $r\le d_{\rm op}$ だが、任意の神経 hidden coordinates や nonlinear readout に同じ上限を流用しない。

この上限は dimension が容量の上界になり得ることを示す。高 dimension が高い性能を保証することも、非埋込性が性能を増すことも示していない。

上の $L^2(\nu)$ は feature が入力履歴から決まる ensemble 座標の場合の表記である。単一の確率的神経軌道から得る $z_t$ には初期化と内部乱数も含む結合分布 $\rho$ を用い、期待値と射影空間を $L^2(\rho)$ に置き換える。targets はその空間へ入力履歴の関数として持ち上げる。正規直交性と同じ有限次元 feature 空間の条件が満たされれば Bessel 上界は変わらない。

同じ入力列 $w$ を長さ $k$ 与えた二準備の distinguishability を使い

$$
b(k)=\sup_{|w|=k}\sup_{\omega,\omega'\in\Omega_{\rm acc}}
\delta_{E_0}(T_w\omega,T_w\omega'),\qquad
\tau_\epsilon=\inf\{k:\forall l\ge k,\ b(l)\le\epsilon\}
$$

とする。ここで $E_0$ は指定終端測定集合である。集合が空なら $\tau_\epsilon=\infty$。平均入力に対する版は別に $\nu$ を用いて定義する。この量は過去準備の忘却で、特定入力 token の保持量とは同一ではない。

$\omega_t$ を入力履歴から誘導される確率変数として $I(X_{t-k};\omega_t)$ を定義できるが、連続無雑音座標では無限大や推定不安定性が起こり得る。測定 transcript $Z_t$ の情報 $I(X_{t-k};Z_t)$ と混同しない。$X_{t-k}\to\omega_t\to Z_t$ が Markov chain になる条件では data processing により後者は前者以下である。未来の生物実験で読めるのは主に後者であり、finite shots を含めた定義を優先する。

## 6. 何が reservoir computing と異なるべきか

connectome を reservoir にして課題を解くことは既存研究である。[06](06_prior_work.md) 参照。本方向が追加すべきものは、準備混合・測定制限・未来同値性を明示した確率表と、状態・効果の対に対する検査、その量が通常の reservoir 指標を超えて成績や介入効果を説明するかの比較である。

PSR dimension、通常の feature rank、spectral/dynamical 指標だけで同じ予測と介入説明ができるなら、GenProbTheory は記述の言い換えに留まる。その場合は contextuality 資源という主張を撤回し、必要なら操作的 system identification の研究へ問いを変更する。

[^1]: Singh, James & Rudary, UAI (2004), [著者本文](https://web.eecs.umich.edu/~baveja/Papers/uai2004psr.pdf)。会議年は 2004、arXiv 登録年は 2012。
[^2]: Dambre, Verstraeten, Schrauwen & Massar, “Information Processing Capacity of Dynamical Systems,” Scientific Reports 2, 514 (2012), [DOI](https://doi.org/10.1038/srep00514)。その容量定理の前提と、ここでの限定された projection 上限を区別する。
