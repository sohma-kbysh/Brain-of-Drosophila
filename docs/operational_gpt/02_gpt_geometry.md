# 有限次元の操作的幾何とトモグラフィー

## 1. 有限次元性は仮定または有限断片の定義

[01](01_operational_formalism.md) の準備集合の形式的な実線形包を $W$ とし、混合手続きの関係を課した上で

$$
\Phi:W\to\mathbb R^{\mathcal Q},\qquad
\Phi(P)_q=p(q\mid P),\qquad
V=W/\ker\Phi\simeq\operatorname{im}\Phi
$$

と定める。これは準備間の統計的な区別だけを残す構成である。**全許容未来について $d=\dim V<\infty$ であることは一般の LIF 系には保証されない。** 有限個の連続的微視変数を持つことと、分布・未来予測の線形包が有限次元であることは別である。非線形 RNN の有限個の hidden coordinates も自動的には有限次元 GPT 座標にならない。

有限個の準備・未来テストだけの確率表に対する商は有限次元である。しかしそれは有限実験の商であり、全ての未来を予測する定常状態空間を既に得たという意味ではない。以下は有限次元仮定が成立する対象、または明示した有限実験だけに適用する。

## 2. 状態・効果・変換

規格化状態の凸集合を $\Omega_{\rm acc}\subset V$、その線形包を $V$ とし、

$$
C=\operatorname{cone}(\Omega_{\rm acc}),\quad
u\in V^*,\quad u(\omega)=1,\quad
E_{\rm acc}\subseteq\{e\in V^*:0\le e(\omega)\le1\ \forall\omega\in\Omega_{\rm acc}\}.
$$

必要なら閉包を取ることを明記する。規格化面の affine dimension を $d_{\rm op}:=\dim\operatorname{aff}\Omega_{\rm acc}=d-1$ と呼ぶ。文献で GPT dimension と呼ばれる値はしばしば $d$ であるため、両方を併記する。

単位効果 $u$ と零効果 $0$ を含め、実装可能な測定 $M$ について $e_{o\mid M}\in E_{\rm acc}$、$\sum_o e_{o\mid M}=u$ とする。統計は

$$
p(o\mid P,\tau,M)=e_{o\mid M}(T_\tau\omega_P)
$$

で与えられる。決定的な操作的変換は $T_\tau C\subseteq C$ かつ $uT_\tau=u$ を満たす線形写像で、許容状態集合も保存する。instrument の分枝は $T_o C\subseteq C$、$uT_o\le u$、和が規格化を保存する。事後状態は $T_o\omega/u(T_o\omega)$ であり、この正規化による非線形性を線形変換と混同しない。

instrument を同じ cone 内で扱う場合、規格化した条件付き状態も状態集合に含めるという閉包規約が必要である。条件付きで到達する状態を、追加費用なく確定的に準備できるとは推論しない。成功率と再準備費用を保持し、初期準備集合が条件付けに閉じていない場合は、到達状態を別集合として扱うか明示的に拡張する。

対象は通常 $\mathfrak F_{\rm op}=(\Omega_{\rm acc},E_{\rm acc},\mathcal T_{\rm acc})$ という操作的 accessible fragment である。親微視理論の未商化 fragment を扱う際は $\mathfrak F_{\rm micro}$ と書く。**no-restriction hypothesis、全双対効果の実装可能性、テンソル積、local tomography、Hilbert 空間は採用しない。** 状態・効果の凸包と、実装可能な測定や instrument のリストは別々に記録する。

## 3. 確率表と gauge

二値効果の正確な表を $D_{ij}=p(1\mid P_i,M_j)$ とし、必ず unit 列 $D_{i0}=1$ を付ける。実測頻度は $F_{ij}=n_{ij}/N_{ij}$ と区別する。逐次の場合は列を $(\tau_{1:h},M,o)$ とする。多値測定では列 $(M,o)$ を展開し、同じ設定内の和 $\sum_o D_{i,(M,o)}=1$ を拘束する。同一試行の outcomes は独立な二項データではなく、多項分布である。

正確な表の rank が $d$ なら

$$
D=SE,\qquad S\in\mathbb R^{m\times d},\quad E\in\mathbb R^{d\times n}.
$$

$S$ の行が状態、$E$ の列が効果の座標である。フル rank の因子を用いれば、同じ rank の他の因子分解は

$$
S\mapsto SG,\quad E\mapsto G^{-1}E,\quad G\in GL(d,\mathbb R)
$$

で関係する。unit を $(1,0,\ldots,0)^\top$ と固定した後は、それを保存する gauge に制限する。状態を行ベクトルで書く本節では変換の行列は右から作用し、$A_\tau\mapsto G^{-1}A_\tau G$ である。他節の抽象写像の式と同じ内容である。

統計、rank、affine な依存関係、正確な単体埋込みの可否は gauge 不変である。状態座標の Euclidean 距離・角度・単独の体積、$S$ の特異値は一般には不変ではない。同じ空間の二つの full-dimensional 集合の体積比は affine gauge の行列式が相殺するが、次元や比較集合が違う場合は使えない。

操作的に意味のある距離の候補は

$$
\delta_E(\omega,\omega')=\sup_{e\in E_{\rm acc}}|e(\omega)-e(\omega')|.
$$

効果が状態を分離すれば距離、そうでなければ擬距離になる。有限の実装テストだけで最大を取った値は、より広い未来集合での識別性の下界である。最適な二者識別成功率 $(1+\delta_E)/2$ との対応は、等事前確率かつ補効果を持つ許容二値決定測定を最大化している場合に限る。

## 4. Mazurek et al. の方法から採るもの

Mazurek et al. は光子偏光の prepare–measure データに対し、状態と効果を同時推定し、候補 rank の適合度と AIC を比較する GPT トモグラフィーを示した。有限データが全系についてトモグラフィー完全であるとの仮定は暫定的である。誤差評価には光子計数を再標本化する Monte Carlo を用いる。ここではその枠組みを、神経の反復試行に対応する二項・多項尤度と未来テスト列へ適用する**提案**であり、同論文が神経系や逐次十分性を検証したわけではない。[^1]

将来の推定では、候補 $d$ ごとに

$$
\widehat D_d\in\arg\max_{0\le D\le1,\ \operatorname{rank}D\le d}
\sum_{ij}\{n_{ij}\log D_{ij}+(N_{ij}-n_{ij})\log(1-D_{ij})\}
$$

を、unit、outcome 和、事前に正当化した混合等式とともに考える。境界 $0,1$ は尤度の極限で扱う。低 rank の最適化は一般には非凸なので局所解を大域解と呼ばない。収束診断・複数初期値を記録する。重み付き最小二乗は計数が十分な場合の近似に留める。

未拘束 rank-$d$ 行列のパラメータ数は $d(m+n-d)$ だが、unit 列固定、outcome 正規化、既知等式で自由度が変わる。AIC/BIC に機械的に同じ数を流用しない。境界や rank の特異点では通常の漸近尤度比近似も保証されない。主に独立試行・未使用プロトコルへの予測尤度、適合診断、合成 null での校正を使い、情報量規準は補助とする。

## 5. 何を再構成したことになるか

有限実験が実現した状態の凸包 $\Omega_{\rm sampled}$ と、許された設定乱択・outcome 後処理で得られる効果集合 $E_{\rm sampled}$ を推定する。候補 $V$ を固定すると

$$
\Omega_{\rm out}=\{v:u(v)=1,\ 0\le e(v)\le1\ \forall e\in E_{\rm sampled}\}
$$

はその効果と矛盾しない外側集合である。これは生物学的に準備可能な集合ではない。逆向きの外側効果集合も、$\Omega_{\rm sampled}$ 上で確率が有効というだけであり、利用可能な測定には追加しない。双方の外側集合を同時に採ると、相互確率が無効になる場合がある。

有限の sampled fragment、その未観測手続きを含む候補 completion、親理論の fragment の shadow を混同しない。theory-agnostic tomography が一般に特定するのは観測確率の shadow であるという既知結果が、とくに重要である。[^2] 全系の dimension を識別したとするには、準備・効果が対象とする部分系で互いに tomographic だという追加根拠が必要になる。

PCA/UMAP は神経活動や推定座標の**表示**にのみ使える。可視化の曲面・クラスタ・球らしさを、確率 pairing と効果集合を伴った GPT 再構成の代用にはしない。

## 6. 有効 rank と不確実性

正確な有限表については $d=\operatorname{rank}D$、$d_{\rm op}=d-1$ である。全許容未来については全ての有限部分表の rank の上限を使う。有限部分表の値は全体の下界にすぎない。

固定したプロトコル設計・列重み・unit 除去規約の下で、特異値 $\sigma_k$ の $p_k=\sigma_k/\sum_l\sigma_l$ により $r_{\rm eff}=\exp(-\sum_k p_k\log p_k)$ を記述量として定義できる。これは $D$ の統計に基づくため factor gauge には依存しないが、同じ測定の複製、shot 配分、行列の中心化で変わる。普遍的 dimension や contextuality witness ではない。$F$ の数値 rank を真の rank と呼ぶことは禁止する。

独立試行なら二項・多項 parametric bootstrap、時間相関があるなら独立 reset 単位またはブロック再標本化を使う。rank 選択・因子分解・等式処理・単体検査まで各再標本で繰り返す。単に最終図の座標を bootstrap するだけではモデル選択の不確実性を伝播できない。gauge を合わせない座標平均には意味がないので、信頼区間は基本的に確率と gauge 不変量に与える。

## 7. 有限 state–effect シナリオの単体埋込み検査

完全な定義と例は [03](03_classicality_and_contextuality.md)。ここでは全ての指定操作的等式を尊重する affine な写像

$$
\iota:\Omega_{\rm acc}\to\Delta_L,\qquad
\kappa:E_{\rm acc}\to[0,1]^L,\qquad
e(\omega)=\kappa(e)\cdot\iota(\omega),\qquad\kappa(u)=\mathbf1
$$

の存在を問う。状態だけを単体の中に置く問題ではない。有限次元の通常の定理は、有限 ontic モデル等の仮定を明記して一般化非文脈性と対応付ける。transformations を含む非文脈モデルには追加の非負遷移行列・合成・変換等式の条件が必要で、state–effect 検査の成功だけでは足りない。

有限の指定準備等式 $\sum_i a_{ri}\omega_i=0$、効果等式 $\sum_j b_{sj}e_j=0$ を固定した場合、次は理解しやすい LP の構成である。

1. 応答 $v_j$ に $0\le v_j\le1$、$v_u=1$、測定毎の outcome 和、$\sum_j b_{sj}v_j=0$ を課し、応答多面体の頂点 $v^{(\ell)}$ を列挙する。
2. 未知数 $\mu_{i\ell}\ge0$ に $\sum_\ell\mu_{i\ell}=1$、$D_{ij}=\sum_\ell\mu_{i\ell}v_j^{(\ell)}$、$\sum_i a_{ri}\mu_{i\ell}=0$ を課す。
3. feasibility は指定シナリオの非文脈モデルの存在を与える。否定するには全候補を覆う列挙または正当な等価 LP と、検証可能な証明が要る。

任意の ontic 応答は応答多面体の頂点の凸結合で表せるため、頂点に ontic 状態を細分化してもこの存在判定を失わない。頂点列挙自体は大きくなり得る。Selby et al. の cone による LP と実装は、一般の有限 fragment の検査と埋込み証明書・noise robustness を扱う直接の方法論である。[^3] インストール・実装は Phase 0 では行わない。

未知の幾何と未知の等式まで含む問題が全て一つの LP になるわけではない。$D=AB$ の単なる非負分解も不十分である。有限の正確な $D$ に対し $A=I,B=D$ はいつでも作れるが、同値な準備混合を $A$ が同じ分布に写すとは限らない。等式を消してから得た feasibility は別問題への解である。

列挙した等式が再構成した状態・効果の**全線形関係を張る場合**に限り、上の feasibility をその商 fragment の単体埋込みと同一視する。等式が一部だけなら、feasible はその既知等式との整合性に留まる。一方、正当化した一部の等式だけでも infeasible なら、それらを全て保つモデルを排除できる。

## 8. 定義を固定した noise robustness

実装可能な reference preparation $\omega_*\in\Omega_{\rm acc}$ を固定し、label 非依存の置換操作

$$
\mathcal N_\eta(\omega)=(1-\eta)\omega+\eta\omega_*,\qquad0\le\eta\le1
$$

により、各原準備をこの操作に通す noisy experiment を作る。確率は $D^{(\eta)}_{ij}=(1-\eta)D_{ij}+\eta e_j(\omega_*)$。比較中は指定した準備・効果等式とシナリオを保持する。

$$
R_{\rm ctx}^{\mathcal N}(\mathfrak F)
=\inf\{\eta\in[0,1]:D^{(\eta)}\text{ がその指定シナリオで非文脈モデルを持つ}\}.
$$

単位効果を含む準備等式の係数和は零なので、この置換は元の準備等式を保存する。効果等式も評価で保存する。$\eta=1$ は単一準備に退化するため、この noise family では feasible な終点がある。有限 polytope の設定で通常最小値を扱えるが、一般集合では上の infimum を用いる。

これは**指定 noise に対する閾値**で、Euclidean 座標への任意ノイズではない。全ての資源理論に対する monotone や、物理的 decoherence time であるとは言わない。実際の synaptic perturbation と同じ noise だとも仮定しない。$R=0$ の数値推定から全生物系の非文脈性を結論せず、$R>0$ も有限誤差と等値関係の根拠を通過して初めて意味を持つ。決定規則は [09](09_future_experiment_spec.md) に定める。

## 参照

[^1]: Mazurek et al., PRX Quantum 2, 020302 (2021), [DOI](https://doi.org/10.1103/PRXQuantum.2.020302), [著者版本文 §III](https://arxiv.org/html/1710.05948).
[^2]: Schmid et al., Quantum 9, 1880 (2025), [DOI](https://doi.org/10.22331/q-2025-10-13-1880), とくに shadow と tomography の結果。親理論についての結論にはその条件を要する。
[^3]: Selby et al., Physical Review Letters 132, 050202 (2024), [DOI](https://doi.org/10.1103/PhysRevLett.132.050202); 非文脈性と単体埋込みの基礎は Schmid et al., PRX Quantum 2, 010331 (2021), [DOI](https://doi.org/10.1103/PRXQuantum.2.010331).
