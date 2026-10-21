# 将来実験の事前仕様と Phase 1 gate

この文書は未実施の実験計画である。Phase 0 で neural run、MaleCNS 取得、トモグラフィー/LP 実装を行わない。大規模データに進む前に、数学的対象と統計的判定をここで固定する。

**履歴注記：** 上記はPhase 0時点の記録である。次段階の計画は、ユーザーの改訂により1A予測十分性、1B既知次元の再構成校正、1C意図的shadow trapへ拡張した。relative tomographic completenessを中心条件とし、200・2,000・20,000試行を比較する。現行の実装仕様・方法改訂・実測結果は [Phase 1仕様](phase1_protocol.md) と [Phase 1結果](phase1_results.md) を参照する。MaleCNSの取得・実行は引き続き行っていない。

## 1. 解析対象を先に宣言する

各研究条件には、少なくとも次の manifest を付ける。

| 項目 | 保存内容 |
|---|---|
| 微視モデル | グラフ来歴、局所変数、kernel、遅延、境界条件、dt、乱数の扱い |
| 準備 | baseline/reset、刺激履歴、混合比、ラベルを渡さない規約 |
| 未来 | 許可入力 alphabet、持続時間、最大 horizon、合成・適応・反復測定の許否 |
| 測定 | output 集合、時間窓、閾値、補 outcome、instrument と postselection |
| 対象理論 | 親古典 fragment、指定未来で商にした operational fragment、有限 sampled shadow の区別 |
| 同値性 | 各等式の係数と証明/仮定、採用する span、近似誤差の意味 |
| 課題 | 入力分布、target、readout、学習・探索・shot 予算、loss、competitor |
| 統計 | 独立単位、試行数、主要量、最小効果量、信頼水準、多重性、打切り規則 |

数値計算に便利な全電位の観測や任意状態の書込みを、操作的な許可 I/O に追加しない。debug/oracle 記録を使う場合は、それが本来の readout から不可視であることを明記する。

## 2. データ生成と rank の同定

正確な確率表を $D$、計数を $n_{ijo}$、試行数を $N_{ij}$、頻度を $F$ とする。各 prepare–future–measure 設定を独立 reset から繰り返す。単一連続軌道の各時間 bin は自動的に独立 shot ではない。準備の乱択 seed と神経雑音 seed を分離し、必要なら同じ入力列を用いる paired comparison を解析に反映する。

rank 推定は [02](02_gpt_geometry.md) の尤度と model selection に従う。候補 rank の集合、学習・検証分割、最適化診断を記録する。新しい準備/未来テストを hold out する検証と、同じ設定の追加 shot を hold out する検証は違う。前者は外挿・閉包を、後者は確率推定を調べる。新しい効果や準備を予測するモデルを明示せず、未測定の行列成分を都合よく補完しない。

未知 rank から最も低いものを選んで等値関係を増やすと、偽の文脈性が生じ得る。単体検査の確認段階は以下の固定等式方式を優先する。rank によって異なるモデルを使うなら、信頼できる候補すべてを残すか、採用 rank の誤選択確率まで誤差予算に含める。

## 3. 厳密等値と近似等値の区別

使用する等式を次の三種に分類する。

1. **手続きまたはモデルで証明した厳密等式。** 例えば明示的に定義した混合操作の線形関係、許可された未来全体で証明した symmetry。同じデータに合うというだけではない。
2. **再構成モデル内だけで厳密な等式。** 低 rank モデルや secondary preparations による等式。モデル条件付きと表示し、未観測未来への成立を別仮説にする。
3. **統計的に近い手続き。** $\max_q|p(q\mid P)-p(q\mid P')|\le\epsilon$ という許容差。これは厳密な等値関係ではなく、推移性も一般にない。

非有意な差を「等しい」と見なさない。小さな操作的差から ontic 分布も小さな差だとは、Spekkens 型の厳密非文脈性だけでは導けない。近似等値を LP で厳密零に丸めて陽性としない。連続性を仮定する緩和非文脈モデル、または実装可能な secondary procedures で厳密等式を得る方法を使うなら、その追加仮定と誤差境界を定式化して別解析にする。有限次元推定モデルでの secondary equivalence は、真の全未来における等値の証明ではない。

古典 simulator の全微視等式を保存したモデルは古典表現を持つ。非埋込みが生じるなら、どの**追加の制限相対的等式**が必要なのかを特定する。この工程を省いて生物学的文脈性に言及しない。実験の不完全性と genuine subsystem の relative tomography を区別する既存結果も確認する。[^1]

## 4. 確認用の保守的な統計決定規則

これはこの研究が提案する判定仕様であり、Mazurek/Selby の結果から自動的に得られる新しい統計定理とはしない。**固定した有限シナリオと正しい厳密等式**を前提とする。

**(a) 同時信頼集合を作る。** 主要解析の family-wise error を $\alpha=0.05$ とする。独立試行の各 outcome probability に Clopper–Pearson の exact binomial 区間を作り、異なる全推定確率数 $J$ に対して各区間の非被覆率を $\alpha/J$ とする。多項 outcome の各 indicator も二項周辺分布を持つため、outcome 間の独立性を仮定せず union bound が使える。unit/補 outcome の既知の重複は新しい独立データとして数えない。

得た直積領域と規格化・証明済み等式の共通部分を $\mathcal C_\alpha(F)$ とする。同一設定での試行間の独立性・同分布性が破れる場合はこの被覆保証を使わず、独立 reset 単位または妥当性を校正した時系列推論へ切り替える。後から選んだ rank や等式で領域を削って被覆保証を保ったと主張しない。

**(b) 非文脈モデルとの共通部分を判定する。** [02 §7](02_gpt_geometry.md) の指定した準備・効果等式から応答多面体を固定する。その頂点 $v^{(\ell)}$ に対し、未知の $D\in\mathcal C_\alpha(F)$ と $\mu_{i\ell}$ を同時に LP で探索する。$D_{ij}=\sum_\ell\mu_{i\ell}v^{(\ell)}_j$ は線形なので、この固定シナリオでの判定は LP にできる。未知の rank・等式・全未来を含む任意の完成理論まで同じ一つの LP で解けるとはしない。

**(c) 三つの報告状態を区別する。**

まず $\mathcal C_\alpha(F)=\varnothing$ なら、固定等式・規格化・生成モデル自体との不整合として未判定にする。全候補を排除しているため、非文脈モデルに固有の失敗と帰属できない。以下は $\mathcal C_\alpha(F)\ne\varnothing$ の場合である。

- $\mathcal C_\alpha(F)\cap\mathcal N\ne\varnothing$：指定モデルの非文脈性と**整合的**。feasible witness を保存する。有限の noisy data から真の全 fragment が埋込可能と証明したとは言わない。
- $\mathcal C_\alpha(F)\cap\mathcal N=\varnothing$：指定等式を前提に、信頼集合全体が非文脈モデルを排除する。検証された certificate と数値誤差 margin がある場合だけ「条件付き非埋込性の証拠」とする。
- 等式根拠、被覆、候補モデル、最適化の完全性、certificate のいずれかが不足：**未判定**。solver の失敗もこの区分に入れる。

ここで $\mathcal N$ は指定した有限シナリオの非文脈確率表集合である。真の表が $\mathcal N$ に属し、$\mathcal C_\alpha$ がその表を少なくとも $1-\alpha$ で含むなら、誤った排除は非被覆時にしか起きない。したがって型 I 誤りは $\alpha$ 以下になる。これは union bound と集合包含による短い保証であり、等式が誤りである可能性を帳消しにはしない。

**(d) 証明書を独立検査する。** feasible の非負性・規格化・等式残差、infeasible の dual inequality を再計算する。表の不確実性、係数の丸め、solver residual が証明の正の margin より小さいことを確認する。可能な小規模例では有理数 certificate を使う。margin がない場合は数値不定とし、恣意的 tolerance で陽性にしない。

**(e) 校正と確認を行う。** `theory_examples/` の厳密な正負例を起点に、将来、有限 shots の synthetic null で偽陽性率、既知の noise 系列で検出力を確認する。これは Phase 0 の厳密算術検算とは別工程である。新しい指標や interface を探索した場合は、固定後の独立確認データで再評価する。

## 5. robustness・多重性・数値感度

$R_{\rm ctx}^{\mathcal N}$ について、最適 fit 一個の値だけを報告しない。固定 noise と等式の下で、概念上

$$
R_- =\inf_{D\in\mathcal C_\alpha} R(D),\qquad
R_+ =\sup_{D\in\mathcal C_\alpha} R(D)
$$

を定める。reference preparation の確率も未知なら同じ信頼集合に含める。$R_->0$ の確認には全 compatible model に対する証明が要る。各 $D$ の LP を数個解いただけでは infimum を証明しない。全体最適化ができない場合、正当な下界・上界、または model-conditioned bootstrap 区間と明記する。

bootstrap は rank 選択を含む解析工程全体に適用し、較正された同時信頼集合の代用だとは自動的に言わない。境界 $R=0$ やモデル選択後の区間の被覆は合成データで確認する。多数の geometry 指標、input sets、horizon、task を試す場合は、主要仮説を事前に一つに絞り、確認的な家族には Holm 等の family-wise 補正を用いる。探索的 FDR は別表示にする。

dt、測定 bin、刺激時間幅の変更は別々に扱う。dt を細かくして同じ物理プロトコルを近似する収束試験は数値誤差を調べる。一方、測定 bin を変えることは効果集合を変える操作でもあり、単なる数値収束ではない。生物学的解釈に進む前に、事前の近隣条件で主要結論がどう変わるかを報告する。

## 6. Phase 1 で最初に行う一件

**選ぶのは、合成の二ニューロン再帰 LIF 装置で「即時 readout 一個では足りない未来予測」を調べる、一つの有限確率表実験である。** 自己結合を使わない最小の再帰 motif として二節点を採用する。これは生物界の最小計算系という主張ではない。MaleCNS、モデル訓練、contextuality 陽性探索は含まない。

以下の数値は pipeline を再現可能にする**人工的な計画規約**で、生理学的に選定した刺激/回路ではない。生物学的 interface の選定は別段階に留める。

| 要素 | 固定した pilot 仕様 |
|---|---|
| 装置 | 合成二節点、興奮性の相互接続、各辺の drive $200W_{\rm syn}$。外部データなし |
| 動力学 | 現行 `LIFParameters` の時定数・閾値・reset・遅延。追加 quantum drive はゼロ。dt $0.1$ ms |
| 入力 | 節点 1 のみ。各 10 ms bin で `0` は外部入力なし、`1` は 100 Hz Poisson drive。イベント振幅は現行刺激規約 $1.2\times\texttt{drive\_to\_fire}$ |
| 準備 | 両節点 rest、$g=0$、不応期なし、遅延 queue 空から、2 bin の履歴 `00`,`01`,`10`,`11`。各試行で独立 reset |
| 測定 | 節点 2 に終端 2 ms 窓で一回以上発火があれば 1、なければ 0。終端窓の外部入力なし |
| 未来テスト | $q_0=M$、$q_1=10$ ms の `0` の後に $M$、$q_2=10$ ms の `1` の後に $M$。一試行に一つの終端測定のみ |
| 表 | 4 preparations × 3 tests、補 outcome と unit 列は正規化で追加 |
| 予算 | 各セル独立 200 試行、合計 2,400 試行。各軌道最大 32 ms、CPU 1 worker、掃引なし |
| 主解析 | 正確な表の $\operatorname{rank}[\mathbf1,D]>2$ を、同時信頼集合が rank $\le2$ の表を排除するかで問う |
| 追加 readout | 電位や完全微視状態は主確率表に加えない。debug で必要なら oracle と明示 |

即時二値 readout 一個と unit の線形 dimension は高々 2 である。全表の rank が 3 以上なら、全準備・許可混合に対する未来予測をその即時確率一個の affine 関数では表せない。これは「現在の発火率が系列計算の十分状態」という安易なモデルを退ける、有限シナリオで意味のある試験になる。各元準備の即時確率が等しいと近似で決めつける必要はない。

rank $\le2$ の排除は非凸問題になり得るため、solver failure で判定しない。小さい表では、事前に指定した 3 行・unit を含む 3 列の小行列式について、同時信頼領域全体で零を排除する検証済み区間評価を十分条件として使える。全部の小行列式を探索する場合も、その根拠は同じ同時領域である。零を排除できなければ「rank 未判定/低 rank と整合的」とする。

この固定 pilot の陰性は、当該刺激・時間窓が未来識別に不足したという結果であり、MaleCNS 仮説全体の反証ではない。全無発火・飽和、数値更新の不整合、信頼区間が広すぎる場合はそのまま報告し、同じ確認実験中に刺激を陽性が出るまで調整しない。独立の設計改訂として一度に一条件を変更する必要性を検討する。

この一件で判定するのは有限 horizon の dimension/予測十分性である。単体検査を行う追加条件は非自明な厳密操作的等式の根拠であり、この pilot で自動的に得られるわけではない。H2–H6 の生物・容量比較を行うには、別の事前設計と検出力予算を必要とする。

実装時には現行 `run_network` が毎回 reset する点に注意し、一試行の準備から未来テストまでを同じ内部状態で連続実行する。現在の集約結果を次の `run_network` に渡すことで代用しない。新たな継続/入力スケジュールの小さなアダプタは Phase 1 の作業であり、ここでは実装しない。

## 7. その後に必要な生物比較の仕様

pilot を通過しても、直ちに全 CNS へ進まない。最初の実測部分回路は I/O の生物学的根拠、境界条件、真正データ来歴を先に選定する。`MaleCNS topology × richer/scalar dynamics` と `degree-preserving randomized topology × same two dynamics` の factorial 比較に、通常の reservoir/PSR baseline を含める。既存のランダム化関数の入 strength・E/I・辺数の限定は [05](05_connectome_mapping.md) の通り、実際の出力で一致条件を監査する。

主要 task は短期記憶か事前の非線形時間課題を一つ選び、loss・readout・最適化予算を合わせる。geometry が通常の容量指標を超えて予測するかを held-out network/input conditions で問う。topology/状態モデル介入の総効果と、geometry の因果効果は別に報告する。後者の同定条件がない場合、予測的関連の結論に留める。[08](08_hypotheses.md) の停止条件を優先する。

## 8. Phase 0 終了時の gate 判定

| 必須条件 | 現段階の判定 | 残る限定 |
|---|---|---|
| 一貫した操作的理論 | 達成 | 全未来の有限次元性は未証明。有限実験の商と区別 |
| $\lambda$ と $\omega$ の明示的区別 | 達成 | 実装 snapshot を operational state としない |
| 非文脈性・単体埋込みの正確な意味 | 達成 | 親 fragment / shadow と変換非文脈性を区別 |
| 真に反証可能な仮説 | 達成 | H1–H6 の本検定の数値予算は pilot 後に固定 |
| 通常 reservoir との相違 | 研究上の相違を定式化 | 追加説明力・有用性は未実証で、失敗し得る |
| gauge・統計的に意味のある測定 | 達成する仕様を定義 | finite-sample 校正と biological interface は未実施 |
| 動機を退ける負の結果 | 明示 | 正当に導出した fragment が平凡、または幾何が性能を追加説明しなければ主張を縮小 |

**判定：上記一件の合成 pilot へは条件付き Go。MaleCNS の文脈性検証・計算資源の実証への gate はまだ開いていない。** 数学的定義と試験可能な問いは残るため、小さい pilot には目的がある。任意の shadow の発見自体を目的にする場合や、relative tomography・制限根拠を説明できないまま生体の非古典性を目的にする場合は No-go とし、操作的 system identification と容量の問いへ変更する。

[^1]: Schmid et al., Quantum 9, 1880 (2025), [DOI](https://doi.org/10.22331/q-2025-10-13-1880)。状態・効果の検査は Schmid et al., PRX Quantum 2, 010331 (2021), [DOI](https://doi.org/10.1103/PRXQuantum.2.010331)、Selby et al., PRL 132, 050202 (2024), [DOI](https://doi.org/10.1103/PhysRevLett.132.050202)。推定問題の基礎は Mazurek et al., PRX Quantum 2, 020302 (2021), [DOI](https://doi.org/10.1103/PRXQuantum.2.020302)。近似等値と実装可能な secondary procedures は Mazurek et al., Nature Communications 7, 11780 (2016), [DOI](https://doi.org/10.1038/ncomms11780)。
