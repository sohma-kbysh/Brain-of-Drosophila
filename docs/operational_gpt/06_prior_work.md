# 06 — 先行研究と確認範囲

調査開始: 2026-09-10。記録確定: 2026-09-11。以下の GPT は Generalized Probabilistic Theory（GenProbTheory）を意味する。

本レビューの結論は「構成要素にも、その複数の接続にも明確な先行研究がある」である。特に、**神経ネットワークを GPT で記述すること自体にも直接の先行研究がある**。本調査で、実測 connectome・生物学的に制約された動力学・operational interface・データからの GPT 再構成・simplex 判定・計算容量の六要素を同じ研究設計で扱う論文は確認できなかった。ただし、これは有限の検索と本文確認における未発見であり、不存在証明でも優先権の主張でもない。[新規性の判定](07_novelty_matrix.md)を参照。

## 調査方法と証拠の強さ

出版社、著者の公開原稿、研究機関の公式資料を優先した。Crossref は出版社登録の題名・DOI・巻頁の照合に限って用いた。検索結果は発見の入口であり、検索語の不一致を新規性の根拠にしない。本文を取得できなかった場合は、確認した版・抄録・機関資料を明記する。補足資料、コード、データはすべてを監査したわけではなく、本フェーズで論文の数値結果を再現していない。

下表の「なし」は、確認範囲でその研究が当該解析を行っていないという意味であり、著者の全業績についての判断ではない。「容量」はタスク性能、memory capacity、情報処理容量を区別する。

## 必須文献: 問題、方法、基質、追加すべき点

### Schmid et al. (2021): 非文脈性の幾何学的特徴づけ

**Characterization of Noncontextuality in the Framework of Generalized Probabilistic Theories**, PRX Quantum **2**, 010331. DOI: [10.1103/PRXQuantum.2.010331](https://doi.org/10.1103/PRXQuantum.2.010331)。BibTeX: `schmid2021noncontextuality`。

問題は prepare–measure operational theory の一般化非文脈性を、operational equivalence で商を取った GPT の条件に移すこと。状態と効果の双方に確率を保存する線形写像を要求し、有限個の ontic states を仮定した対応として simplex embeddability を導く。状態集合自体が simplex である条件より弱い。抽象理論であり、生体データも神経計算容量測定もない。本計画が追加し得るのは、この既知の判定を明示的な古典神経モデルの指定 interface に適用することである。**新しい古典性定理の発見とは主張しない。** 出版社抄録および著者公開本文を確認した。[著者公開本文](https://ravikunjwal.com/wp-content/uploads/2021/06/prxquantum.2.010331.pdf)

### Mazurek et al. (2021): 自己整合的 GPT tomography

**Experimentally Bounding Deviations From Quantum Theory in the Landscape of Generalized Probabilistic Theories**, PRX Quantum **2**, 020302. DOI: [10.1103/PRXQuantum.2.020302](https://doi.org/10.1103/PRXQuantum.2.020302)。BibTeX: `mazurek2021tomography`。

問題は量子モデルや既知の測定装置を先に置かず、prepare–measure 確率から許される状態・効果の範囲を推定すること。単一光子偏光の実験を、低 rank の確率行列と自己整合的因子分解で扱い、モデル選択、Monte Carlo 誤差評価、tomographic completeness の仮定を検討する。量子理論からの逸脱や非文脈性不等式に関係する境界を扱うが、神経 capacity の研究ではない。本計画の推定対象も $D=SE$ の operational statistics であり、PCA 図の形ではない。神経系では未観測の未来に関する完全性を別に検討する必要がある。出版社本文・付録および[公開論文](https://eprints.whiterose.ac.uk/173030/1/PRXQuantum.2.020302.pdf)を確認した。

### Selby et al. (2024): simplex embeddability の LP

**Linear Program for Testing Nonclassicality and an Open-Source Implementation**, Physical Review Letters **132**, 050202. DOI: [10.1103/PhysRevLett.132.050202](https://doi.org/10.1103/PhysRevLett.132.050202)。BibTeX: `selby2024linearprogram`。

与えた状態・効果集合を再現する非文脈モデルの存在を線形計画で判定し、可能ならモデル、不可能なら指定したノイズに対する埋込可能化の量を得る。対象は量子 prepare–measure 集合に限らず accessible GPT fragments に及ぶ。生物 connectome も神経タスクの容量測定もない。本計画はこの既知の計算手法を使う側であり、データの誤差、同値関係、正規化、ノイズ写像の事前指定が追加課題となる。**任意の非負行列分解が見つかることと、同値関係を保存する埋込があることを同一視しない。** 出版社抄録と[著者原稿](https://arxiv.org/abs/2204.11905)を確認。実装を実行・監査したとは主張しない。

### Aloy et al. (2026): 時間発展する実験系の理論独立解析

**Theory-independent monitoring of the decoherence of a superconducting qubit with generalized contextuality**, Nature Communications **17**, 2474, 2026-02-07. DOI: [10.1038/s41467-026-69030-x](https://doi.org/10.1038/s41467-026-69030-x)。BibTeX: `aloy2026decoherence`。

超伝導 qubit の待機時間を変えた準備・測定統計から、動的な GPT 再構成、状態空間の収縮、一般化文脈性の変化、非 Markov 性を監視する。生物データも reservoir capacity も扱わない。**動的 tomography と contextuality を結びつけること自体は既知**であり、本計画の追加候補は古典神経 dynamics と計算課題への接続である。実物の量子デバイスに対する結論を古典 LIF シミュレータへ移植してはならない。出版社本文・方法・書誌を確認した。2024 年の [arXiv 原稿](https://arxiv.org/abs/2411.13421)と 2026 年の刊行版は区別する。

### Suárez et al. (2024): conn2res

**Connectome-based reservoir computing with the conn2res toolbox**, Nature Communications **15**, 656. DOI: [10.1038/s41467-024-44900-4](https://doi.org/10.1038/s41467-024-44900-4)。BibTeX: `suarez2024conn2res`。

connectome の構造が計算機能にどう関係するかを、固定 reservoir と訓練する readout により調べる toolbox。結合行列、入出力ノード、局所 dynamics、タスクを選べ、人工ニューロン、spiking、memristive dynamics を扱う。実測 human diffusion MRI connectome の memory capacity と degree sequence を制御した比較、複数動物 connectome の例がある。GPT 再構成や simplex 判定は行わない。本計画は operational な同値関係・状態効果対の制約が、通常の reservoir 記述以上の予測力を持つかを追加検証しなければならない。**topology、局所 dynamics、I/O を分ける発想も既知。** 出版社本文・方法を確認した。[公式実装](https://github.com/netneurolab/conn2res)

### Costi et al. (2025): fly connectome を reservoir にする先行研究

**The Drosophila Connectome as a Computational Reservoir for Time-Series Prediction**, Biomimetics **10**(5), 341. DOI: [10.3390/biomimetics10050341](https://doi.org/10.3390/biomimetics10050341)。BibTeX: `costi2025drosophila`。

FlyWire 系の成虫 Drosophila 解剖データから ESN の結合構造・重みを作り、多変量のカオス時系列予測と過学習耐性を比較する。神経選択の違い、topology と重み分布を別々に交換する hybrid controls を含む。これは MaleCNS v1.0 の研究ではなく、完全な生物 LIF dynamics の再現でもない。GPT・simplex 判定はない。予測誤差を測るが、それをそのまま Dambre の総情報処理容量と呼べない。**「Drosophila reservoir」および構造・重みの要因分解を新規性に数えない。** MDPI 書誌、[PMC 本文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12109256/)の研究内容、[著者所属 ESA の説明](https://www.esa.int/gsp/ACT/projects/fly_connectome/)を確認。MDPI 本文の直接取得には制限があった。

### Berg et al. (2026): MaleCNS の構造的基質

刊行版の題名は **Sexual dimorphism in the complete Drosophila male central nervous system connectome**, Cell **189**(18), 5504–5526.e15. DOI: [10.1016/j.cell.2026.08.015](https://doi.org/10.1016/j.cell.2026.08.015)。BibTeX: `berg2026malecns`。

依頼文の **Sexual dimorphism in the complete connectome of the Drosophila male central nervous system** は preprint と Google 著者ページに残る題名である。刊行版書誌は[出版社登録メタデータ](https://api.crossref.org/works/10.1016/j.cell.2026.08.015)で照合した。雄の脳と nerve cord を通した EM connectome・細胞型・性差比較が問題であり、実行可能な生理 dynamics の測定や GPT・contextuality・capacity の論文ではない。[Google 著者抄録](https://research.google/pubs/sexual-dimorphism-in-the-complete-connectome-of-the-drosophila-male-central-nervous-system/)は 166,691 neurons を報告する。この解剖学的情報から $K$、受容体依存符号、I/O を別途定義するのが本計画の課題となる。

確認範囲は [2025 年公開原稿](https://pmc.ncbi.nlm.nih.gov/articles/PMC12636603/)、著者公式抄録、刊行版書誌、公式公開記録。[Cell 本文](https://www.cell.com/cell/fulltext/S0092-8674(26)00942-6)は取得制限があり、**最終本文の全改訂を照合したとは主張しない**。

### Shiu et al. (2024): connectome 制約のある古典 LIF dynamics

**A Drosophila computational brain model reveals sensorimotor processing**, Nature **634**, 210–219. DOI: [10.1038/s41586-024-07763-9](https://doi.org/10.1038/s41586-024-07763-9)。BibTeX: `shiu2024brain`。

FlyWire connectome と予測 neurotransmitter identity に基づく Brian2 LIF モデルで、味覚・摂食開始・触角 grooming の sensorimotor 変換を予測し、一部を実験検証する。GPT や simplex test、一般的 sequence capacity の benchmark はない。本計画の $K(\lambda'\mid\lambda,x)$ の先例だが、この論文の局所パラメータを MaleCNS に移すことは追加のモデル仮定となる。解剖データだけで個体の活動を一意に決めたとも、全行動を検証したとも解釈しない。本計画の追加候補は同じ dynamics に制約された operational interface の統計を定義すること。出版社本文・方法と [公開論文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11446845/)を確認した。

## MaleCNS v1.0: 論文とは別に固定する版

Janelia 公式サイトは v1.0 の公開を **2026-06-08**、Cell 論文公開を **2026-09-03** と記録する。[公式公開記録](https://male-cns.janelia.org/)、[release notes](https://male-cns.janelia.org/release/)を確認した。v1.0 は v0.9 に対する proofreading・annotation 更新を含む。将来は [download documentation](https://male-cns.janelia.org/download/)の dataset identifier、取得日、対象ニューロン・辺・フィルタを manifest に固定する必要がある。公式資料が公開する画像・形態・synapse・annotation と、仮定する dynamics は別の入力である。ここではデータ本体を取得していない。BibTeX: `malecns2026docs`。

Google Research の [2026-09-03 著者記事](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/)も刊行版へのリンクとプロジェクトの位置づけを確認する一次機関資料である。広報記事を physiological parameter の検証資料にしない。BibTeX: `google2026malecns`。

## 直接関係する追加理論と、神経・認知の近接研究

| 文献・確認範囲 | 問題・formalism・基質 | 本計画が追加し得る点／結論の限界 |
|---|---|---|
| Selby et al. (2023), **Accessible fragments of generalized probabilistic theories, cone equivalence, and applications to witnessing nonclassicality**; 出版社抄録・著者原稿。[DOI](https://doi.org/10.1103/PhysRevA.107.062203) | 実験で到達できる状態・効果を accessible fragment として定式化し、cone equivalence と埋込可能性の関係を扱う。抽象 prepare–measure 理論。 | 「accessible fragment」自体は既知。制約された生体 interface への仕様化が追加対象。 |
| Schmid et al. (2025), **Shadows and subsystems of generalized probabilistic theories: when tomographic incompleteness is not a loophole for contextuality proofs**; 出版書誌・著者公開本文。[DOI](https://doi.org/10.22331/q-2025-10-13-1880)、[原稿](https://arxiv.org/abs/2409.13024) | GPT subsystem と shadow を定義し、tomography 不完全性が contextuality 判定を変える場合と、relative tomographic completeness により判定が保たれる場合を区別する。 | arbitrary quotient と subsystem を混同できない。「深い古典記述がある」だけでも「粗視化した」だけでも決着しない。神経固有の結論はない。 |
| Müller & Garner (2023), **Testing Quantum Theory by Generalizing Noncontextuality**; 出版社本文。[DOI](https://doi.org/10.1103/PhysRevX.13.041001) | effective theory と fundamental theory の simulations/embeddings を、univalence と近似誤差で区別する。古典 embedding の特殊例で Spekkens 非文脈性に接続。 | classical simulation と非文脈 representation の区別は既知。神経系でどの同値性を保存するかを固定する必要がある。 |
| Khrennikov, Ozawa, Benninger & Shor (2025), **Coupling quantum-like cognition with the neuronal networks within generalized probability theory**; 出版社抄録・arXiv v2 本文。[DOI](https://doi.org/10.1016/j.jmp.2025.102923)、[本文](https://arxiv.org/html/2411.00036v2) | 神経の通信重み行列を ordered linear state space に置き、effects と instruments による順序・反復・干渉効果をモデル化。実測 connectome 再構成ではない。v2 §3 の非負正規化行列の状態集合は明示的に simplex。 | **神経ネットワーク + GPT は直接先行あり。** 本計画の候補差分は固定した古典 dynamics からの tomography、両側の simplex 判定、容量との制御比較。論文中の quantum-like は主に instrument による順序・干渉等を指し、本計画の contextuality 判定と同義でない。 |
| Spekkens (2005), **Contextuality for preparations, transformations, and unsharp measurements**; 出版社・著者抄録。[DOI](https://doi.org/10.1103/PhysRevA.71.052108) | operational equivalence を ontological representation が保存する要請を、準備・測定・変換へ一般化する。量子理論の no-go 例。 | 採用する定義の出典。prepare–measure テストだけから transformation noncontextuality を結論しない。 |
| Spekkens et al. (2009), **Preparation Contextuality Powers Parity-Oblivious Multiplexing**; 著者原稿・出版社書誌。[DOI](https://doi.org/10.1103/PhysRevLett.102.010401) | parity を隠す通信制約下で bit の retrieval 成功率を制限する非文脈性不等式と量子実験。 | contextuality と計算／情報タスクの接続は既知。ただしこのタスクの free class・制約を reservoir 全般に移せない。 |
| Mazurek et al. (2016), **An experimental test of noncontextuality without unphysical idealizations**; 出版社書誌・2021 年論文からの方法上の参照。[DOI](https://doi.org/10.1038/ncomms11780) | 不完全な実験手続きでも非文脈性を検査するため、実現した統計と operational equivalence の関係を扱う光学実験。 | 近い確率を恣意的に同値と宣言しないための方法上の先行。神経 dynamics の tomography や容量の結果ではない。 |
| Sulis (2021), **Contextuality in Neurobehavioural and Collective Intelligence Systems**; 出版社本文の公開部分。[DOI](https://doi.org/10.3390/quantum3040038) | neurobehavioural systems・社会性昆虫を、process algebra と contextual probability で論じる。 | 概念的類似。文脈依存、結合分布、process algebra の議論と、Spekkens 同値関係を保つ simplex embedding の証明を分ける。 |
| Wallentin (2021), **Contextuality in Classical Physics and Its Impact on the Foundations of Quantum Mechanics**; 出版社本文の公開部分。[DOI](https://doi.org/10.3390/e23080968) | 古典系の観測文脈と Koopman–von Neumann 表現を通して確率と干渉を論じる。 | 概念的類似。一般の「classical contextuality」記述を本計画の generalized contextuality の定理として採用しない。 |
| Staley (2026), **Contextuality Signatures in Organoid Intelligence: Operationalizing Quantum-Like Cognition in Wetware**; 2026-05-11 v1 本文。[原稿](https://www.preprints.org/manuscript/202605.0579) | 培養神経・organoid/MEA を想定し、CbD、不等式、順序、反復、干渉の検査を提案。実験結果ではなく未査読の理論仕様。 | 生物 neural substrate の operational contextuality 提案にも隣接先行あり。CbD と Spekkens の判定対象は同一でない。実測 connectome と GPT tomography の結合を示すものではない。個々の比較主張は本計画で採用しない。 |
| Kominis et al. (2026), **Searching for Quantum Effects in the Brain: A Bell-Type Test for Nonclassical Latent Representations in Autoencoders**; 著者抄録、全文未精査。[原稿](https://arxiv.org/abs/2601.10588) | autoencoder の decoding contexts に対する単一の正の latent distribution の整合性を Bell 型制約で問う未査読提案。 | 表現と測定制約の関係は近いが、実測 connectome の GPT fragment 判定とは区別する。題名の物理的含意を本計画の結論に持ち込まない。 |

## 計算容量・予測状態・幾何に関する先行研究

| 文献・確認範囲 | 問題・formalism・基質 | 本計画との関係 |
|---|---|---|
| Dambre et al. (2012), **Information processing capacity of dynamical systems**; 出版社本文と著者機関書誌。[DOI](https://doi.org/10.1038/srep00514) | 入力履歴の直交ターゲットを線形 readout で再構成する容量を定義し、適切な仮定で総容量を観測状態の独立成分数で制約する。一般 dynamical systems。 | memory/nonlinear capacity の定義を借りる。$d_{\rm op}$ と利用する readout feature 数、有限標本の effective rank は同じでない。 |
| Singh, James & Rudary (2004), **Predictive State Representations: A New Theory for Modeling Dynamical Systems**; UAI 公開原稿。[本文](https://web.eecs.umich.edu/~baveja/Papers/uai2004psr.pdf)、[書誌](https://arxiv.org/abs/1207.4167) | future tests の結果の予測を state にし、history–test system-dynamics matrix とその rank を使って離散動的系を記述。 | **未来の実験に対する予測で状態を定義する発想は既知。** GPT の状態・効果の凸構造、埋込制約、神経 capacity への接続を別途示す必要。arXiv 登録は 2012 年だが論文は UAI 2004, pp. 512–519。 |
| Crutchfield & Young (1989), **Inferring Statistical Complexity**; 著者公開論文・出版社書誌。[DOI](https://doi.org/10.1103/PhysRevLett.63.105) | 測定系列から予測構造と最小記述を再構成し、統計的複雑性を扱う computational mechanics の初期研究。 | 将来統計による履歴の集約に先行。統計的複雑性、operational dimension、実用タスク容量を混同しない。 |
| Cohen et al. (2020), **Separability and geometry of object manifolds in deep neural networks**; 出版社公開内容・著者コード概要。[DOI](https://doi.org/10.1038/s41467-020-14578-5) | 人工視覚ネットワークの object manifolds の幾何と線形分類容量の関係を解析。 | **神経表現の幾何と性能の関係も既知。** GPT tomography との相違は、状態だけでなく accessible effects と確率対を共同で扱うこと。 |
| Mijalkov et al. (2025), **Computational memory capacity predicts aging and cognitive decline**; 出版社本文。[DOI](https://doi.org/10.1038/s41467-025-57995-0) | 実測 human anatomical connectomes を reservoir にし、linear memory capacity と年齢・認知等の関連を評価。 | 実測構造と計算容量の統計的関係に直接先行。GPT geometry を導入すれば説明が改善するかは、本計画で反証可能に比較すべき点。 |

## 文献ごとの機能比較

「GPT」は明示的な GenProbTheory 使用、「実測 C」は biological connectome の使用、「判定」は simplex embeddability または明示的 contextuality の解析を示す。「予測状態」は類似する確率モデルであり、GPT 使用の肯定ではない。

| 文献 | GPT | 実測 C | simplex/contextuality | 計算容量・タスク測定 |
|---|---|---|---|---|
| Schmid 2021 | あり | なし | simplex 対応定理 | なし |
| Mazurek 2021 | あり | なし | 非文脈性に関する境界 | 神経容量なし |
| Selby 2024 | あり | なし | 埋込 LP・指定ノイズ | 神経容量なし |
| Aloy 2026 | あり | なし | 時間に伴う文脈性判定 | 神経容量なし |
| Suárez 2024 | なし | あり | なし | memory capacity・タスク性能 |
| Costi 2025 | なし | あり | なし | 時系列予測誤差・過学習 |
| Berg 2026 | なし | あり | なし | なし |
| MaleCNS 公式資料 | なし | あり | なし | なし |
| Shiu 2024 | なし | あり | なし | sensorimotor 予測検証；一般容量ではない |
| Selby 2023 | あり | なし | fragment の埋込理論 | なし |
| Schmid 2025 | あり | なし | shadow/subsystem の理論 | なし |
| Müller–Garner 2023 | あり | なし | embedding と非文脈性 | reservoir benchmark なし |
| Khrennikov 2025 | あり | なし | instrument 効果；simplex LP なし | conceptual；神経容量 benchmark なし |
| Spekkens 2005 | operational theory | なし | 一般化文脈性の定義・no-go | なし |
| Spekkens 2009 | operational theory | なし | preparation contextuality | POM 通信課題 |
| Mazurek 2016 | operational/GPT 方法 | なし | 実験的非文脈性検査 | 神経容量なし |
| Dambre 2012 | なし | なし | なし | 非線形情報処理容量 |
| Singh 2004 | 予測状態 | なし | なし | 表現力・モデル dimension |
| Crutchfield 1989 | なし | なし | なし | 統計的複雑性；task capacity と別 |
| Cohen 2020 | なし | なし | なし | manifold 分類容量 |
| Mijalkov 2025 | なし | あり | なし | linear memory capacity |
| Sulis 2021 | GPT fragment 構成なし | なし | 別定義の文脈性論 | reservoir 容量測定なし |
| Wallentin 2021 | GPT fragment 構成なし | なし | 別定義の文脈性論 | なし |
| Staley 2026 preprint | GPT への言及；再構成なし | なし | CbD 等の提案、実測判定なし | 容量測定なし |
| Kominis 2026 preprint | 抄録で確認できず | 抄録で確認できず | latent Bell 型提案 | 容量測定は未確認 |

## 検索記録と未解決部分

指定された八つの語を検索し、必要に応じ引用符あり／なしと関連論文の引用を追跡した。

| 検索語 | 得られた主要な方向・除外理由 |
|---|---|
| `generalized probabilistic theory neural network` | Khrennikov 2025 が直接近接。通常の probabilistic neural network と Transformer の GPT を除外。 |
| `contextuality neural dynamics` | Sulis、認知・neurodynamics 文献。定義が Spekkens のものか本文で区別。 |
| `operational state space neuroscience` | 神経集団表現・operational manifold 文献。活動／hyperparameter manifold と確率同値商は別物。 |
| `non-Kolmogorov neural model` | quantum cognition に加え Kolmogorov–Arnold network が混入。後者は確率論の contextuality と無関係。 |
| `classical system generalized contextuality coarse graining` | effective embedding、GPT subsystem/shadow、relative tomography を追加確認。 |
| `effective contextuality classical dynamics` | Wallentin 等の古典文脈性議論。単なる文脈依存・干渉と generalized noncontextuality を区別。 |
| `connectome reservoir computational capacity` | conn2res、Costi、Mijalkov。構造・容量の関係に広い直接先行。 |
| `neural manifold operational geometry` | Cohen 等の幾何と容量研究。画像上の manifold を GPT 状態空間と同一視しない。 |

さらに `connectome simplex embeddability`、`connectome generalized probabilistic`、`neural GPT tomography`、`reservoir generalized contextuality`、`neural simplex-embeddability`、`connectome generalized noncontextuality` を調べた。Gaussian-process tomography など略語の別義が混入した。Staley と Kominis の 2026 年原稿はこの追加探索で見つかり、査読済みの定理と分けて記録した。

残る不確実性は、未索引・異分野用語・新刊・学位論文・非英語文献に同じ組合せがある可能性、Khrennikov 刊行版と引用する後続論文の詳細比較、2026 年の神経 contextuality preprint の精査である。投稿前には検索式・データベース・対象期間を固定した追跡調査が必要。本フェーズの最強の結論は、**既知の枠組みを組み合わせた研究設計の候補が残るが、理論的新規性も生物学的現象もまだ実証していない**、である。
