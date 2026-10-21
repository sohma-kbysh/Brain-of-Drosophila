# 操作的幾何の小さな数学検算

Python 標準ライブラリの `fractions` と有限集合だけを使う。パッケージの追加、MaleCNS データ、神経シミュレーション、乱数標本化は不要である。リポジトリのルートから実行する。

```bash
python theory_examples/operational_geometry_sanity.py
```

検算対象は次の通り。

* 古典単体上の二つの置換が非可換になること。
* 同じ正方形の状態集合に、弱い効果を付けると状態と効果の同時単体埋込みが成立すること。
* 古典二ビットの制限読出しで even/odd 混合の観測確率が等しく、微視分布が異なること。
* parity 等価性を満たす事後分布多面体の全 4 頂点と、全 4 個の決定論的読出しについて、非文脈的成功率の最大値が厳密に `3/4` であること。
* 即時読出しの商が将来変換に対して閉じない例と、strong lumpability を満たす分割の例。

有理数の厳密演算であり、ノイズからの文脈性推定や数値最適化は実装しない。一般の場合の証明、仮定、出典は [03_classicality_and_contextuality.md](../docs/operational_gpt/03_classicality_and_contextuality.md) に記載する。この例を生物学的発見、新しい定理、計算優越と解釈しない。
