"""有限の人工例による厳密検算。神経シミュレーションや一般 LP ではない。"""

from fractions import Fraction as F
from itertools import product


def dot(left, right):
    return sum((x * y for x, y in zip(left, right)), F(0))


def matvec(matrix, vector):
    return tuple(dot(row, vector) for row in matrix)


def check(condition, description):
    if not condition:
        raise AssertionError(description)


def classical_noncommutativity():
    a = (1, 0, 2)
    b = (0, 2, 1)
    check(a[b[0]] == 1 and b[a[0]] == 2, "古典置換の非可換性")
    print("PASS: 古典置換 AB(0)=1, BA(0)=2")


def square_embedding():
    # 効果を (定数, x係数, y係数) で表す。全極点を確認すれば凸包に及ぶ。
    half, quarter = F(1, 2), F(1, 4)
    effects = (
        (0, 0, 0), (1, 0, 0),
        (half, quarter, 0), (half, -quarter, 0),
        (half, 0, quarter), (half, 0, -quarter),
    )
    # 線形写像の積が恒等写像であることを標準基底でも確認する。
    state_map = ((quarter, quarter, 0), (quarter, -quarter, 0),
                 (quarter, 0, quarter), (quarter, 0, -quarter))
    effect_map = ((1, 2, 0), (1, -2, 0), (1, 0, 2), (1, 0, -2))
    basis = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    for left in basis:
        for right in basis:
            check(dot(matvec(effect_map, left), matvec(state_map, right))
                  == dot(left, right), "双線形確率則の完全な保存")
    for x, y in product((-1, 1), repeat=2):
        state = (1, x, y)
        mu = matvec(state_map, state)
        check(sum(mu) == 1 and min(mu) >= 0, "単体状態の規格化")
        check(mu[0] + mu[1] == mu[2] + mu[3] == half, "単体の断面")
        for effect in effects:
            response = matvec(effect_map, effect)
            check(all(0 <= value <= 1 for value in response), "効果の有界性")
            check(dot(response, mu) == dot(effect, state), "許可確率の保存")
    check(matvec(effect_map, (1, 0, 0)) == (1, 1, 1, 1), "単位効果")
    print("PASS: 非単体の正方形と弱い効果は Δ4 に同時埋込み可能")


def parity_quotient_and_bound():
    bits = tuple(product((0, 1), repeat=2))  # 00, 01, 10, 11
    even = (F(1, 2), 0, 0, F(1, 2))
    odd = (0, F(1, 2), F(1, 2), 0)
    q = ((1, 1, 1, 1), (0, 0, 1, 1), (0, 1, 0, 1))
    check(even != odd, "微視分布は異なる")
    check(matvec(q, even) == matvec(q, odd) == (1, F(1, 2), F(1, 2)),
          "parity 混合の操作的同値性")
    check(matvec(q, (1, -1, -1, 1)) == (0, 0, 0), "失われる parity 方向")
    readout_table = []
    for index in range(4):
        microstate = tuple(int(index == other) for other in range(4))
        readout_table.append(matvec(q, microstate)[1:])
    perfect_score = sum(F(readout_table[i][j] if bits[i][j]
                          else 1 - readout_table[i][j], 8)
                        for i in range(4) for j in range(2))
    check(perfect_score == 1, "二ビット古典装置の完全読出し")
    # r00+r11=r01+r10=1/2 の全頂点は、even/oddから各1点を選ぶ4通り。
    posterior_vertices = []
    for even_index, odd_index in product((0, 3), (1, 2)):
        posterior = [F(0)] * 4
        posterior[even_index] = posterior[odd_index] = F(1, 2)
        posterior_vertices.append(tuple(posterior))
    scores = []
    for posterior, response in product(posterior_vertices, product((0, 1), repeat=2)):
        score = sum(posterior[i] * F(int(response[j] == bits[i][j]), 2)
                    for i in range(4) for j in range(2))
        scores.append(score)
    check(max(scores) == F(3, 4), "非文脈性境界 3/4")
    # 閾値の状態–効果表は前関数の正例と同じ。η=1/2では正解率3/4。
    check((1 + F(1, 2)) / 2 == max(scores), "弱い効果で境界を達成")
    print("PASS: parity 混合は商で同一、微視的には別分布")
    print("PASS: 4 個の事後分布頂点 × 4 個の応答で非文脈境界 = 3/4; 装置 = 1")


def quotient_dynamics():
    # 列が遷移元。即時読出しは状態1だけ検出し、0と2を同一視する。
    q = ((1, 1, 1), (0, 1, 0))
    k = ((0, 0, 0), (1, 1, 0), (0, 0, 1))
    state0, state2 = (1, 0, 0), (0, 0, 1)
    check(matvec(q, state0) == matvec(q, state2), "即時同値")
    check(matvec(q, matvec(k, state0)) != matvec(q, matvec(k, state2)),
          "将来実験で分離")
    # 分割 {0,1}, {2} に対し各ブロックへの遷移確率が等しい核。
    c = ((1, 1, 0), (0, 0, 1))
    lumpable_k = ((F(1, 2), 0, 0), (0, F(1, 2), 0),
                  (F(1, 2), F(1, 2), 1))
    coarse_k = ((F(1, 2), 0), (F(1, 2), 1))
    for state in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        check(matvec(c, matvec(lumpable_k, state))
              == matvec(coarse_k, matvec(c, state)), "CK = Kbar C")
    print("PASS: 即時商の更新不成立例と strong lumpability の成立例")


def main():
    classical_noncommutativity()
    square_embedding()
    parity_quotient_and_bound()
    quotient_dynamics()
    print("全検算成功（有理数の厳密演算、外部依存なし）")


if __name__ == "__main__":
    main()
