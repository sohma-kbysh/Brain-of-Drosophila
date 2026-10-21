# Capability inventory

What this construction can and cannot do, sorted by what blocks it. Every entry
in the first table is implemented and tested in this repository; everything
below it is argued from the constraints, not measured.

**The single constraint that decides most of the list:** the carriers are
independent single-qubit stores. Nothing here entangles two molecular spins in
two different neurons. Any primitive that needs inter-host entanglement is out,
and any primitive that needs only single-qubit preparation and measurement is
potentially in.

---

## 1. Verified — implemented and tested

| capability | evidence | limits found |
|---|---|---|
| **Tamper-evident storage** | information–disturbance curve, Exp 4 | Eve's information and Bob's error rise together at a near-fixed ratio (~1.44) |
| **Key establishment (QKD-style)** | Protocol C, Exp 10 | ~700 bits from 1024 carriers at η = 0.25; aborts above ~11% QBER |
| **Detection of state theft** | Exp 7 + Protocol C | SWAP drives QBER to 0.52 → round discarded, 0 of 225 configurations issued a key |
| **Certified deletion** | `crypto.deletion` | honest accepted, cheater rejected, **a decayed state also rejected**; needs ≥ 10⁴ check positions |
| **Delayed key release** | `crypto.noisy_storage` | 5 × T2_Eve drops Eve from 399 bits to 0.016; tolerable adversary advantage grows only *linearly* with waiting |
| **Classical secret sharing across k hosts** | `crypto.secret_sharing` | below threshold leaks exactly zero; **no entanglement required** |

### What secret sharing buys, quantified

Splitting is the one place where separate neurons do cryptographic rather than
decorative work — and it defeats the attack that defeats everything else.

| hosts (k-of-k XOR) | effective BER (per-host 1%) | must compromise | P(detect \| full attack) |
|---|---|---|---|
| 1 | 0.010 | 1 | 0.50 |
| 3 | 0.029 | 3 | 0.88 |
| 5 | 0.048 | 5 | 0.97 |
| 10 | 0.092 | 10 | 0.999 |

Two costs, both real:

* **Error rates add.** A reconstructed bit is wrong when an odd number of shares
  erred, so k stores at rate e give ≈ k·e. At a 1% per-host rate the scheme
  crosses the ~11% abort threshold somewhere between k = 10 and k = 20 — a hard
  cap on the number of XOR shares.
* **The finite-size penalty is paid per store.** 2·log₂(1/ε) ≈ 66 bits each, so
  a fixed carrier budget yields less total secret as it is divided: 2048
  carriers give 1508 bits in one store but only 511 across sixteen.

**Threshold schemes dominate.** A t-of-k Shamir scheme improves security *and*
reliability at once, because it tolerates k − t lost stores:

| scheme | P(reconstruct), per-host 0.95 | adversary needs | tolerates losing |
|---|---|---|---|
| 3-of-3 | 0.857 | 3 | 0 |
| 3-of-5 | **0.999** | 3 | 2 |
| 5-of-5 | 0.774 | 5 | 0 |
| 5-of-10 | **~1.000** | 5 | 5 |

---

## 2. Feasible but not implemented — no entanglement needed

These need only single-qubit preparation and measurement, so nothing in the
physical model forbids them.

| capability | why it fits | what it would need |
|---|---|---|
| **Unforgeable expiring tokens** (Wiesner-style) | A token *is* a string of BB84 states plus a verification table. Optimal counterfeiting succeeds with probability (3/4)ⁿ, so 64 carriers give odds ~1 in 10⁸. | A verification interface. Note the token's life is the readable window: ~3.6 µs at cryogenic EYFP *with* error correction, 32 ns without. A credential that expires by physics rather than by policy. |
| **Bit commitment / oblivious transfer** | Impossible unconditionally, but achievable in the noisy-storage model — whose assumption this setting already makes and which the simulator already parameterises. | The formal noisy-storage analysis, which this repository does not reproduce. |
| **Randomness generation** | Measuring a conjugate-basis state yields true randomness. | Almost nothing; low value, but genuinely available. |

---

## 3. Blocked by the absence of inter-host entanglement

| capability | what it needs |
|---|---|
| **Quantum secret sharing of a quantum state** (Hillery–Bužek–Berthiaume; Cleve–Gottesman–Lo) | GHZ states or a quantum code across the shares. This is the variant the name usually means, and it is **not** what section 1 implements. |
| **Uncloneable encryption** (Broadbent–Lord) | Entangled encodings plus an oracle/QROM. |
| Teleportation, entanglement swapping | Shared entangled pairs between neurons. |
| Device-independent protocols | A loophole-free Bell violation. |
| Blind / delegated quantum computation | Universal quantum computation in the host. |

Entangling molecular spins in two different cells is a categorically stronger
capability than the single-spin control this project already assumes
speculatively. Treating it as available would hide the assumption that matters
most.

---

## 4. Blocked by coherence time

Anything needing storage beyond ≈ 0.22 × T2 (with error correction) or
≈ 0.002 × T2 (without). At the measured cryogenic EYFP value of 16 µs that is
3.6 µs and 32 ns respectively. In particular:

* Multi-round interactive protocols, which need several exchanges inside one
  coherence time.
* Storage across a behavioural timescale (~100 ms) — that needs T2 ≈ 0.45 s.

See `docs/protocol_design.md`: this gap is architectural, not fundamental. An
electron spin in a warm protein is the worst available place to store, and the
standard solid-state answer is to split the roles — electron spin as interface,
nuclear spin as memory.

---

## 5. The honest summary

What the construction actually is, on the evidence gathered:

> A **short-lived, tamper-evident, splittable classical store** with a real
> deletion certificate, whose security rests on a bound on the adversary's
> quantum memory.

That is a genuine and non-trivial set of properties. It is also considerably
less than "quantum cryptography in a fly": there is no entanglement, no
uncloneable encryption, no quantum secret sharing in the usual sense, and the
storage lifetime is microseconds at best under a coherence time that has never
been measured in an insect.

Every one of these capabilities remains conditional on assumption A-1 — that an
arbitrary quantum state can be written into a molecular subsystem inside a
chosen neuron — which no known biological system provides. See
`docs/assumptions.md`.
