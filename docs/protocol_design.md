# Protocol design: choosing and composing primitives

How the toy protocols were rebuilt into something that survives contact with an
adversary, and what each choice costs. Every claim here is produced by
`bioquantumfly sweep --experiment protocol_stack` and checked in
`tests/test_primitives.py`.

---

## The three failures to repair

The diagnostic protocols failed in three distinguishable ways. Naming them
separately matters, because each needs a different primitive.

| # | failure | measured |
|---|---|---|
| F1 | The strict error budget throws away storage time | usable window = 0.002 T2 |
| F2 | The adversary keeps a fraction of a bit forever | 0.15 bits at t = 50 T2 |
| F3 | "It decohered" certifies nothing | host at 0 bits, Eve unchanged |

---

## What was chosen, and what was rejected

### F1 → error correction

The usable window is *linear* in the tolerated error rate, because
`t/T2 = -ln(1 - 2 BER)`. Demanding BER ≤ 1e-3 outright rather than tolerating
10% and reconciling it costs a factor of **111** in storage time, for a code
rate of 0.53 — a net **59×** in bit-seconds.

This is the single largest number in the whole analysis, and it is bookkeeping,
not physics. Any feasibility statement that omits error correction overstates
the coherence requirement by two orders of magnitude.

### F2 → privacy amplification

Reporting "Eve has 0.15 bits" and stopping is the wrong move: the leftover hash
lemma exists precisely to remove them. With min-entropy
`H_min(X|E) = -Σ log2 p_i` (exact for a product encoding) and reconciliation
leakage `f · n · H2(e)`, the extractable secret is

```
ell = H_min(X|E) - leak_EC - 2 log2(1/eps)
```

Experiment 10 shows this is the step that changes the verdict: with error
correction alone the secret is empty; adding privacy amplification yields ~700
bits from 1024 carriers at η = 0.25.

### F3 → certified deletion, not natural decay

Deletion has to be an *action* that produces checkable evidence. The mechanism
(after Broadbent–Islam, but not their scheme or proof): message bits sit in
computational-basis positions, check bits in Hadamard-basis positions; to
delete, measure **everything** in the Hadamard basis and return the outcomes.
The check positions reproduce deterministically; the message positions are
destroyed in the same stroke.

Measured behaviour:

| certificate from | accepted? |
|---|---|
| honest deleter | yes |
| cheater who kept the message | no (mismatch ≈ 0.5) |
| **a party who just waited for decay** | **no** |

The third row is the whole point. You cannot certify deletion by letting the
state rot, because the check correlations rot with it.

### The adversary → a noisy-storage assumption, and delayed key release

This project's own data chose this one. In Experiment 6 an adversary whose
memory matches the host's ends with **exactly zero** information, while an
ideal-memory adversary keeps hers forever. That is the noisy-storage /
bounded-quantum-storage model, and it fits unusually well here: an adversary
extracting a spin state from a warm cell faces the same decoherence physics as
the host, so the assumption is a claim about her apparatus rather than an
unfalsifiable computational one.

It buys a knob unavailable to computational security: **delay the key release.**
Eve cannot use her stored ancilla until (s, θ) are public, so every unit of
waiting decays her copy. Waiting is free.

```
required delay = T2_Eve * ln( (p0 - 1/2) / (p_target - 1/2) )
```

Measured, with a Breidbart-strength attack and Eve's memory equal to the host's:
waiting 5 × T2 drops her from 399 bits to 0.016, and the extractable secret
rises from 5.6 bits to 782.

**The honest caveat.** Tolerable adversary advantage grows only *linearly* with
the delay: waiting 20 × T2 tolerates a memory 6.8× better than the host's, not
6.8 orders better. Patience does not substitute for the assumption.

### Rejected, and why

| primitive | why not |
|---|---|
| **Uncloneable encryption** (Broadbent–Lord) | Needs an oracle/QROM and entangled encodings. Out of scope for a product-state simulator, and the oracle assumption is not physically motivated here. |
| **Quantum one-time pad** | Perfect, but needs 2n key bits and full quantum control of the host — strictly more capability than we are already speculatively assuming. |
| **Decoy states** | Solves imperfect-source photon-number splitting. There is no source here; the carrier count is known exactly. |
| **Device independence** | Requires loophole-free Bell violation. Not remotely available in this setting. |
| **Quantum secret sharing across hosts** | Genuinely attractive against local extraction — see "next steps". Not implemented: it needs multi-host entanglement, which is outside the product-state scope. |

---

## The composed scheme (Protocol C)

```
    Alice                        host                         Bob
      |  conjugate encoding        |                            |
      |  message -> Z positions    |                            |
      |  checks   -> X positions   |                            |
      |--------------------------->|                            |
      |                       decoherence                       |
      |                            |------- read in theta ----->|
      |                            |                            |
      |<---------- error reconciliation (leaks f n H2(e)) ------>|
      |<---------- privacy amplification -> eps-secure key ----->|
      |                                                         |
      |  ... withhold (s, theta) for delta t ...                |
      |                            |<--- Hadamard certificate --|
      |  verify -> deletion certified                           |
```

Layer ablation, at η = 0.25 with the adversary's memory equal to the host's:

| layers enabled | secret bits |
|---|---|
| bare (Protocol B) | 0 |
| + error correction | 0 |
| + privacy amplification | 696 |
| + delayed key release | 869 |
| + certified deletion | 869 (deletion adds security, not key length) |

---

## The result that reframes the whole project

**State theft stops being fatal.** Under Protocol B, an ideal-memory adversary
who SWAPs the carriers wins outright — the central negative result of the
original analysis. Under Protocol C the same attack drives the error rate to
0.52, far past the ~11% one-way threshold, and **the round is discarded**:
0 of 225 swept configurations yielded a key.

Theft is defeated by *detecting* it and refusing to proceed. That is the
parameter-estimation step every real QKD protocol has and Protocol B lacked. The
failure was never a physical wall; it was a missing layer.

What remains true, and is not repaired by any of this: an adversary who takes
the state still *has* it. What changes is that she gets a string which was never
used to encrypt anything.

---

## Design requirements this produced

Concrete numbers a designer can act on:

1. **≥ 10⁴ check positions** for certified deletion at a 1% theft bound. Below
   about 10³ the binomial spread on the check set is wider than the signal and
   no tolerance separates an honest deleter from a partial cheater. This is why
   the 1024-carrier runs report deletion as not certifiable.
2. **Error rate below ~11%.** Past the one-way threshold no post-processing
   yields a secret; abort instead.
3. **Withhold the key for ≥ 5 × T2_Eve.** Cheap, and the only lever that touches
   an adversary who already holds the state.
4. **Bound the adversary's memory explicitly.** The scheme's security rests on
   it, so it belongs in the threat model as a stated assumption, not an
   afterthought — see `docs/threat_model.md`.
5. **Certification outlives readout by roughly 10×.** Reading needs near-perfect
   data; certifying needs only a statistically visible correlation. Budget the
   two windows separately.

---

## Next steps

1. **Secret sharing across k hosts.** The strongest remaining answer to local
   extraction: an adversary who reaches one neuron gets nothing. Needs
   multi-host encoding, which is beyond the current product-state scope.
2. **Entangled two-carrier encodings**, to test whether the
   information–disturbance curve moves — the first real step toward an
   uncloneable-encryption analogue rather than an approximation of one.
3. **A composable finite-key analysis.** The accounting here uses the standard
   asymptotic form with a finite-size term; it is a design tool, not a proof.
4. **A defensible bound on adversarial extraction fidelity.** Everything above
   rests on the noisy-storage assumption, and that number is the one an
   experimentalist would have to supply.
