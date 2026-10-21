# Threat model

The single most important distinction in this project:

> ## no-cloning != no-theft

An adversary who cannot *copy* an unknown quantum state may still be able to
*take* it. If Eve can transfer the carriers into her own memory, the host's
subsequent decoherence protects nothing: it destroys the copy she left behind,
not the one she kept. This document states where the toy protocol works and
where it fails, and the failure is treated as a result, not as something to be
hidden.

---

## Two access regimes

### 1. Constrained-access regime

Eve may:

- measure the carriers (projectively, in any basis)
- interact weakly (a POVM of strength `eta < 1`)
- couple coherently to an ancilla for a limited duration
- keep whatever she obtained, in a memory of specified quality
- learn the key `(s, theta)` later

Eve may **not** transfer the host state into her own memory.

In this regime the protocol behaves as conjugate coding should. An adversary who
gains information leaves a disturbance, the two rise together, and the tradeoff
is smooth in `eta`.

### 2. Unrestricted-extraction regime

Eve may additionally perform

```
rho_Q (x) |0><0|_E   ->   |0><0|_Q (x) rho_E
```

a perfect state transfer, implemented as `swap_theft`.

**In this regime the scheme fails completely, and no amount of host decoherence
repairs it.** The measured outcome: host information falls to zero immediately
while Eve's stays flat at its initial value forever. The `swap_negative_control`
experiment exists to make this unavoidable in the results.

---

## Timeline

```
t = 0         Alice writes the carriers into the host
t = t_attack  Eve interacts
t = t_read    Bob reads, in the basis he knows
t = t_reveal  the key becomes public; Eve extracts from what she kept,
              which has meanwhile decohered in HER memory
```

Eve's memory quality is a parameter, not an assumption:

| model | meaning |
|---|---|
| `ideal` | perfect memory. **The default**, because a security claim should assume the strong adversary. |
| `same_as_host` | Eve is no better off than the biology |
| `better` | Eve's memory beats the host by a configurable factor |

---

## What is measured, and against which adversary

| quantity | adversary assumed |
|---|---|
| `eve_pguess_before_key` | optimal (Helstrom) discrimination of the two message hypotheses |
| `eve_pguess_after_key` | optimal discrimination once `(s, theta)` are known |
| `eve_holevo_before_key` | Holevo bound -- an **upper bound** on accessible information |
| `bob_ber`, `bob_exact_recovery` | exact, closed-form |
| `bob_disturbance_ber` | BER with the attack minus BER without, at the same storage time |

Reporting the *optimal* adversary rather than one chosen measurement keeps Eve's
power from being understated. Holevo information is reported separately from
achieved mutual information and the two are never conflated.

---

## What the protocol does and does not provide

| property | status |
|---|---|
| privacy before key reveal (Protocol B) | **Yes, information-theoretically** -- but from the one-time pad, not from quantum mechanics |
| privacy before key reveal (Protocol A) | **No.** Breidbart recovers the bit with p = 0.854 and no key |
| information-disturbance tradeoff | **Yes**, in the constrained-access regime |
| detection of an interacting adversary | **Partially** -- disturbance is visible in Bob's error rate |
| uncloneability | **Not implemented.** Product-state encoding only; this is not the Broadbent-Lord scheme |
| certified deletion | **No, and never claimed.** See below |
| security against state theft | **No.** See the unrestricted-extraction regime |

---

## Why natural decoherence is not certified deletion

Certified deletion means the receiver can produce a *certificate* that the
ciphertext is gone, and the certificate is verifiable. Natural decoherence
provides neither:

1. **No certificate.** Nothing is produced that anyone can check.
2. **No guarantee about copies.** Decoherence acts on the host. An adversary who
   already transferred the state holds something decoherence cannot reach.
3. **Not adversarially controlled.** The erasure time is set by the environment,
   not by a protocol participant, and an adversary who can slow decoherence --
   by cooling, by isolation -- extends the window at will.

The repository therefore uses the term **self-erasure** for what the host does,
and reserves *deletion* for the cryptographic property it does not have.

Quantitatively (Experiment 6, coherent ancilla attack at `t_attack = 0.05 T2`):

| t / T2 | host information | Eve information |
|---|---|---|
| 0.1 | 0.42 bits | 0.09 bits |
| 1 | 0.06 bits | 0.09 bits |
| 10 | ~0 bits | 0.09 bits |
| 50 | ~0 bits | 0.09 bits |

The host forgets. Eve does not.

---

## Detection

`tamper_alarm` mode asks whether Eve's disturbance could be *noticed* through
the biology, by coupling the molecular readout shift into host-neuron drive and
measuring the downstream response. This is the most speculative claim in the
project and is disabled by default. A detection probability computed from it is
a statement about a model, not about a fly.

---

## Out of scope

- side channels of any kind (timing, optical, thermal)
- an adversary who can modify the organism, the apparatus, or the code
- composability with any other protocol
- multi-round or interactive protocols
- authentication -- Eve is passive-to-active on the quantum channel only
- an adversary who slows decoherence by manipulating the host's environment,
  which would extend the storage window in her favour and is not modelled
