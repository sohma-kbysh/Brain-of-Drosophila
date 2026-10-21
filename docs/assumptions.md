# Assumptions

Every assumption the results depend on, with its evidence class. Sorted by how
much damage it would do if false.

Evidence classes: **MEASURED** (reported in a cited experiment, in the regime
cited) | **PROXY** (order-of-magnitude stand-in from a different system) |
**TRANSFERRED** (model validated elsewhere, applied here) | **SPECULATIVE** (no
experimental support) | **CONVENTION** (a modelling choice).

---

## A-1. Arbitrary quantum states can be written into a biological host

**SPECULATIVE. The load-bearing assumption of the whole project.**

`AbstractEphemeralMemory` grants the ability to write an arbitrary
`alpha|0> + beta|1>` into a molecular subsystem inside a chosen neuron and read
it out later. No known biological system offers this.

- Feder et al. demonstrate *coherent control* of an EYFP triplet, which is less
  than an addressable arbitrary-write memory.
- Burd et al. manipulate radical-pair dynamics *in vivo*, but a radical pair is
  born in a specific photochemical state.

*If false:* the cryptographic layer describes nothing physically realisable, and
only the network-science results (host placement, propagation) survive. The
capability is deliberately isolated in one class so the assumption is visible.

## A-2. Coherence times

**PROXY to SPECULATIVE, depending on the regime.**

| regime | T2 | class |
|---|---|---|
| in vivo | 4 - 100 ns | PROXY: Burd et al. imply >4 ns in *C. elegans*; Feder et al. report room-temperature polarisation loss <100 ns in EYFP |
| cryogenic | 1 - 20 us | MEASURED for EYFP, but `T2_CPMG` is a *decoupled* coherence time, so treating it as a storage lifetime is optimistic |
| future | 100 us - 10 ms | SPECULATIVE |

No T2 has been measured for any quantum memory in Drosophila, because no such
memory exists.

## A-3. Eve's memory is ideal

**CONVENTION**, and deliberately the pessimistic one. A security claim should
assume the strong adversary. Configurable via `crypto.eve_memory`.

## A-4. The message ensemble is uniform and known

**CONVENTION.** Standard for this kind of analysis, and what makes the Holevo
and Helstrom quantities well defined.

## A-5. Product-state encoding

**CONVENTION**, and a real scope limit. Each carrier is independent, which is
what makes n = 128 affordable. Entangled encodings -- necessary for a genuine
uncloneable-encryption construction -- are out of scope, so the protocol
approximates rather than implements those schemes.

## A-6. No Hamiltonian during storage

**CONVENTION.** The encoding basis is taken to be the memory eigenbasis, so a
resonant Hamiltonian would add only a known, invertible rotation. A real system
with an unknown or drifting Hamiltonian would be worse off than modelled.

## A-7. Shiu et al. LIF parameters apply to MaleCNS

**TRANSFERRED.** The model was validated on the **female FlyWire** brain.
Applying the same parameterisation to MaleCNS connectivity is a model-transfer
assumption.

> Neural dynamics are a Shiu-et-al.-style LIF transfer model applied to MaleCNS
> connectivity; this does not constitute experimental validation of MaleCNS
> neural dynamics.

## A-8. Neurotransmitter sign policy

**TRANSFERRED**, with an acknowledged simplification. GABAergic and
glutamatergic inhibitory; cholinergic excitatory. Dopamine, octopamine and
serotonin are treated as **excitatory**, following Shiu et al.

That last is a simplification, not physiology: these are predominantly
neuromodulatory and their net effect is receptor- and state-dependent. Neurons
with low-confidence predictions are handled by an explicit configurable policy
(`ignore` / `excitatory` / `exclude`).

## A-9. Host neurons

**SPECULATIVE placement, honestly labelled.** The default `engineered` mode
represents a hypothetical scenario in which a spin qubit has been genetically
expressed in a chosen population.

**CRY expression is never inferred from MaleCNS annotations.** MaleCNS records
anatomy, connectivity and a predicted transmitter; nothing about cryptochrome or
spin physics. Curated host sets require a user-supplied CSV.

`DNge104` appears only because the MaleCNS documentation uses it as an API
example. No quantum-biological significance is claimed for it.

## A-10. Quantum-to-neural coupling

**SPECULATIVE, uncalibrated, disabled by default.**

There is no evidence that a molecular spin state influences neuronal firing in
Drosophila. `kappa` has no empirical calibration and is swept dimensionlessly.
Experiment 9 locates a *sensitivity threshold* -- the gain at which an effect
would become visible -- rather than predicting an effect.

## A-11. Radical-pair constants

**PROXY where present, UNKNOWN where absent.** Hyperfine (~1 mT) and
recombination (~1e6 /s) are generic organic-radical-pair magnitudes, not
measured constants for any named cryptochrome. Exchange and dipolar couplings
default to `None`, meaning unknown, and their Hamiltonian terms are **omitted
rather than guessed**.

## A-12. EYFP spectroscopic constants

**UNKNOWN.** Zero-field splitting (D, E), g-factor and hyperfine tensor were not
verified against the primary source, so `spin_one_hamiltonian` raises
`NotImplementedError` instead of running on invented numbers. Only the coherence
envelope is used.

## A-13. Synthetic connectome (when used)

**CONVENTION, tuned placeholder.** Used when no `NEUPRINT_TOKEN` is available.
Connectivity (~30 partners, ~18 synapses/connection, heavy-tailed) was chosen so
the network propagates at a few Hz under the Shiu parameters. These are **not**
MaleCNS statistics. Every such graph is stamped `synthetic: True` and every
figure built from one says so.

## A-14. Detection thresholds

**CONVENTION.** `BER <= 1e-3`, `I(M:E) <= 0.01`, `P_detect >= 0.95` are
configurable modelling choices, **not universal cryptographic standards**. The
regime boundaries in the feasibility map move if they move.

## A-15. Betweenness centrality is approximated

**CONVENTION.** Exact betweenness is O(NM) and infeasible at connectome scale,
so pivots are sampled (seeded from the run's RNG, so still reproducible).
