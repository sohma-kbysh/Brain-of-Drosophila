# Limitations

What this repository cannot tell you, and why.

---

## The headline

Simulation scale is not biological validation:

```
simulation scale  !=  biological validation
```

Running 166,691 neurons, millions of synapses and millions of parameter points
would not make any of the speculative assumptions more likely to be true. A
larger machine buys a sharper feasibility boundary, not a stronger biological
claim.

---

## 1. Physical limitations

**No quantum memory in Drosophila has ever been measured.** Every coherence time
here is a proxy from a different organism, a different molecule, or a different
temperature -- or is openly speculative. See `docs/assumptions.md` A-2.

**Arbitrary write is assumed and unavailable.** A-1. This is the assumption that
would invalidate the most if false.

**`T2_CPMG` is not a storage lifetime.** The cryogenic 16 us figure is measured
under dynamical decoupling. A free-induction storage time would be shorter, so
the cryogenic regime here is optimistic.

**No thermal or environmental model.** The environment enters only as (T1, T2).
Temperature dependence, spectral diffusion, non-Markovian dynamics, and the
possibility that an adversary *slows* decoherence by cooling or isolating the
host are all unmodelled -- and the last would favour Eve.

**No mechanism for how a state would be written or read.** Optical addressing,
microwave control and their invasiveness are entirely outside the model.

## 2. Cryptographic limitations

**These are toy protocols.** Not the Broadbent-Lord uncloneable encryption
scheme; not Broadbent-Islam certified deletion; no security proof; no
composability analysis.

**Protocol B's pre-key privacy comes from the pad, not the physics.** The
one-time pad would give the same zero-information result with classical bits.
The quantum content shows up only after key reveal, and there the SWAP control
shows it can be defeated outright.

**Product states only.** A-5. Uncloneability-style constructions need entangled
encodings, which are out of scope, so the tradeoff curves here are for
independent carriers.

**Only single-qubit adversary interactions.** Eve attacks each carrier
independently. Collective measurements across carriers -- which for some tasks
strictly outperform independent ones -- are not implemented, so Eve's power may
be *understated*.

**Detection is measured, not certified.** Disturbance shows up in Bob's error
rate. There is no protocol that turns that into a verifiable tamper certificate.

**Thresholds are choices.** A-14. `BER <= 1e-3` and `I(M:E) <= 0.01` are
configurable modelling parameters. The regime boundaries move with them.

**No finite-key analysis, no error correction, no privacy amplification.** A real
protocol would need all three, and they would change the usable window.

## 3. Biological limitations

**MaleCNS supplies anatomy, nothing else.** No claim is made that any neuron
contains a quantum memory, expresses cryptochrome in a quantum-relevant way, or
does anything quantum at all.

**The neural model is transferred, not validated.** A-7. Shiu et al. validated
on the female FlyWire brain.

**Neuromodulators are simplified to excitatory.** A-8. Dopamine, octopamine and
serotonin are not simply excitatory in real circuits.

**No neuromodulation, plasticity, development, or behavioural state.** The
network is static and deterministic given its seed.

**No wet-lab pathway.** This repository does not design an experiment and is not
a protocol for one.

**Offline runs use a synthetic graph.** A-13. Without a `NEUPRINT_TOKEN`, every
network-level number describes a tuned placeholder, not MaleCNS. Such results
are stamped `synthetic` in metadata and on the figures, and the
connectome-specific questions remain open until a real subgraph is used.

## 4. Coupling limitations

**The quantum-to-neural coupling is the weakest link.** A-10. No evidence, no
calibration, disabled by default. Experiment 9 locates a sensitivity threshold;
it does not predict an effect.

**A single scalar readout.** `y = Tr[M rho]` with one Pauli observable is the
crudest possible model of a molecular readout.

**Instantaneous, memoryless coupling.** No transduction delay, no saturation, no
receptor dynamics.

## 5. Computational limitations

**Tier discipline.** A result obtained at Tier 1 (100-2,000 neurons) must not be
described as whole-brain. The tiers are: 0 = unit physics, 1 = local circuit,
2 = mesoscale (2k-50k), 3 = whole MaleCNS (166,691).

**Betweenness is approximate.** A-15.

**Brian2 spike-time recording is all-or-nothing.** `SpikeMonitor.record` is a
boolean and subgroups need contiguous indices, so spike *times* for an arbitrary
host set require recording the whole group and filtering. Above 50,000 neurons
this is skipped and only per-neuron counts are kept.

**The graph-shuffle controls are heuristic.** Double-edge swaps preserve in- and
out-degree exactly but do not control for reciprocity, motifs, or spatial
structure.

## 6. What would most change the conclusions

In order of leverage:

1. **A measured coherence time for an addressable memory in a nervous system.**
   Everything downstream is conditional on A-1 and A-2.
2. **A real MaleCNS subgraph instead of the synthetic substitute.** The host-
   placement question is the one that genuinely needs the measured connectome.
3. **Entangled encodings.** Would test whether the information-disturbance curve
   moves, and is the first step toward a real uncloneable-encryption analogue.
4. **A bound on adversarial extraction fidelity.** The whole security question
   turns on whether the unrestricted-extraction regime is physically reachable.
   If it is, natural decoherence provides no secrecy and the interesting
   parameter region is empty.
5. **Any calibration at all for kappa.** Without it, Experiment 9 is a
   sensitivity study rather than a prediction.
