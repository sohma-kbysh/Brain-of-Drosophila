# Sources

Full citations for every empirical claim the project relies on, plus an
explicit statement of what each source does and does not establish.

---

## Connectome

**Stuart Berg et al., "Sexual dimorphism in the complete connectome of the
Drosophila male central nervous system," *Cell*, 2026.**

- Dataset identifier: `male-cns:v1.0`
- Official neuPrint server: <https://neuprint.janelia.org>
- 166,691 neurons spanning brain and ventral nerve cord; 11,691 annotated types.

*Establishes:* measured anatomy, connectivity and predicted neurotransmitter
identity for the male CNS.

*Does not establish:* anything about quantum states, cryptochrome-based
magnetoreception, or spin physics in any of those neurons. This project uses
MaleCNS purely as a classical network substrate.

---

## Neural dynamics

**Philip K. Shiu et al., "A Drosophila computational brain model reveals
sensorimotor processing," *Nature*, 2024.
DOI: [10.1038/s41586-024-07763-9](https://doi.org/10.1038/s41586-024-07763-9)**

The LIF parameterisation used in `bioquantumfly.neural.lif`:

| parameter | value |
|---|---|
| V_resting, V_reset | -52 mV |
| V_threshold | -45 mV |
| R_mbr | 10 kohm cm^2 |
| C_mbr | 2 uF/cm^2 |
| T_mbr = R_mbr * C_mbr | ~20 ms |
| T_refractory | 2.2 ms |
| tau_synapse | 5 ms |
| T_delay | 1.8 ms |
| W_syn | 0.275 mV per synapse |
| baseline_rate | 0 Hz |

Sign policy: GABAergic and glutamatergic inhibitory, cholinergic excitatory;
dopamine, octopamine and serotonin simplified to excitatory.

*Establishes:* a whole-brain LIF model validated against the **female FlyWire**
brain.

*Does not establish:* that the same parameterisation describes MaleCNS. Applying
it here is a model-transfer assumption; see `docs/assumptions.md` A-7.

---

## Genetically encoded spin qubit

**Jacob S. Feder et al., "A fluorescent-protein spin qubit," *Nature*, 2025.
DOI: [10.1038/s41586-025-09417-w](https://doi.org/10.1038/s41586-025-09417-w)**

Empirical constraints used:

- EYFP triplet spin, optically addressable
- coherent microwave control demonstrated
- **T2_CPMG = 16 +/- 2 us at cryogenic temperature**
- coherent control inside mammalian cells at low temperature
- room-temperature ODMR in bacterial cells
- **room-temperature spin polarisation disappears within < 100 ns**

*Establishes:* that a genetically encodable, optically addressable, coherently
controllable protein spin exists.

*Does not establish:* an arbitrary-write quantum **memory**; a Drosophila
result; or that the cryogenic and room-temperature numbers may be interchanged.
`T2_CPMG` is a *dynamically decoupled* coherence time, so using it as a storage
lifetime is optimistic. The two regimes are kept separate in
`bioquantumfly.quantum.eyfp_proxy`, and the spectroscopic constants a full
spin-1 model would need (D, E, g-factor, hyperfine tensor) are recorded as
`unknown` rather than guessed.

---

## In-vivo radical-pair control

**Shaun C. Burd et al., "Magnetic resonance control of spin-correlated radical
pair dynamics in vivo," *Nature*, 2026.
DOI: [10.1038/s41586-026-10282-4](https://doi.org/10.1038/s41586-026-10282-4)**

- live transgenic *C. elegans*
- spin-correlated radical pair
- RF magnetic resonance manipulation
- room temperature
- observations imply coherence **> 4 ns**

*Establishes:* that spin coherence can be manipulated inside a living animal at
room temperature.

*Does not establish:* the same in Drosophila; a coherence time longer than
nanoseconds; or the ability to write arbitrary states. A radical pair is born in
a specific photochemical state -- which is why
`bioquantumfly.quantum.radical_pair` is a separate class hierarchy from the
abstract memory and is never used by the cryptographic layer.

---

## Cryptography

**Anne Broadbent and Sebastien Lord, "Uncloneable Quantum Encryption via
Oracles," TQC 2020.
DOI: [10.4230/LIPIcs.TQC.2020.4](https://doi.org/10.4230/LIPIcs.TQC.2020.4)**

**Anne Broadbent and Rabib Islam, "Quantum Encryption with Certified Deletion,"
TCC 2020.
DOI: [10.1007/978-3-030-64381-2_4](https://doi.org/10.1007/978-3-030-64381-2_4)**

*Conceptual foundation only.* The toy protocols in `bioquantumfly.crypto` are
**not** these schemes and must not be described as achieving their guarantees.
In particular:

- `MaskedConjugateEncoding` is not uncloneable encryption. It has no oracle, no
  security proof, and its pre-key-reveal privacy comes from a one-time pad.
- Nothing here implements certified deletion. Natural host decoherence is not a
  deletion certificate, and this repository never calls it one -- see
  `docs/threat_model.md`.

---

## Physical constants

| constant | value | source |
|---|---|---|
| electron gyromagnetic ratio | 1.76085963e11 rad/s/T | CODATA |

## Parameters that are deliberately NOT sourced

These carry `Status.PROXY` or `unknown` in code, because no primary-source value
was verified while writing this repository:

| parameter | status | where |
|---|---|---|
| hyperfine coupling | PROXY, generic ~1 mT | `quantum/radical_pair.py` |
| exchange coupling J | unknown -- term omitted | `quantum/radical_pair.py` |
| dipolar coupling D | unknown -- term omitted | `quantum/radical_pair.py` |
| recombination rates k_S, k_T | PROXY, generic ~1e6 /s | `quantum/radical_pair.py` |
| EYFP zero-field splitting D, E | unknown | `quantum/eyfp_proxy.py` |
| EYFP g-factor, hyperfine tensor | unknown | `quantum/eyfp_proxy.py` |
| quantum-to-neural coupling gain kappa | speculative, uncalibrated | `coupling/quantum_neural.py` |
| synthetic graph connectivity | tuned placeholder | `connectome/synthetic.py` |

## Results quoted from memory, pending verification

Stated in `docs/capabilities.md` as context and NOT verified against a primary
source while writing this repository. Check before citing:

| claim | attribution believed to be | where used |
|---|---|---|
| Optimal Wiesner-money counterfeiting probability is (3/4)^n | Molina, Vidick & Watrous, 2012 | capabilities.md section 2 |
| Bit commitment and OT are achievable in the noisy-storage model | Damgard, Fehr, Salvail & Schaffner; Wehner, Schaffner & Terhal | noisy_storage.py, capabilities.md |
| NV nuclear-spin coherence reaches ~1 s at room temperature | standard NV-centre literature | requirements.py (context only, flagged in-code) |

Supply real values from primary sources before drawing any chemical or
physiological conclusion.
