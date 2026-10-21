# BioQuantumFly

**MaleCNS-based simulation of biologically hosted ephemeral quantum cryptography.**

A research proof-of-concept. It asks one conditional question:

> Given experimentally plausible or future biological quantum coherence times,
> what cryptographic functionality could a biologically hosted finite-lifetime
> quantum state provide under explicitly limited adversarial access?

We simulate a hypothetical biologically hosted quantum memory whose physical
parameters are constrained, where possible, by current molecular-spin
experiments. MaleCNS supplies anatomical and network connectivity, not evidence
for quantum storage in *Drosophila*.

---

## What this project does NOT claim

Read this first. None of the following is asserted anywhere in this repository:

1. That arbitrary quantum states can be written into a real *Drosophila*.
2. That CRY or EYFP quantum memories actually exist in MaleCNS neurons.
3. That natural decoherence implements cryptographic **certified deletion**.
4. That quantum information influences neuron firing in any real organism.
5. Anything about consciousness, "quantum brain theory", or Penrose/Hameroff.
6. A wet-lab implementation procedure.
7. Readiness for real-world cryptographic deployment.

The fly brain is **not** modelled as a quantum computer. Specifically:

```
neuron  !=  qubit          synapse  !=  entanglement
```

The model is strictly layered, and the quantum part is small:

```
molecular quantum subsystem  <  selected neuron  <  MaleCNS classical connectome
```

---

## The central result

Two adversary regimes, reported separately, and the scheme behaves completely
differently in each:

| regime | Eve may | outcome |
|---|---|---|
| **constrained access** | measure, interact weakly, couple coherently for a limited time | the expected information-disturbance tradeoff; a usable storage window exists |
| **unrestricted extraction** | transfer the state into her own memory | **total failure**, and host decoherence does not help |

> ## no-cloning != no-theft

The SWAP negative control (Experiment 7) shows the host's information falling to
zero while the adversary's copy is untouched, permanently. So:

**Natural host decoherence is not certified deletion**, and this repository never
calls it that. It uses the term *self-erasure* for what the host does and
reserves *deletion* for the cryptographic property the model does not have.

**But the failure turned out to be repairable — by a missing layer, not by
better physics.** Adding the standard post-processing discipline (parameter
estimation, error correction, privacy amplification) means a state-theft attack
drives the error rate to 0.52, past the ~11% one-way threshold, and the round is
simply discarded. See [`docs/protocol_design.md`](docs/protocol_design.md).

Measured, at `t_attack = 0.05 T2` with a coherent-ancilla adversary:

| t / T2 | host information | Eve information |
|---|---|---|
| 0.1 | 0.42 bits | 0.09 bits |
| 1 | 0.06 bits | 0.09 bits |
| 10 | ~0 bits | 0.09 bits |
| 50 | ~0 bits | 0.09 bits |

A second result worth stating plainly: **a secret basis alone is not
encryption.** Against raw BB84-like encoding, a Breidbart measurement recovers
the message bit with probability `cos^2(pi/8) = 0.854` using **no key at all**.

---

## Quick start

```bash
source scripts/activate_env.sh          # conda env + conservative thread limits
pytest -q                               # no network, no token needed
bioquantumfly check-resources
bioquantumfly reproduce all --quick     # ~12 s: all experiments, figures, report
```

Outputs land in `results/`: per-experiment `metrics.csv`, `config.yaml`,
`metadata.json`, plus `results/figures/` and `results/initial_report.md`.

### Without a neuPrint token

Everything above runs offline. Where a real connectome would be used, a clearly
labelled **synthetic substitute** stands in; it is stamped `synthetic: True` in
provenance and marked on every figure built from it. It is *not* MaleCNS, and the
connectome-specific questions stay open until a real subgraph is fetched.

### With a neuPrint token

```bash
export NEUPRINT_TOKEN="..."             # never commit this
bioquantumfly fetch-subgraph --seed-type DNge104 --hops 2 --min-weight 5
bioquantumfly simulate-neural --config configs/male_cns_subgraph.yaml
```

The full ~1.1 GB connectivity table is never downloaded implicitly; the default
path fetches a k-hop neighbourhood and caches it. `DNge104` is used only because
the MaleCNS documentation uses it as an API example — no quantum-biological
significance is claimed for it.

---

## Experiments

| command | spec | question |
|---|---|---|
| `simulate-quantum` | Exp 1 & 5 | how long does the memory last, and when has the host forgotten? |
| `simulate-eve` | Exp 2 & 3 | raw vs masked encoding under each attack |
| `simulate-eve --experiment information_disturbance` | Exp 4 | Eve's information against Bob's error |
| `simulate-eve --experiment eve_ancilla_persistence` | Exp 6 | what survives host decoherence |
| `simulate-eve --experiment swap_negative_control` | Exp 7 | the worst case, stated plainly |
| `simulate-neural` | Exp 8 | does the network propagate at all? |
| `simulate-hybrid` | Exp 9 | **SPECULATIVE** tamper amplification |
| `host-placement` | addendum 28 | where would a perturbation be most amplified? |
| `sweep --experiment feasibility_map` | addendum 29 | the feasibility boundary |
| `sweep --experiment protocol_stack` | Exp 10 | which primitive repairs which failure |
| `sweep --experiment coherence_requirements` | Exp 11 | what coherence time would each error budget demand? |

## Figures

`results/figures/`, regenerated by `bioquantumfly figures`. Each PNG is written
with its source data as a sibling `.csv`.

| figure | shows |
|---|---|
| 1 | system architecture |
| 2 | Bob's readout, normalised and in absolute seconds |
| 3 | information-disturbance tradeoff |
| 4 | **host self-decay vs Eve's retained information** |
| 5 | usable storage window by physical regime |
| 6 | connectome subgraph with hosts highlighted |
| 7 | downstream neural response with and without Eve (speculative) |
| 8 | cryptographic feasibility map |
| 9 | raw vs masked encoding: a secret basis is not encryption |
| 10 | what each primitive buys (layer ablation) |
| 11 | required coherence time against the tolerated error rate |

Evidence class is encoded by **line style** as well as in the legend — solid for
measured, dashed for proxy, dotted for speculative — so a speculative curve can
never be mistaken for a measured one, and the distinction survives printing and
colour-vision differences.

---

## Physical regimes

| regime | T2 | evidence class |
|---|---|---|
| `in_vivo_current` | 4 ns – 100 ns | **proxy** (Burd et al. in *C. elegans*; Feder et al. room-T) |
| `cryogenic_protein` | 1 – 20 us | **measured** for EYFP at cryogenic temperature |
| `future` | 100 us – 10 ms | **speculative**, no experimental support |

No coherence time has ever been measured for a quantum memory in *Drosophila*,
because no such memory exists. See [`docs/sources.md`](docs/sources.md) for what
each citation does and does not establish.

### Is the gap fundamental?

No. Two corrections shrink it substantially, and both are quantified in
[`docs/protocol_design.md`](docs/protocol_design.md):

* **Error correction.** The usable window is linear in the tolerated error rate,
  so demanding BER ≤ 1e-3 outright instead of correcting a 10% rate costs a
  factor of **111** in storage time. Quoting only the strict number overstates
  the physical shortfall by two orders of magnitude.
* **Architecture.** An electron spin in a warm protein is the worst place to
  store: the coupling that makes it optically addressable is the coupling that
  decoheres it. Splitting the roles — electron spin as interface, nuclear spin
  as memory — is the standard solid-state answer, and a nuclear-spin memory
  would already exceed every target in the table. Whether that can be built
  inside a protein in a living cell is open, and nothing here answers it.

---

## Documentation

| file | contents |
|---|---|
| [`docs/model.md`](docs/model.md) | the model, the protocols, the adversaries, the experiment map |
| [`docs/threat_model.md`](docs/threat_model.md) | the two access regimes, and why decoherence is not deletion |
| [`docs/assumptions.md`](docs/assumptions.md) | every assumption with its evidence class, worst first |
| [`docs/limitations.md`](docs/limitations.md) | what this cannot tell you |
| [`docs/capabilities.md`](docs/capabilities.md) | **what this can and cannot do**, sorted by what blocks it |
| [`docs/protocol_design.md`](docs/protocol_design.md) | which primitives were chosen, what each buys, what was rejected |
| [`docs/sources.md`](docs/sources.md) | full citations, and what each does not establish |
| [`docs/environment.md`](docs/environment.md) | environment, filesystem policy, thread policy |

---

## Design notes

**Speed.** Single-qubit channels have closed forms on the Bloch vector, so
production sweeps are vectorised NumPy rather than per-carrier `mesolve` calls;
`tests/test_channels_vs_qutip.py` holds the two backends to ~5e-7. A 3,780-row
feasibility map takes under a second. QuTiP is used for validation, general
Lindblad models and the multi-spin radical pair.

**Scale.** Product-state encoding means an n-bit message costs O(n): at n = 128
the state is a (128, 3) array, never a 2^128 statevector. The connectome uses
`scipy.sparse` CSR, never a dense 166,691² matrix.

**Reproducibility.** Every run derives its seed from its own parameters, so
`--workers 1` and `--workers 64` produce bit-identical output — enforced by
`tests/test_determinism.py`. Sweeps are resumable through an append-only
manifest and survive interruption.

**Shared-server hygiene.** The default profile uses 4 workers, never 384. BLAS
threads are pinned to 1 per worker so process parallelism cannot multiply into
thousands of threads. Nothing is written to the root filesystem.

---

## Not installed, deliberately

`qiskit`, `torch`, `tensorflow`, `jax`, `cupy`, CUDA. The model needs density
matrices, Lindblad dynamics, radical-pair physics, LIF simulation and graph
processing; QuTiP and Brian2 cover all of it. CPU-first.

---

## Status

Environment and Stages 1–9 implemented and tested. The scientific content is a
feasibility boundary, not a capability claim. Positive and negative results are
treated equally: the regions where coherence is too short, where extraction is
too strong, where coupling is too weak, and where a classical alternative would
do just as well are all part of the output.

## Licence

MIT.
