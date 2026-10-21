# Model

## Layering

```
molecular quantum subsystem  <  selected neuron  <  MaleCNS classical connectome
```

Neuron *i* is an ordinary classical dynamical system. Inside it sits a small
quantum subsystem with density matrix `rho_i(t)`. The brain is never simulated
as a quantum system: **neurons are not qubits, synapses are not entanglement**,
and no whole-brain Hilbert space is ever constructed.

## Quantum storage model A -- abstract finite-coherence qubit

The primary model for the cryptographic analysis
(`bioquantumfly.quantum.abstract_memory`).

Each encoded bit is an independent single-qubit density matrix, so an n-bit
message is a product state costing O(n), never O(2^n). At n = 128 the simulation
holds a (128, 3) array, not a 2^128 statevector.

Evolution follows the Lindblad equation

```
d(rho)/dt = -i[H, rho] + gamma_1 D[sigma_-] rho + gamma_phi D[Z] rho
```

with `T1 = 1/gamma_1` and `1/T2 = 1/(2 T1) + 1/T_phi`, so `T2 <= 2 T1` is
enforced (`channels.validate_t1_t2` raises rather than silently clamping).

### Bloch representation and why

With `rho = (I + r.sigma)/2` the channel is a closed form:

```
x, y  ->  x, y * exp(-t/T2)
z     ->  1 - (1 - z) * exp(-t/T1)
```

An entire sweep -- carriers x times x coherence times -- is then a handful of
vectorised NumPy operations. QuTiP is reserved for validation, general Lindblad
models and multi-spin systems; `tests/test_channels_vs_qutip.py` checks the two
backends agree to ~5e-7.

> **Convention note.** QuTiP's `sigmam()` uses a spin convention in which
> `basis(2,0)` is the *upper* state, so it relaxes toward Bloch -z. This project
> uses the quantum-information convention (|0> is the ground state, damping goes
> to +z), and `quantum.lindblad.lowering_to_ground` builds the operator
> explicitly rather than relying on the name.

## Quantum storage model B -- radical pair

`bioquantumfly.quantum.radical_pair`, a **separate class hierarchy**:

```
H_RP = H_Z + H_hf + H_exchange + H_dipolar
```

with Haberkorn spin-selective recombination
`d(rho)/dt = -i[H,rho] - (1/2){k_S Q_S + k_T Q_T, rho}` -- not of Lindblad
`D[C]` form, so the Liouvillian is assembled explicitly. The singlet yield is
evaluated exactly as `Y_S = k_S Tr[Q_S (-L^-1 rho_0)]`, no time grid needed.

**Why separate.** A natural radical pair is created by photochemistry in a
specific initial state. You cannot write `alpha|0> + beta|1>` into it. Treating
it as a general memory would smuggle an unjustified capability into the security
claims, so the cryptographic layer never touches this module.

## Physical regimes

| regime | T2 range | evidence class |
|---|---|---|
| `in_vivo_current` | 4 ns - 100 ns | PROXY (Burd et al.; Feder et al. room-T) |
| `cryogenic_protein` | 1 - 20 us | MEASURED for EYFP at cryogenic temperature |
| `future` | 100 us - 10 ms | SPECULATIVE, no experimental support |

Default sweep: 4 ns, 10 ns, 100 ns, 1 us, 10 us, 100 us, 1 ms.

## Protocols

### Protocol A -- raw BB84-like (`raw_bb84`)

`|psi> = H^theta |m>`. The message bit *is* the encoded bit; only the basis is
secret.

**This is not encryption**, and the code says so in three places. A Breidbart
measurement recovers the raw bit with probability `cos^2(pi/8) = 0.8536` using
no key at all. Experiment 2 demonstrates this as a negative result.

### Protocol B -- masked conjugate encoding (`masked_conjugate`)

Random pad `s`, basis key `theta`, `x = m XOR s`,
`|psi> = tensor_i H^theta_i |x_i>`, key `k = (s, theta)`.

Before key reveal, `x` is uniform and independent of `m`, so Eve's state carries
**exactly zero** information about the message whatever her attack -- verified
numerically for every implemented attack. That privacy comes from the one-time
pad, not from quantum mechanics, which is precisely why the post-key-reveal
column matters and why the SWAP control is decisive.

**Still a toy model.** Not the Broadbent-Lord scheme, not certified deletion.

### Protocol C -- the composed scheme (`crypto.stack`)

Protocols A and B are diagnostic; C is the one built to survive an adversary. It
adds four layers, each repairing a specific measured failure:

| layer | module | repairs |
|---|---|---|
| error correction | `crypto.postprocessing` | the strict-threshold penalty (~111x in storage time) |
| privacy amplification | `crypto.postprocessing` | the fraction of a bit the adversary keeps forever |
| certified deletion | `crypto.deletion` | "it decohered" certifying nothing |
| delayed key release | `crypto.noisy_storage` | an adversary who already holds the state |

Full rationale, ablation numbers and the primitives that were *rejected* are in
`docs/protocol_design.md`.

Two design requirements this produced, both concrete: certified deletion needs
**>= 10^4 check positions** at a 1% theft bound (below ~10^3 the binomial spread
on the check set exceeds the signal), and the key should be withheld for
**>= 5 x T2_Eve**.

## Adversary models

Each attack is a set of probability-weighted branches, each with a linear map on
the host Bloch vector and a map giving what Eve retains. Bob's post-attack state
is the weighted sum; Eve's guessing probability is the weighted average of her
per-branch Helstrom optimum. Everything is exact -- no Monte Carlo noise.

| attack | Eve keeps | note |
|---|---|---|
| `none` | nothing | control |
| `z_intercept_resend` | classical outcome | |
| `random_basis_intercept_resend` | classical outcome + basis | |
| `breidbart` | classical outcome | optimal single measurement for unknown-basis BB84 |
| `weak_measurement(eta)` | classical outcome | eta interpolates 0 -> projective |
| `coherent_ancilla(alpha)` | a qubit | joint 4x4 state kept explicitly |
| `swap_theft` | a perfect copy | **negative control** |

A stored classical outcome is represented as the corresponding diagonal Bloch
vector, so one Helstrom formula scores classical and quantum adversaries alike.

## Neural model

`bioquantumfly.neural`, following Shiu et al. 2024:

```
dv_i/dt = [g_i - (v_i - V_resting)] / T_mbr
dg_i/dt = -g_i / tau
```

with `g_i <- g_i + w_{j,i}` on a presynaptic spike and
`w_{j,i} = N^syn_{j,i} * s_j * W_syn`.

> Neural dynamics are a Shiu-et-al.-style LIF transfer model applied to MaleCNS
> connectivity; this does not constitute experimental validation of MaleCNS
> neural dynamics.

**A detail that matters numerically.** Because `tau_synapse` (5 ms) is four
times shorter than `T_mbr` (20 ms), an injected impulse reaches only ~15.7% of
its nominal amplitude at the membrane (`LIFParameters.psp_peak_factor`).
Threshold therefore needs roughly **162 coincident excitatory synapses**, not
the naive `gap / W_syn = 25`. Sizing a stimulus without this factor
under-drives the network about sixfold and produces a network in which
stimulated cells fire but nothing propagates.

## Quantum-to-neural coupling (speculative, default OFF)

```
y_i(t)       = Tr[M rho_i(t)]
Delta g_i(t) = kappa * [y_i(t) - y_0]
```

`kappa` is `speculative_coupling_gain` throughout the code. It has no empirical
calibration, so it is swept dimensionlessly over {0, 1e-3, 1e-2, 1e-1, 1, 10}.
Dimensional consistency comes from scaling by `drive_to_fire`, the impulse that
just brings a resting neuron to threshold, so `kappa = 1` means "a full-swing
change in the molecular observable is worth one spike's worth of drive".

### Two hybrid modes -- never conflated

| mode | coupling | role |
|---|---|---|
| `storage_only` | off | the memory sits in a neuron but does not affect it. **Primary mode for every cryptographic question.** MaleCNS supplies host selection, connectivity context and visualisation. |
| `tamper_alarm` | on | speculative; asks whether measurement disturbance could propagate into network activity. |

`storage_only` is guaranteed numerically identical to no coupling at all
(`quantum_drive_vector` returns zeros), and a test enforces it.

## Experiment map

| module name | spec | question |
|---|---|---|
| `quantum_memory_decay` | Exp 1 & 5 | how long does the memory last, and when has the host forgotten? |
| `eve_attack_sweep` | Exp 2 & 3 | raw vs masked encoding under each attack |
| `information_disturbance` | Exp 4 | Eve's information against Bob's error |
| `eve_ancilla_persistence` | Exp 6 | what survives host decoherence |
| `swap_negative_control` | Exp 7 | the worst case, stated plainly |
| `neural_baseline` | Exp 8 | does the network propagate at all? |
| `hybrid_tamper_propagation` | Exp 9 | SPECULATIVE tamper amplification |
| `host_placement` | addendum 28 | where is a perturbation best amplified? |
| `feasibility_map` | addendum 29 | the boundary between regimes |

## Controls

| control | isolates |
|---|---|
| infinite-coherence memory (`MemoryParams.ideal`) | everything that is not decoherence |
| classical-memory comparison | what quantum disturbance buys |
| `degree_preserving` graph shuffle | topology beyond degree distribution |
| `weight_shuffled` graph | edge weights vs wiring |
| `erdos_renyi` matched graph | pure size effect |
| host-location shuffle | placement vs host count |
| `storage_only` mode | the coupling itself |
