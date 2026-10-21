"""Analytic reference, accounting, information access and independent solver checks."""
import numpy as np
import pytest
from bioquantumfly.operational.resource_benchmark import (
    BITS, CostLedger, gate_words, metrics, reference_fixtures, relaxed_pom_optimum,
    transformed_bits,
)
from bioquantumfly.operational.resource_lif import sample


def test_reference_bounds_and_opened_interface():
    ref = reference_fixtures()
    assert ref["one_classical_bit"]["success_interval"] == [.75, .75]
    assert ref["qubit_reference"]["success_interval"][0] == pytest.approx((1+1/np.sqrt(2))/2)
    assert ref["two_bit_shadow"]["conditional_witness_noise_interval"] == [.5, .5]
    assert ref["weak_shadow"]["conditional_witness_noise_interval"] == [0., 0.]
    assert ref["opened_classical_interface"]["parity_gap"] == 1.
    assert not ref["opened_classical_interface"]["old_parity_equivalence_valid"]
    assert len(ref["shadow_gate_success"]) == 31
    assert set(ref["shadow_gate_success"].values()) == {1.}


@pytest.mark.parametrize("theta", [0., .02, .1, .2, .5, 1.])
def test_relaxed_ontic_bound(theta):
    assert relaxed_pom_optimum(theta) == pytest.approx(.75+theta/4, abs=1e-10)


def test_uncertain_parity_is_not_assumed_equal():
    result = metrics(np.zeros((4, 2)), np.ones((4, 2)))
    assert result["success_interval"] == [0., 1.]
    assert result["parity_gap_interval"] == [0., 1.]
    assert not result["finite_interface_candidate_pass"]
    assert not result["exact_noncontextuality_certified"]
    # Perfect selected readouts have exact observed parity equality, while
    # any classical reproduction requires full hidden parity TV = 1.
    sharp = metrics(BITS, BITS)
    assert sharp["parity_gap_interval"] == [0., 0.]
    assert sharp["required_ontic_parity_tv_lower"] == 1.


def test_gate_order_against_explicit_truth_tables():
    np.testing.assert_array_equal(transformed_bits("FW"), [[0, 1], [1, 1], [0, 0], [1, 0]])
    np.testing.assert_array_equal(transformed_bits("WF"), [[1, 0], [0, 0], [1, 1], [0, 1]])
    np.testing.assert_array_equal(transformed_bits("FFWW"), BITS)


def test_expanded_interface_rank_witness_and_interior_containment():
    from bioquantumfly.operational.resource_experiment_report import unit_minor_interval
    d = np.c_[BITS, BITS[:, 0] ^ BITS[:, 1]].astype(float)
    lower, upper = np.maximum(0, d-.01), np.minimum(1, d+.01)
    interval = unit_minor_interval(lower, upper)
    assert max(interval[0], -interval[1]) > 1.
    rng = np.random.default_rng(91)
    for _ in range(20):
        interior = lower+rng.random((4, 3))*(upper-lower)
        det = np.linalg.det(np.c_[np.ones(4), interior])
        assert interval[0]-1e-12 <= det <= interval[1]+1e-12
    assert unit_minor_interval(np.c_[BITS, np.zeros(4)], np.c_[BITS, np.zeros(4)]) == [0., 0.]


def test_ledger_rejects_overrun_before_sampling_and_counts_copies():
    ledger = CostLedger(8)
    ledger.reserve(8)
    with pytest.raises(RuntimeError):
        ledger.reserve(1)
    assert ledger.physical_trials == ledger.reset_operations == 8
    ledger.record(trials=8, neurons=2, duration_ms=19, spikes=4, noise_events=3,
                  absolute_drive_neuron_ms=100, readout_bits_per_trial=2,
                  simulator_seconds=.1)
    assert ledger.readout_bits == 16
    assert ledger.neuron_time_ms == 304


def test_query_has_no_effect_on_state_evolution_and_joint_audit_agrees():
    args = (8, [1.6, .3], np.zeros((2, 2, 2)), 2., "")
    results = [sample(*args, q, 128, seed=92001) for q in (0, 1, 2)]
    np.testing.assert_array_equal(results[0][0], results[2][0]//2)
    np.testing.assert_array_equal(results[1][0], results[2][0]%2)
    for _, c in results[1:]:
        np.testing.assert_array_equal(c["activity"], results[0][1]["activity"])
        np.testing.assert_array_equal(c["drive"], results[0][1]["drive"])


def brian_reference(n, prep, gates, gain, word, dt):
    """Independent Brian2 construction, no reuse of the compiled integrator."""
    import brian2 as b2
    b2.start_scope()
    b2.prefs.codegen.target = "numpy"
    clock = b2.Clock(dt=dt*b2.ms)
    duration = 19+10*len(word)
    times = np.arange(round(duration/dt))*dt
    currents = np.zeros((len(times), n))
    half = n//2
    currents[times < 12, :half] = prep[0]
    currents[times < 12, half:] = prep[1]
    for k, letter in enumerate(word):
        for phase in (0, 1):
            mask = (times >= 14+10*k+5*phase) & (times < 19+10*k+5*phase)
            amp = gates["FW".index(letter), phase]
            currents[mask, :half] = amp[0]; currents[mask, half:] = amp[1]
    start_read = 14+10*len(word)
    currents[times >= start_read] = 2.5
    drive = b2.TimedArray(currents, dt=dt*b2.ms)
    group = b2.NeuronGroup(n,
        "dv/dt=(-v+g+drive(t,i))/(20*ms) : 1 (unless refractory)\n"
        "dg/dt=-g/(5*ms) : 1",
        threshold="v >= 1", reset="v=0", refractory=2.2*b2.ms,
        method="exact", clock=clock, namespace={"drive": drive})
    syn = b2.Synapses(group, group, "w : 1", on_pre="g_post += w",
                      delay=1.8*b2.ms, clock=clock)
    ii, jj, ww = [], [], []
    for i in range(n):
        if half > 1:
            for j in range((i//half)*half, (i//half+1)*half):
                if i != j: ii.append(i); jj.append(j); ww.append(gain/(half-1))
        ii.append(i); jj.append((i+half)%n); ww.append(.05*gain)
    syn.connect(i=ii, j=jj); syn.w = ww
    mon = b2.SpikeMonitor(group)
    net = b2.Network(group, syn, mon)
    net.run(duration*b2.ms)
    fired = np.zeros(n, bool)
    fired[np.asarray(mon.i)[np.asarray(mon.t/b2.ms) >= start_read-1e-8]] = True
    return int(2*(fired[:half].sum() >= (half+1)//2)
               +(fired[half:].sum() >= (half+1)//2)), int(mon.num_spikes)


@pytest.mark.parametrize("n,prep,gain,word", [
    (2, [0., 0.], 0., ""), (2, [1.9, 0.], 2., ""),
    (8, [2.6, .1], 4., "F"), (16, [1.7, 2.1], 3., "FW"),
])
def test_noiseless_integrator_against_brian2(n, prep, gain, word):
    gates = np.array([[[-1., 2.], [2., -1.]], [[.5, 1.], [1., .5]]])
    out, cost = sample(n, prep, gates, gain, word, 2, 1, seed=92,
                       dt_ms=.05, jitter=0., background_hz=0.)
    expected, spikes = brian_reference(n, prep, gates, gain, word, .05)
    assert int(out[0]) == expected
    assert cost["spikes"] == spikes


@pytest.mark.parametrize("kwargs", [
    {"neurons": 17}, {"shots": 0}, {"query": 3}, {"word": "XXXXX"},
    {"dt_ms": .0001}, {"gain": 9.}, {"preparation": [float("nan"), 1.]},
])
def test_integrator_bounds(kwargs):
    args = dict(neurons=2, preparation=[1., 1.], gates=np.zeros((2, 2, 2)),
                gain=2., word="", query=0, shots=1, seed=1)
    args.update(kwargs)
    with pytest.raises(ValueError):
        sample(**args)
