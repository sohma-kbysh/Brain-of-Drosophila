"""Two-neuron recurrent LIF protocol, independent trials batched in Brian2.

Each pair has its own Poisson input, voltages, synaptic variables, refractory
states and delay events. There are no connections between trials. Preparation,
continuation and readout use one uninterrupted Network and Clock per cell.
"""

import numpy as np

from bioquantumfly.neural.lif import LIFParameters


def simulate_protocol(preparation, future, shots, *, seed, dt_ms=0.1):
    if preparation not in ("00", "01", "10", "11") or future not in ("", "0", "1"):
        raise ValueError("expected a two-bit preparation and at most one future bit")
    if not 1 <= shots <= 20_000 or dt_ms not in (0.1, 0.05):
        raise ValueError("Phase 1 limits: at most 20000 independent trials, dt=0.1 or 0.05 ms")
    return simulate_sequence(preparation, future, shots, seed=seed, dt_ms=dt_ms)


def simulate_sequence(preparation, future, shots, *, seed, dt_ms=0.05):
    """Bounded extension for fixed-shot confirmation and open-loop prediction.

    The preparation/future boundary inserts no readout, pause or reset. Only
    the final 2 ms window is measured. Same dynamics as simulate_protocol.
    """
    import brian2 as b2

    if (not isinstance(preparation, str) or not isinstance(future, str)
            or not preparation or len(preparation)+len(future) > 9
            or set(preparation+future)-set("01")):
        raise ValueError("expected binary input strings with total length 1..9")
    if type(shots) is not int or not 1 <= shots <= 20_000 or dt_ms not in (0.1, 0.05, 0.025):
        raise ValueError("at most 20000 independent trials; dt=0.1, 0.05 or 0.025 ms")
    b2.set_device("runtime")
    b2.prefs.codegen.target = "numpy"
    b2.start_scope()
    b2.seed(int(seed))
    params = LIFParameters()
    clock = b2.Clock(dt=dt_ms*b2.ms)
    rate = b2.TimedArray([100.0*int(x) for x in preparation+future]+[0.0], dt=10*b2.ms)
    ns = {"V_rest": params.V_resting*b2.volt, "tau_m": params.T_mbr*b2.second,
          "tau_s": params.tau_synapse*b2.second}
    neurons = b2.NeuronGroup(
        2*shots, "dv/dt = (g-(v-V_rest))/tau_m : volt (unless refractory)\n"
        "dg/dt = -g/tau_s : volt", threshold=f"v > {params.V_threshold}*volt",
        reset=f"v = {params.V_reset}*volt", refractory=params.T_refractory*b2.second,
        method="exact", namespace=ns, clock=clock)
    neurons.v = params.V_resting*b2.volt
    neurons.g = 0*b2.volt
    recurrent = b2.Synapses(neurons, neurons, model="w : volt", on_pre="g_post += w",
                           delay=params.T_delay*b2.second, clock=clock)
    indices = np.arange(2*shots, dtype=np.int32)
    recurrent.connect(i=indices, j=indices ^ 1)
    recurrent.w = 200*params.W_syn*b2.volt
    inputs = b2.PoissonGroup(shots, rates="rate(t)*Hz", namespace={"rate": rate}, clock=clock)
    stimulus = b2.Synapses(inputs, neurons, model="w : volt", on_pre="g_post += w", clock=clock)
    stimulus.connect(i=np.arange(shots), j=2*np.arange(shots))
    stimulus.w = 1.2*params.drive_to_fire*b2.volt
    monitor = b2.SpikeMonitor(neurons, record=False)
    network = b2.Network(neurons, recurrent, inputs, stimulus, monitor)
    before_readout_ms = 10*(len(preparation)+len(future))
    network.run(before_readout_ms*b2.ms)
    before = np.asarray(monitor.count[1::2]).copy()
    # No reset: voltages, g, refractory state and delay queues survive this boundary.
    network.run(2*b2.ms)
    outcome = np.asarray(monitor.count[1::2]) > before
    return outcome.copy()
