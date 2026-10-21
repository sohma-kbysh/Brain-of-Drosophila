// A bounded classical LIF integrator. Units: threshold gap (7 mV), milliseconds.
// Exact subthreshold flow with piecewise constant current; event times on dt grid.
#include <cmath>
#include <cstdint>
#include <algorithm>

static uint64_t next64(uint64_t& state) {
    state += UINT64_C(0x9e3779b97f4a7c15);
    uint64_t z = state;
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}
static double uniform(uint64_t& state) {
    return (next64(state) >> 11) * 0x1.0p-53;
}

extern "C" void resource_lif(
    int n, int shots, const double* prep, const double* gates, double gain,
    const int* word, int length, int query, double dt, uint64_t seed,
    double jitter, double background_hz, uint8_t* outcomes,
    uint32_t* activity, uint32_t* noise_events, double* drive_cost) {
    const int half = n/2;
    const int prep_steps = int(std::llround(12./dt));
    const int hold_steps = int(std::llround(2./dt));
    const int phase_steps = int(std::llround(5./dt));
    const int gate_steps = 2*phase_steps;
    const int read_steps = phase_steps;
    const int read_start = prep_steps + hold_steps + length*gate_steps;
    const int total_steps = read_start + read_steps;
    const int delay = int(std::llround(1.8/dt));
    const int refractory = int(std::ceil(2.2/dt));
    const double am = std::exp(-dt/20.), as = std::exp(-dt/5.);
    const double coupling = (5./(5.-20.))*(as-am);
    for (int trial=0; trial<shots; ++trial) {
        uint64_t rng = seed ^ (UINT64_C(0xd1342543de82ef95)*(trial+1));
        double v[16] = {}, g[16] = {}, scale[16] = {};
        uint8_t queue[80][16] = {};
        int ref_until[16] = {};
        int read_counts[2] = {};
        bool read_fired[16] = {};
        uint32_t spikes=0, noise=0;
        double cost=0.;
        for (int i=0; i<n; ++i) scale[i] = 1. + jitter*(2*uniform(rng)-1);
        for (int step=0; step<total_steps; ++step) {
            const int slot = step % delay;
            uint8_t fired[16] = {};
            for (int i=0; i<n; ++i) {
                const int pool = i < half ? 0 : 1;
                double current = 0.;
                if (step < prep_steps) current = prep[pool];
                else if (step >= prep_steps+hold_steps && step < read_start) {
                    const int rel = step-prep_steps-hold_steps;
                    int gate = word[rel/gate_steps];
                    if (gate >= 0)
                        current = gates[gate*4+(rel%gate_steps)/phase_steps*2+pool];
                } else if (step >= read_start) current = 2.5;
                current *= scale[i];
                cost += std::abs(current)*dt;
                if (step >= ref_until[i])
                    v[i] = am*v[i] + (1-am)*current + coupling*g[i];
                g[i] *= as;
                if (uniform(rng) < background_hz*dt/1000.) {g[i] += .2; ++noise;}
                if (step >= ref_until[i] && v[i] >= 1.) {
                    v[i]=0.; ref_until[i]=step+refractory;
                    fired[i]=1; ++spikes;
                    if (step >= read_start && !read_fired[i]) {
                        read_fired[i]=true; ++read_counts[pool];
                    }
                }
            }
            // As in the usual groups/thresholds/synapses schedule: events
            // delivered now affect the following subthreshold update.
            for (int source=0; source<n; ++source) {
                if (queue[slot][source]) {
                    const int base = source < half ? 0 : half;
                    if (half>1) for (int target=base; target<base+half; ++target)
                        if (target != source) g[target] += gain/(half-1);
                    g[(source+half)%n] += .05*gain;
                }
                queue[slot][source] = fired[source];
            }
        }
        // query is consulted only after preparation and gates. Common probe
        // stimulus is independent of query; the specified monitor selects a pool.
        int b0 = read_counts[0] >= (half+1)/2;
        int b1 = read_counts[1] >= (half+1)/2;
        outcomes[trial] = query == 2 ? uint8_t(2*b0+b1) : uint8_t(query ? b1 : b0);
        activity[trial]=spikes; noise_events[trial]=noise; drive_cost[trial]=cost;
    }
}
