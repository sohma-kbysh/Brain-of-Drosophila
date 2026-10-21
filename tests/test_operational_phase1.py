"""Mathematical and cross-boundary regression tests for synthetic Phase 1."""

from itertools import product
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binom

from bioquantumfly.operational.embedding import embedding_lp, replacement_robustness, pom_success_bounds
from bioquantumfly.operational.statistics import (
    simultaneous_binomial, affine_rank_witness, factor_probabilities, select_dimension,
)
from bioquantumfly.operational.toys import (
    compression_toy, shadow_toy, relative_tomography, preparation_relations, rational_matrix,
)


def test_exact_binomial_endpoints_and_finite_coverage():
    lo, hi = simultaneous_binomial(np.array([0, 20]), 20, family_size=2)
    assert hi[0] == pytest.approx(1-(.05/4)**(1/20))
    assert lo[1] == pytest.approx((.05/4)**(1/20))
    intervals = [simultaneous_binomial(np.array([k]),20) for k in range(21)]
    for p in (.001,.1,.5,.9,.999):
        coverage = sum(binom.pmf(k,20,p) for k,(l,h) in enumerate(intervals) if l[0] <= p <= h[0])
        assert coverage >= .95-1e-12


@pytest.mark.parametrize("counts,shots", [([-1],20),([21],20),([1.5],20),([0],0),([np.nan],20)])
def test_invalid_counts_rejected(counts,shots):
    with pytest.raises(ValueError): simultaneous_binomial(counts,shots)


def test_affine_rank_witness_and_uncertainty():
    d=np.array([[0,0],[1,0],[0,1],[.5,.5]])
    assert affine_rank_witness(d,d)["certified_rank_at_least_3"]
    assert not affine_rank_witness(np.zeros_like(d),np.ones_like(d))["certified_rank_at_least_3"]
    line=np.array([[0,0],[.5,.25],[1,.5]])
    assert not affine_rank_witness(line,line)["certified_rank_at_least_3"]


def test_sufficient_compression_and_relative_not_global_tomography():
    toy=compression_toy()
    audit=relative_tomography(toy["states"],np.c_[np.ones(6),toy["effects"]])
    assert audit["relative_tomography"]
    assert audit["pairing_rank"]==3 < audit["parent_microstate_count"]==6
    np.testing.assert_allclose(toy["lift"]@toy["collect"],np.eye(3),atol=1e-15)
    np.testing.assert_allclose(toy["lift"]@toy["kernel"],toy["coarse_kernel"]@toy["lift"],atol=1e-15)
    for steps in (0,1,2,5):
        full=toy["states"]@np.linalg.matrix_power(toy["kernel"],steps)@toy["effects"]
        compressed=toy["coarse_states"]@np.linalg.matrix_power(toy["coarse_kernel"],steps)@toy["coarse_effects"]
        np.testing.assert_allclose(full,compressed,atol=1e-15)


def test_probability_factorization_and_gauge():
    d=compression_toy()["probabilities"]
    selected,_=select_dimension(d,d)
    # Exact zero residual is subject to roundoff, so test minimal adequate d directly.
    s,e=factor_probabilities(d,3)
    np.testing.assert_allclose(s@e,d,atol=1e-14)
    g=np.array([[1,.7,-.1],[0,2,1],[0,0,.5]])
    np.testing.assert_allclose((s@g)@np.linalg.solve(g,e),d,atol=1e-14)
    assert selected >= 3


def test_noise_aware_rank_selection_uses_validation_uncertainty():
    d=compression_toy()["probabilities"]
    # Known noiseless probabilities are used only to construct this oracle fixture.
    selected,scores=select_dimension(d,d,validation_shots=20_000)
    assert selected==3
    assert any(row.get("paired_hoeffding_radius",0)>0 for row in scores)
    with pytest.raises(ValueError): select_dimension(d,d,validation_shots=0)


@pytest.mark.parametrize("dimension", [2,3,4])
def test_dimension_selector_is_not_fixed_to_three(dimension):
    rng=np.random.default_rng(480+dimension)
    states=np.vstack([np.eye(dimension),rng.dirichlet(np.ones(dimension),size=dimension)])
    effects=np.c_[.15+.7*np.eye(dimension),.2+.6*rng.random((dimension,2))]
    d=states@effects
    train,validation=[rng.binomial(20_000,d)/20_000 for _ in range(2)]
    selected,_=select_dimension(train,validation,validation_shots=20_000)
    assert selected==dimension


def scenario(case,visibility):
    toy=shadow_toy(visibility); d=toy[f"{case}_probabilities"]; e=toy[f"{case}_effects"]
    exact=rational_matrix(toy["states"])@rational_matrix(np.c_[np.ones(4),e])
    relations=preparation_relations(toy["states"] if case=="adequate" else np.array(exact.tolist(),dtype=object))
    return toy,d,e,relations


@pytest.mark.parametrize("case,visibility", list(product(("adequate","hidden"),(1.,.4))))
def test_shadow_trap_and_weak_square_control(case,visibility):
    toy,d,e,relations=scenario(case,visibility)
    audit=relative_tomography(toy["states"],np.c_[np.ones(4),e])
    assert audit["relative_tomography"]==(case=="adequate")
    result=embedding_lp(d,d,relations)
    negative=case=="hidden" and visibility==1
    assert result["status"]==("infeasible_numeric" if negative else "feasible")
    robustness=replacement_robustness(d,relations)
    assert robustness["replacement_probability"]==pytest.approx(.5 if negative else 0,abs=1e-9)
    if result["status"]=="feasible":
        mu=np.array(result["ontic_preparations"]); response=np.array(result["ontic_responses"])
        np.testing.assert_allclose(mu@response,d,atol=1e-8)
        np.testing.assert_allclose(relations@mu,0,atol=1e-8)


def test_pom_certificate_and_wrong_equivalence_are_distinguished():
    toy,d,e,relations=scenario("hidden",1.)
    counts=(200*d).astype(int)
    lo,hi=simultaneous_binomial(counts,200,family_size=210)
    assert pom_success_bounds(lo,hi)[0] > .75
    assert embedding_lp(lo,hi,relations)["status"]=="infeasible_numeric"
    full=toy["adequate_probabilities"]
    assert embedding_lp(full,full,relations)["status"]=="scenario_inconsistent"


def test_effect_witness_is_not_used_without_parity_equivalence():
    toy,d,_,_=scenario("hidden",1.)
    # Inherited parent identities omit even=odd; a classical simulation exists.
    inherited=preparation_relations(toy["states"])
    assert embedding_lp(d,d,inherited)["status"]=="feasible"


def test_lif_reset_silence_reproducibility_and_history_survives():
    from bioquantumfly.operational.lif import simulate_protocol
    assert not simulate_protocol("00","0",32,seed=12).any()
    first=simulate_protocol("11","0",256,seed=13)
    repeat=simulate_protocol("11","0",256,seed=13)
    np.testing.assert_array_equal(first,repeat)
    # With no stimulus in either future or readout, nonzero output must be
    # carried from preparation (including recurrent drive and delayed spikes).
    assert first.any()


def test_registered_config_is_bounded_and_honest():
    from bioquantumfly.operational.phase1 import validate_config
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/"configs/operational_phase1.json").read_text())
    validate_config(config)
    with pytest.raises(ValueError): validate_config({**config,"lif_rate_hz":200})
    with pytest.raises(ValueError): validate_config({**config,"shot_levels":[200,2000000]})
