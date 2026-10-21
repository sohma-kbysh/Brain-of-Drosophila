import numpy as np
import pytest

from bioquantumfly.operational.resource_profile import allocation, anchor_frequencies, arithmetic_profile, order_profile


@pytest.mark.parametrize("budget",[200,2000,10000,70000])
def test_total_access_budget_is_conserved(budget):
    sizes = allocation(budget,7)
    assert sum(sizes)==budget and max(sizes)-min(sizes)<=1
    outcomes = np.zeros((2,7,10000),dtype=bool)
    outcomes[0,:,:50]=True
    means, actual=anchor_frequencies(outcomes,budget)
    assert sizes==actual
    np.testing.assert_allclose(means[0],[min(n,50)/n for n in sizes])
    with pytest.raises(ValueError): anchor_frequencies(outcomes,70001)


def test_compact_affine_counts_are_distinct_from_dense_payload():
    model={"dimension":4,"core":[""]*7,"mean":np.zeros(7),"basis":np.zeros((7,4)),
           "effect":np.zeros(5),"updates":{"0":np.zeros((5,5)),"1":np.zeros((5,5))}}
    p=arithmetic_profile(model)
    assert p['state_float64_bytes_compact']==32
    assert p['model_float64_bytes_dense_arrays']==720
    assert p['model_float64_bytes_compact_with_encoder']==640
    assert p['one_future_arithmetic_ops_compact']==136


def test_order_intervals_detect_effect_and_preserve_uncertainty():
    counts=np.array([[0,0],[10000,0],[5000,5000]])
    r=order_profile(counts,10000,['silent','ordered','equal'],['01','10'])
    assert r['maximum_absolute_gap_interval'][0]>.99
    assert r['maximum_absolute_gap_interval'][1]==pytest.approx(1)
    assert r['witness_history']=='ordered'
    assert r['histories_with_zero_excluded']==1
