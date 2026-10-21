"""Independent exact-system checks for predictive updates and evaluation."""

from itertools import product
import json
from pathlib import Path

import numpy as np
import pytest

from bioquantumfly.operational.predictive import (
    words, words_through, fit_predictive, predict, encode, choose_candidate,
    probability_error_bounds, brier_improvement_bounds, probability_diagnostics,
    fit_suffix_baseline, predict_suffix,
)
from bioquantumfly.operational.phase1_next import prediction_split, validate_config, sample_table


def controlled_system():
    actions = {"0": np.array([[.85,.1,.05],[.15,.75,.1],[.2,.2,.6]]),
               "1": np.array([[.1,.2,.7],[.6,.3,.1],[.2,.7,.1]])}
    effect = np.array([.05,.5,.95])

    def probability(history, future):
        state = np.array([1.,0.,0.])
        for u in history+future:
            state = state@actions[u]
        return state@effect

    return probability


def exact_fixture():
    probability = controlled_system()
    histories = words(1)+words(2)+words(3)
    futures = words_through(3)
    table = np.array([[probability(h,f) for f in futures] for h in histories])
    return probability, futures, table


def test_exact_controlled_classical_system_generalizes_to_untrained_sequences():
    probability, futures, table = exact_fixture()
    core = words_through(2)
    model = fit_predictive(table, futures, core, 2, 0)
    histories, targets = words(5), words(4)
    anchors = np.array([[probability(h,q) for q in core] for h in histories])
    expected = np.array([[probability(h,f) for f in targets] for h in histories])
    np.testing.assert_allclose(predict(model, anchors, targets), expected, atol=2e-12)
    assert probability("01", "0011") != pytest.approx(probability("01", "1100"))
    unit = np.array([1.,0.,0.])
    for action in model["updates"].values():
        np.testing.assert_array_equal(action[:,0], unit)
    small = fit_predictive(table, futures, core, 1, 0)
    assert np.max(abs(predict(small, anchors, targets)-expected)) > .001


def test_affine_mixtures_and_gauge_do_not_change_future_probabilities():
    probability, futures, table = exact_fixture()
    model = fit_predictive(table, futures, words_through(2), 2, 0)
    anchors = np.array([[probability(h,q) for q in model["core"]] for h in ["01011", "10110"]])
    target = ["0101", "1111"]
    mixed = predict(model, anchors.mean(axis=0, keepdims=True), target)
    np.testing.assert_allclose(mixed[0], predict(model, anchors, target).mean(axis=0), atol=1e-12)
    g = np.array([[1,.3,-.2],[0,2,.5],[0,0,.7]])
    z = encode(model, anchors)@g
    e = np.linalg.solve(g, model["effect"])
    for u in target[0]:
        z = z@np.linalg.solve(g, model["updates"][u]@g)
    np.testing.assert_allclose(z@e, predict(model, anchors, target)[:,0], atol=1e-12)


def test_core_requires_action_calibration_and_rejects_invalid_probability_table():
    _, futures, table = exact_fixture()
    with pytest.raises(ValueError): fit_predictive(table[:,:3], futures[:3], words_through(2), 2, 0)
    invalid = table.copy(); invalid[0,0] = -1
    with pytest.raises(ValueError): fit_predictive(invalid, futures, words_through(2), 2, 0)


def test_interval_metrics_bound_exact_brier_and_probability_errors():
    p = np.array([[.2,.7],[.4,.9]])
    baseline = np.array([[.5,.3],[.6,.4]])
    lo, hi = np.array([[.1,.5],[.3,.8]]), np.array([[.3,.6],[.5,1.]])
    error = probability_error_bounds(p,lo,hi)
    improvement = brier_improvement_bounds(baseline,p,lo,hi)
    corners = []
    for bits in product((False,True), repeat=4):
        truth = np.where(np.array(bits).reshape(2,2),hi,lo)
        rmse = np.sqrt(np.mean((p-truth)**2))
        assert error["rmse_lower"] <= rmse+1e-15 <= error["rmse_upper"]+1e-15
        diff = np.mean(baseline**2-p**2-2*(baseline-p)*truth)
        corners.append(diff)
    assert improvement == pytest.approx([min(corners),max(corners)])
    assert not probability_diagnostics(np.array([[1.01]]))["valid"]
    assert not probability_diagnostics(np.array([[np.nan]]))["valid"]


def test_validation_selection_rejects_invalid_candidates_and_prefers_smallest_adequate():
    models = [{"dimension": k, "ridge": .001} for k in [1,2,3]]
    predictions = [np.full((4,3),.2), np.full((4,3),.7), np.full((4,3),1.1)]
    chosen, scores = choose_candidate(models,predictions,np.full((4,3),14000),20000)
    assert chosen == 1
    assert not scores[2]["probabilities"]["valid"]
    models.append({"dimension": 4, "ridge": .001})
    predictions.append(predictions[1].copy())
    assert choose_candidate(models,predictions,np.full((4,3),14000),20000)[0] == 1


def test_input_suffix_baseline_is_distinct_from_future_anchor_inference():
    model = fit_suffix_baseline(words(2),np.array([.1,.2,.3,.4]))
    np.testing.assert_allclose(predict_suffix(model,["0101","1010"],["0000","1111"]),[[.1,.4],[.1,.4]])


def config():
    return json.loads((Path(__file__).resolve().parents[1]/"configs/operational_phase1_next.json").read_text())


def test_fixed_history_split_and_bounded_configuration():
    c = config(); validate_config(c)
    train, validation, test = prediction_split(c)
    assert (len(train),len(validation),len(test)) == (28,16,16)
    assert not (set(train)&set(validation) or set(train)&set(test) or set(validation)&set(test))
    assert set(validation+test) == set(words(5))
    c["confirmation"]["probability_tolerance"] = .04
    with pytest.raises(ValueError): validate_config(c)


def test_batch_aggregation_independent_seeds_and_saved_raw_counts(tmp_path, monkeypatch):
    calls = []
    def simulator(history,future,shots,*,seed,dt_ms):
        calls.append((shots,seed))
        return np.arange(shots)%2 == 0
    monkeypatch.setattr("bioquantumfly.operational.phase1_next.simulate_sequence",simulator)
    counts = sample_table(config(),tmp_path,"fixture",["01"],["0"],50001,.05,77)
    assert [n for n,_ in calls] == [20000,20000,10001]
    assert len({s for _,s in calls}) == 3
    assert counts[0,0] == 25001
    raw = np.load(tmp_path/"fixture_outcomes.npz")
    assert raw["outcomes"].sum() == 25001


def test_lif_sequence_preserves_continuation_and_old_interface():
    from bioquantumfly.operational.lif import simulate_protocol, simulate_sequence
    old = simulate_protocol("11","0",256,seed=951,dt_ms=.05)
    np.testing.assert_array_equal(old,simulate_sequence("1","10",256,seed=951,dt_ms=.05))
    long = simulate_sequence("11010","1010",128,seed=952,dt_ms=.025)
    assert long.shape == (128,)
    assert not simulate_sequence("00000","0000",64,seed=953,dt_ms=.025).any()
    with pytest.raises(ValueError): simulate_sequence("111111","1111",16,seed=1)
