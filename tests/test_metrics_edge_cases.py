"""Actual metric functions must not turn missing baselines into perfect skill."""
import json
import numpy as np
import pytest
from src.evaluation.metrics import mae, rmse, mape, skill_score, evaluate_all, compare_models, quantile_coverage

@pytest.mark.parametrize("prediction", [20.0, 30.0])
def test_singleton_skill_is_undefined_not_perfect(prediction):
    assert skill_score(np.array([20.0]), np.array([prediction])) is None

@pytest.mark.parametrize("actuals,predictions", [([20,20],[30,30]), ([0,0],[1,1]), ([-5,-5],[-4,-4])])
def test_constant_reference_has_no_defined_relative_skill(actuals,predictions):
    assert skill_score(np.array(actuals),np.array(predictions)) is None

@pytest.mark.parametrize("fn", [mae,rmse,mape,skill_score])
@pytest.mark.parametrize("a,p", [([],[]),([1,2],[1]),([float('nan')],[1]),([1],[float('inf')]),([[1]],[[1]])])
def test_invalid_metric_input_is_an_error(fn,a,p):
    with pytest.raises(ValueError): fn(np.array(a),np.array(p))

def test_zero_actual_mape_and_skill_are_json_safe_with_honest_status():
    result=evaluate_all(np.array([0.0]),np.array([10.0]))
    assert result['mae']==10
    assert result['rmse']==10
    assert result['mape'] is None
    assert result['skill_score'] is None
    assert result['n_observations']==1
    assert result['skill_score_status']=='undefined_zero_baseline'
    json.dumps(result,allow_nan=False)

def test_explicit_nonzero_reference_preserves_singleton_skill():
    assert skill_score(np.array([20.0]),np.array([20.0]),climatology=10)==1
    assert skill_score(np.array([20.0]),np.array([30.0]),climatology=10)==0

def test_normal_negative_temperatures_are_supported():
    a=np.array([-10.,-2.]);p=np.array([-9.,-1.])
    assert mae(a,p)==1
    assert rmse(a,p)==1
    assert skill_score(a,p)==0.75
    assert mape(a,p)==30

def test_bad_intervals_are_rejected():
    with pytest.raises(ValueError): quantile_coverage(np.array([1.]),np.array([2.]),np.array([0.]))
    with pytest.raises(ValueError): evaluate_all(np.array([1.]),np.array([1.]),lower=np.array([0.]))

def test_compare_empty_and_constant_series():
    with pytest.raises(ValueError): compare_models(np.array([]),{})
    result=compare_models(np.array([20.]),{'wrong':np.array([30.])})
    assert result.loc['wrong','skill_score_status']=='undefined_zero_baseline'

def test_api_contract_accepts_undefined_metrics_without_fabricated_zero():
    from src.api.routes.metrics import HorizonMetrics
    model=HorizonMetrics(**evaluate_all(np.array([20.]),np.array([30.])))
    payload=model.model_dump()
    assert payload['skill_score'] is None
    assert payload['n_observations']==1
    assert payload['skill_score_status']=='undefined_zero_baseline'
    json.dumps(payload,allow_nan=False)
