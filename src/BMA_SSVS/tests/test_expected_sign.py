import pytest

from BMA_SSVS.BMA import expected_sign_check


def test_expected_sign_check_respects_expected_signs():
    regressors = ["x1", "x2", "x3"]
    betas = [1.5, -2.0, 0.1]
    sign_dict = {"x1": "positive", "x2": "neg", "x3": "+"}

    assert expected_sign_check(sign_dict, regressors, betas) is True


def test_expected_sign_check_rejects_wrong_sign():
    regressors = ["x1", "x2", "x3"]
    betas = [1.5, -2.0, 0.0]

    assert expected_sign_check({"x2": "positive"}, regressors, betas) is False
    assert expected_sign_check({"x1": "negative"}, regressors, betas) is False


def test_expected_sign_check_invalid_inputs():
    regressors = ["x1", "x2"]
    betas = [1.0, -1.0]

    with pytest.raises(TypeError):
        expected_sign_check("not a dict", regressors, betas)

    with pytest.raises(ValueError):
        expected_sign_check({"x3": "positive"}, regressors, betas)
