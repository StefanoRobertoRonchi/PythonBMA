import numpy as np
import pandas as pd

from BMA_SSVS.BMA import Bayesian_MA_SSVS, PiP, TopModels, data_preparation


def test_data_preparation_reads_csv(tmp_path):
    csv_path = tmp_path / "test_data.csv"
    df = pd.DataFrame({"y": [1, 2, 3], "x1": [4, 5, 6]})
    df.to_csv(csv_path, index=False)

    y, X = data_preparation(str(tmp_path), "test_data.csv", ["x1"], ["y"])

    assert "y" in y.columns
    assert "x1" in X.columns
    assert y.shape == (3, 1)
    assert X.shape == (3, 1)
    assert y.to_numpy().ravel().tolist() == [1, 2, 3]
    assert X.to_numpy().ravel().tolist() == [4, 5, 6]


def test_pip_computes_posterior_inclusion_probabilities():
    bma_output = {
        "Variable_Selected": np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]]),
        "added_intercept": False,
    }
    X = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6], "x3": [7, 8, 9]})

    pip = PiP(bma_output, X)

    assert pip.shape == (3,)
    assert list(pip.index) == ["x1", "x2", "x3"]
    assert pytest.approx(pip.iloc[0]) == np.mean([1, 1, 0])


def test_topmodels_returns_dataframe_with_expected_columns():
    X = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [2.0, 1.0, 0.0]})
    y = pd.Series([1.0, 2.0, 3.0])
    bma_output = {
        "Variable_Selected": np.array([[1, 1, 0], [1, 0, 1], [1, 1, 1], [1, 0, 0]]),
        "Betas": np.array([[0.5, 0.2, 0.1], [0.4, 0.0, -0.1], [0.6, 0.2, 0.2], [0.3, -0.1, 0.0]]),
        "added_intercept": True,
    }

    result = TopModels(bma_output, y, X, n_models=3)

    expected_cols = ["Intercept", "x1", "x2", "R2_mean", "R2_median", "R2_q05", "R2_q95", "count"]
    assert list(result.columns) == expected_cols
    assert result.shape[0] <= 3


def test_bayesian_ma_ssvs_returns_expected_structure():
    np.random.seed(0)
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    X = np.column_stack((np.arange(1, 6), np.arange(2, 7)))

    output = Bayesian_MA_SSVS(y=y, X=X, n=10, burn_in=5, seed=42, add_intercept=True, normalize_x=False, normalize_y=False)

    assert isinstance(output, dict)
    assert set(output.keys()) == {"Model_Number", "Variable_Selected", "Betas", "Model Variance", "added_intercept"}
    assert output["Variable_Selected"].shape == (5, 3)
    assert output["Betas"].shape == (5, 3)
    assert output["added_intercept"] is True
