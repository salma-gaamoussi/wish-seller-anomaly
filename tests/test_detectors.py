import numpy as np
from src.detectors import (
    fit_iforest, score_iforest,
    fit_mahal, score_mahal
)


def sample_X():
    # 50 rows, 5 features
    return np.random.rand(50, 5)


def test_iforest_fit_and_score():
    X = sample_X()

    art = fit_iforest(X)

    # check keys exist
    assert "model" in art
    assert "scaler" in art
    assert "thresh" in art

    # threshold is float
    assert isinstance(art["thresh"], float)

    # score shape matches input
    scores = score_iforest(art, X)
    assert scores.shape[0] == X.shape[0]


def test_mahal_fit_and_score():
    X = sample_X()

    art = fit_mahal(X)

    # check keys
    assert "mean" in art
    assert "inv_cov" in art
    assert "thresh" in art

    # threshold is float
    assert isinstance(art["thresh"], float)

    # score shape matches input
    scores = score_mahal(art, X)
    assert scores.shape[0] == X.shape[0]