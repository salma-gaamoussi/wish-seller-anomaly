import pandas as pd
import numpy as np
import os
import sys
# sys.path.append(os.path.abspath('..'))

from src.features import engineer, FEATURE_COLS


def sample_df():
    return pd.DataFrame({
        "retail_price":   [100, 200, 150],
        "price":          [80, 150, 150],
        "rating_count":   [10, 50, 0],
        "units_sold":     [5, 40, 0],
        "uses_ad_boosts": [1, 5, 0],
        "rating":         [4.5, 3.8, 5.0],
        "merchant_rating":[4.0, np.nan, 4.5],
    })


def test_columns_exist():
    df = sample_df()
    out = engineer(df)

    for col in FEATURE_COLS:
        assert col in out.columns


def test_no_nulls():
    df = sample_df()
    out = engineer(df)

    assert not out.isnull().any().any()


def test_discount_rate_range():
    df = sample_df()
    out = engineer(df)

    assert (out["discount_rate"] >= 0).all()
    assert (out["discount_rate"] <= 1).all()


def test_discount_rate_correct():
    df = sample_df()
    out = engineer(df)

    expected = (df["retail_price"] - df["price"]) / df["retail_price"]
    expected = expected.clip(0, 1)

    assert np.allclose(out["discount_rate"], expected)