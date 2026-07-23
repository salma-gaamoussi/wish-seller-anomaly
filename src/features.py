import pandas as pd
import numpy as np

FEATURE_COLS = [
    "discount_rate", "review_to_sales_ratio",
    "ad_efficiency", "rating_quality", "merchant_trust_gap",
]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # 1. how deep is the discount really?
    out["discount_rate"] = (
        (df["retail_price"] - df["price"]) / df["retail_price"]
    ).clip(0, 1)

    # 2. reviews relative to sales — fake review signal
    out["review_to_sales_ratio"] = (
        df["rating_count"] / (df["units_sold"] + 1)
    )

    # 3. sales given ad-boost usage. Dataset only exposes a binary
    # uses_ad_boosts flag (no $ spend), so this is a coarse proxy: it flags
    # products that turned on paid promotion but still sold poorly, not
    # actual wasted spend.
    out["ad_efficiency"] = (
        df["units_sold"] / (df["uses_ad_boosts"] + 1)
    )

    # 4. rating weighted by review volume — penalises thin 5-stars
    out["rating_quality"] = (
        df["rating"] * np.log1p(df["rating_count"])
    )

    # 5. product looks better than its seller history
    out["merchant_trust_gap"] = (
        df["rating"] - df["merchant_rating"].fillna(df["rating"])
    )

    return out[FEATURE_COLS]