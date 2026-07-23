import pandas as pd

RAW_COLS = [
    "price", "retail_price", "units_sold",
    "uses_ad_boosts", "rating", "rating_count",
    "merchant_rating", "merchant_rating_count",
    "countries_shipped_to", "badges_count",
    "badge_fast_shipping", "shipping_option_price",
]


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # keep only columns we need + title for display
    cols = RAW_COLS + ["title_orig", "merchant_name"]
    df   = df[[c for c in cols if c in df.columns]].copy()

    # units_sold is stored as strings like "1000+" — strip and cast
    df["units_sold"] = (
        df["units_sold"].astype(str)
        .str.replace("+", "", regex=False)
        .str.replace(",", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )

    # drop rows missing any key numeric field
    df = df.dropna(subset=["price", "retail_price",
                            "units_sold", "rating", "rating_count"])

    # sanity: retail_price must be >= price
    df = df[df["retail_price"] >= df["price"]]
    df = df[df["price"] > 0]
    df = df.reset_index(drop=True)
    return df