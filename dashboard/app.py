import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ROOT, "data", "summer-products-with-rating-and-performance_2020-08.csv")

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from src.drift import drift_report
from src.features import engineer, FEATURE_COLS
from data.data import load

st.set_page_config(page_title="Seller Anomaly Monitor", layout="wide")
st.title("🛍 Wish Seller Anomaly Monitor")

scored = pd.read_csv(os.path.join(ROOT, "models", "scored_products.csv"))

# Combined verdict mirrors api/main.py's predict(): SUSPICIOUS if either
# detector flags it. Keeping this logic identical in both places means the
# dashboard and the API never disagree on the same product.
scored["verdict"] = np.where(
    scored["if_flag"] | scored["mh_flag"], "SUSPICIOUS", "NORMAL"
)

# ── Panel 1: Top suspicious products ─────────────────────
st.subheader("Most anomalous products")

n_susp = (scored["verdict"] == "SUSPICIOUS").sum()
st.metric("Flagged SUSPICIOUS (IF OR Mahalanobis)", f"{n_susp} / {len(scored)}")

tab0, tab1, tab2 = st.tabs(["Combined verdict", "Isolation Forest", "Mahalanobis"])

with tab0:
    combined = (scored[scored["verdict"] == "SUSPICIOUS"]
                [["title_orig", "price", "retail_price", "units_sold",
                  "rating", "if_score", "mh_score", "verdict"]]
                .sort_values("if_score", ascending=False))
    st.dataframe(combined, use_container_width=True)

with tab1:
    top_if = (scored.sort_values("if_score", ascending=False)
              [["title_orig", "price", "retail_price",
                "units_sold", "rating", "if_score", "if_flag"]]
              .head(20))
    st.dataframe(
        top_if,
        column_config={
            "if_score": st.column_config.ProgressColumn(
                "Anomaly score", min_value=0, max_value=float(scored["if_score"].max())),
            "if_flag": st.column_config.CheckboxColumn("Flagged"),
        },
        use_container_width=True,
    )

with tab2:
    top_mh = (scored.sort_values("mh_score", ascending=False)
              [["title_orig", "price", "retail_price",
                "units_sold", "rating", "mh_score", "mh_flag"]]
              .head(20))
    st.dataframe(top_mh, use_container_width=True)

st.divider()

# ── Panel 2: Feature drift across seller segments ─────────
st.subheader("Feature drift — seller segments")
segment = st.selectbox(
    "Compare baseline (no ad boosts) against:",
    ["Heavy ad boosts (uses_ad_boosts=1)",
     "Low-rated merchants (merchant_rating < 4)",
     "High discount sellers (discount_rate > 0.5)"]
)

df = load(DATA_PATH)
feats  = engineer(df)
feats["uses_ad_boosts"]   = df["uses_ad_boosts"].values
feats["merchant_rating"]  = df["merchant_rating"].values
feats["discount_rate_raw"] = feats["discount_rate"].values

ref  = feats[feats["uses_ad_boosts"] == 0][FEATURE_COLS]

if "Heavy ad" in segment:
    prod = feats[feats["uses_ad_boosts"] == 1][FEATURE_COLS]
elif "Low-rated" in segment:
    prod = feats[feats["merchant_rating"] < 4][FEATURE_COLS]
else:
    prod = feats[feats["discount_rate_raw"] > 0.5][FEATURE_COLS]

report = drift_report(ref, prod)
st.dataframe(
    report,
    column_config={"psi": st.column_config.ProgressColumn(
        "PSI", min_value=0, max_value=0.5)},
    use_container_width=True,
)
n = (report["status"] == "🔴 ALERT").sum()
if n:
    st.error(f"⚠ {n} features show significant drift")
else:
    st.success("✓ All features stable in this segment")