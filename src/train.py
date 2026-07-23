import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.data import load
from src.features import engineer
from src.detectors import fit_iforest, fit_mahal, score_iforest, score_mahal
import argparse, joblib
import pandas as pd

def main(path):
    df      = load(path)
    feats   = engineer(df)
    X       = feats.values

    if_art  = fit_iforest(X)
    mh_art  = fit_mahal(X)

    # attach scores + flags back to original df for inspection
    df["if_score"]  = score_iforest(if_art, X)
    df["mh_score"]  = score_mahal(mh_art, X)
    df["if_flag"]   = df["if_score"] > if_art["thresh"]
    df["mh_flag"]   = df["mh_score"] > mh_art["thresh"]

    # persist trained artefacts + scored dataset for the API/dashboard
    joblib.dump(if_art, "models/iforest.pkl")
    joblib.dump(mh_art, "models/mahal.pkl")
    joblib.dump(feats.columns.tolist(), "models/feature_cols.pkl")
    df.to_csv("models/scored_products.csv", index=False)

    n_if = df["if_flag"].sum()
    n_mh = df["mh_flag"].sum()
    print(f"Trained on {len(df)} products")
    print(f"  IF flagged:    {n_if} ({n_if/len(df)*100:.1f}%)")
    print(f"  Mahal flagged: {n_mh} ({n_mh/len(df)*100:.1f}%)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True)
    main(p.parse_args().data_path)