import numpy as np
import pandas as pd

PSI_WARN  = 0.10
PSI_ALERT = 0.20


def psi(ref: np.ndarray, prod: np.ndarray, n_bins: int = 10) -> float:
    breaks = np.percentile(ref, np.linspace(0, 100, n_bins + 1))
    breaks[0], breaks[-1] = -np.inf, np.inf

    r = np.clip(np.histogram(ref,  bins=breaks)[0] / len(ref),  1e-6, None)
    p = np.clip(np.histogram(prod, bins=breaks)[0] / len(prod), 1e-6, None)

    return float(np.sum((p - r) * np.log(p / r)))


def drift_report(X_ref: pd.DataFrame, X_prod: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for col in X_ref.columns:
        v = psi(X_ref[col].values, X_prod[col].values)

        status = (
            "🔴 ALERT" if v > PSI_ALERT
            else "🟡 WARN" if v > PSI_WARN
            else "🟢 OK"
        )

        rows.append({
            "feature": col,
            "psi": round(v, 4),
            "status": status
        })

    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def production_batches(df: pd.DataFrame, n: int = 5) -> list:
    from src.features import FEATURE_COLS

    anomalies = df[(df["if_flag"]) | (df["mh_flag"])]

    return np.array_split(anomalies[FEATURE_COLS], n)