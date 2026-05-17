import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def fit_iforest(X: np.ndarray) -> dict:
    scaler = StandardScaler().fit(X)
    X_sc   = scaler.transform(X)
    model  = IsolationForest(n_estimators=200, contamination=0.01,
                              random_state=42, n_jobs=-1).fit(X_sc)
    thresh = float(np.percentile(-model.decision_function(X_sc), 99))
    artefact = {"model": model, "scaler": scaler, "thresh": thresh}
    joblib.dump(artefact, "models/iforest.pkl")
    return artefact

def score_iforest(art: dict, X: np.ndarray) -> np.ndarray:
    return -art["model"].decision_function(art["scaler"].transform(X))


def fit_mahal(X: np.ndarray) -> dict:
    mean    = X.mean(axis=0)
    inv_cov = np.linalg.pinv(np.cov(X, rowvar=False))
    thresh  = float(np.percentile(
        [float(d @ inv_cov @ d) for d in X - mean], 99
    ))
    artefact = {"mean": mean, "inv_cov": inv_cov, "thresh": thresh}
    joblib.dump(artefact, "models/mahal.pkl")
    return artefact

def score_mahal(art: dict, X: np.ndarray) -> np.ndarray:
    diff = X - art["mean"]
    return np.array([float(d @ art["inv_cov"] @ d) for d in diff])