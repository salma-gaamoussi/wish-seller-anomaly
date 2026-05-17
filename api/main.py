import sqlite3, joblib, datetime
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from src.features import engineer
from src.detectors import score_iforest, score_mahal

app = FastAPI(title="Wish Seller Anomaly API")

_if   = joblib.load("models/iforest.pkl")
_mh   = joblib.load("models/mahal.pkl")

DB = "models/predictions.db"
with sqlite3.connect(DB) as c:
    c.execute("""CREATE TABLE IF NOT EXISTS logs
        (ts TEXT, product TEXT, if_score REAL, mh_score REAL,
         if_flag INT, mh_flag INT)""")


class Product(BaseModel):
    title:               str   = "unknown"
    price:               float
    retail_price:        float
    units_sold:          float
    uses_ad_boosts:      float
    rating:              float
    rating_count:        float
    merchant_rating:     float = 4.0
    merchant_rating_count: float = 100
    countries_shipped_to: float = 10
    badges_count:        float = 0
    badge_fast_shipping: float = 0
    shipping_option_price: float = 0

class AnomalyResult(BaseModel):
    if_score: float;  mh_score: float
    if_flag:  bool;   mh_flag:  bool
    verdict:  str     # "SUSPICIOUS" / "NORMAL"


@app.get("/health")
def health(): return {"status": "ok"}


@app.post("/predict", response_model=AnomalyResult)
def predict(p: Product):
    row  = pd.DataFrame([p.model_dump()])
    feat = engineer(row).values     # feature engineering happens here
    ifs  = float(score_iforest(_if, feat)[0])
    mhs  = float(score_mahal(_mh, feat)[0])
    flag = (ifs > _if["thresh"]) or (mhs > _mh["thresh"])

    with sqlite3.connect(DB) as c:
        c.execute("INSERT INTO logs VALUES (?,?,?,?,?,?)",
                  (datetime.datetime.utcnow().isoformat(),
                   p.title, ifs, mhs,
                   int(ifs > _if["thresh"]),
                   int(mhs > _mh["thresh"])))

    return AnomalyResult(
        if_score=round(ifs, 4), mh_score=round(mhs, 2),
        if_flag=ifs > _if["thresh"], mh_flag=mhs > _mh["thresh"],
        verdict="SUSPICIOUS" if flag else "NORMAL",
    )