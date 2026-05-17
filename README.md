# wish-seller-anomaly
## What it does
Detects anomalous seller behavior on a digital marketplace using
unsupervised ML on 5 engineered behavioral signals: price manipulation,
fake reviews, ad fraud, rating inflation, and merchant-product trust gaps.

## Engineered features
| Feature               | Signal                                      |
|-----------------------|---------------------------------------------|
| discount_rate         | Inflated retail prices to fake big discounts|
| review_to_sales_ratio | Too many reviews relative to actual sales   |
| ad_efficiency         | Heavy ad spend with near-zero sales         |
| rating_quality        | 5-star products with 2 reviews              |
| merchant_trust_gap    | Product looks better than merchant history  |

## How to run
```bash
make install && make train
make serve        # API → localhost:8000/docs
make dashboard    # UI  → localhost:8501
```