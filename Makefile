DATA = data/summer-products-with-rating-and-performance_2020-08.csv
IMG  = wish-anomaly-api

install:    pip install -r requirements.txt
train:      python src/train.py --data-path $(DATA)
test:       pytest tests/ -v
serve:      uvicorn api.main:app --reload --port 8000
dashboard:  streamlit run dashboard/app.py
build:      docker build -t $(IMG) .
run:        docker run -p 8000:8000 $(IMG)