# =========================
# app.py – Customer Churn API
# =========================

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
import time

# Monitoring
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

# -------------------------
# PATH & MODEL LOADING
# -------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "best_churn_pipeline.pkl")

print("📂 Base directory:", BASE_DIR)
print("📂 Model directory exists:", os.path.exists(MODEL_DIR))

if os.path.exists(MODEL_DIR):
    print("📄 Files in model directory:", os.listdir(MODEL_DIR))
else:
    print("❌ Model directory NOT FOUND")

print("📦 Loading model from:", MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. "
        "Make sure best_churn_pipeline.pkl exists in the model folder."
    )

model = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully")

# -------------------------
# FASTAPI APP
# -------------------------

app = FastAPI(title="Customer Churn Prediction API")

# -------------------------
# MONITORING METRICS
# -------------------------

REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total prediction requests"
)

PREDICTION_TIME = Histogram(
    "prediction_latency_seconds",
    "Prediction latency"
)

CHURN_COUNTER = Counter(
    "churn_predictions_total",
    "Churn predictions by class",
    ["label"]
)

# -------------------------
# INPUT SCHEMA
# -------------------------

class Customer(BaseModel):
    credit_score: int
    country: str
    gender: str
    age: int
    tenure: int
    balance: float
    products_number: int
    credit_card: int
    active_member: int
    estimated_salary: float

# -------------------------
# HEALTH CHECK
# -------------------------

@app.get("/")
def root():
    return {"status": "Customer Churn API is running"}

# -------------------------
# PREDICTION ENDPOINT
# -------------------------

@app.post("/predict")
def predict_churn(customer: Customer):
    start_time = time.time()
    REQUEST_COUNT.inc()

    # Convert input to DataFrame
    data = pd.DataFrame([customer.dict()])

    # -------- Feature Engineering (MUST MATCH TRAINING) --------
    data["balance_per_product"] = data["balance"] / data["products_number"].replace(0, np.nan)
    data["balance_per_product"].fillna(0, inplace=True)

    data["salary_balance_ratio"] = data["estimated_salary"] / data["balance"].replace(0, np.nan)
    data["salary_balance_ratio"].replace([np.inf, -np.inf], np.nan, inplace=True)
    data["salary_balance_ratio"].fillna(data["salary_balance_ratio"].median(), inplace=True)

    data["age_group"] = pd.cut(
        data["age"],
        bins=[0, 25, 35, 45, 55, 65, 100],
        labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
    )

    data["tenure_bucket"] = pd.cut(
        data["tenure"],
        bins=[-1, 0, 2, 5, 10, 100],
        labels=["0", "1-2", "3-5", "6-10", "10+"]
    )

    data["high_balance"] = (data["balance"] > 50000).astype(int)

    # -------- Prediction --------
    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0, 1])

    CHURN_COUNTER.labels(label=str(prediction)).inc()
    PREDICTION_TIME.observe(time.time() - start_time)

    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability, 3)
    }

# -------------------------
# METRICS ENDPOINT
# -------------------------

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
