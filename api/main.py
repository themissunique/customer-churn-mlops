from fastapi import FastAPI
from pydantic import BaseModel

from src.pipeline.prediction_pipeline import PredictionPipeline
from prometheus_client import Counter, Histogram, generate_latest
from fastapi import Response
import time

app = FastAPI(
    title="Customer Churn Prediction API"
)

REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total prediction requests"
)

REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency"
)

class CustomerData(BaseModel):

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


pipeline = PredictionPipeline()


@app.get("/")
def home():

    return {
        "message": "Customer Churn Prediction API"
    }

@app.get("/metrics")
def metrics():

    return Response(
        generate_latest(),
        media_type="text/plain"
    )

@app.post("/predict")
def predict(data: CustomerData):

    REQUEST_COUNT.inc()

    start = time.time()

    prediction = pipeline.predict(data.model_dump())

    REQUEST_LATENCY.observe(
        time.time() - start
    )

    return {
        "prediction": prediction
    }