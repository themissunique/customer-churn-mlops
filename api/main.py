from fastapi import FastAPI
from pydantic import BaseModel

from src.pipeline.prediction_pipeline import PredictionPipeline


app = FastAPI(
    title="Customer Churn Prediction API"
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


@app.post("/predict")
def predict(data: CustomerData):

    prediction = pipeline.predict(data.model_dump())

    return {
        "prediction": prediction.item()
    }