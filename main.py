"""FastAPI service that serves the Telco customer-churn model saved by train.py."""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
CLASS_NAMES = {0: "stay", 1: "churn"}

ml_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the trained model artifact once, when the service starts.
    if not MODEL_PATH.exists():
        raise RuntimeError(f"{MODEL_PATH.name} not found. Run `python train.py` first.")
    ml_model["pipeline"] = joblib.load(MODEL_PATH)
    yield
    ml_model.clear()


app = FastAPI(
    title="Telco Customer Churn API",
    description=(
        "Predicts whether a telecom customer will **churn** (leave the company). "
        "Model: scikit-learn pipeline (imputer + StandardScaler + OneHotEncoder + classifier) "
        "trained on the IBM Telco Customer Churn dataset."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

YesNo = Literal["Yes", "No"]
InternetAddon = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    SeniorCitizen: YesNo = Field(description="Customer is 65 or older")
    Partner: YesNo = Field(description="Customer has a partner")
    Dependents: YesNo = Field(description="Customer has dependents")
    tenure: int = Field(ge=0, le=120, description="Months the customer has stayed with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0, description="Current monthly charge, USD")
    TotalCharges: float = Field(ge=0, description="Total amount charged so far, USD")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "SeniorCitizen": "No", "Partner": "No", "Dependents": "No", "tenure": 2,
                "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
                "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
                "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check", "MonthlyCharges": 95.5, "TotalCharges": 190.0,
            }]
        }
    }


class PredictionResponse(BaseModel):
    prediction: int = Field(description="0 = customer stays, 1 = customer churns")
    label: Literal["stay", "churn"]
    churn_probability: float = Field(description="Predicted probability of churn")


@app.get("/")
def root():
    return {"message": "ML API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: Customer):
    pipeline = ml_model.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    X = pd.DataFrame([customer.model_dump()])
    pred = int(pipeline.predict(X)[0])
    proba = float(pipeline.predict_proba(X)[0][1])
    return PredictionResponse(prediction=pred, label=CLASS_NAMES[pred], churn_probability=round(proba, 4))
