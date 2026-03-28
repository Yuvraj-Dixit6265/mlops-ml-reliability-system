from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
import os

from pydantic import BaseModel, Field
from app.drift import detect_drift

# ✅ DEFINE CLASS FIRST
class IrisInput(BaseModel):
    sepal_length: float = Field(..., ge=0, le=10)
    sepal_width: float = Field(..., ge=0, le=10)
    petal_length: float = Field(..., ge=0, le=10)
    petal_width: float = Field(..., ge=0, le=10)

# FastAPI app
app = FastAPI()

# Load model
model = joblib.load("model/model.joblib")

# Root
@app.get("/")
def read_root():
    return {"message": "ML Model API is running"}

# ✅ Predict API
@app.post("/predict")
def predict(data: IrisInput):

    input_data = [
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]

    input_df = pd.DataFrame([{
        "sepal length (cm)": data.sepal_length,
        "sepal width (cm)": data.sepal_width,
        "petal length (cm)": data.petal_length,
        "petal width (cm)": data.petal_width
    }])

    prediction = model.predict(input_df)[0]

    log_data = {
        "sepal_length": data.sepal_length,
        "sepal_width": data.sepal_width,
        "petal_length": data.petal_length,
        "petal_width": data.petal_width,
        "prediction": int(prediction)
    }

    log_df = pd.DataFrame([log_data])
    log_file = "logs/data_log.csv"

    if os.path.exists(log_file):
        log_df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_file, index=False)

    return {"prediction": int(prediction)}

# Drift API
@app.get("/drift")
def check_drift():
    return detect_drift()