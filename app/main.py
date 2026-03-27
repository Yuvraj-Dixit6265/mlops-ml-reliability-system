from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

# NEW IMPORT
import whylogs as why
from datetime import datetime

#  FastAPI app initialize
app = FastAPI()

# Load trained model
model = joblib.load("model/model.joblib")

# Define input schema using Pydantic
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "ML Model API is running"}

# Prediction endpoint
@app.post("/predict")
def predict(data: IrisInput):

    input_data = [
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]

    # Convert to numpy
    input_array = np.array([input_data])

    # Prediction
    prediction = model.predict(input_array)[0]

    # 🔥 WHYLOGS LOGGING
    log_data = {
        "sepal_length": data.sepal_length,
        "sepal_width": data.sepal_width,
        "petal_length": data.petal_length,
        "petal_width": data.petal_width,
        "prediction": int(prediction)
    }

    profile = why.log(log_data)

    # Save log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    profile.write(f"logs/profile_{timestamp}.bin")

    return {
        "prediction": int(prediction)
    }