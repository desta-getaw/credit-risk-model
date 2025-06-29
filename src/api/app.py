# src/api/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load the best model (update path as needed)
model = joblib.load("models/best_model.pkl")

app = FastAPI(title="Credit Risk Model API")

class PredictRequest(BaseModel):
    feature_vector: list  # input features as a list

class PredictResponse(BaseModel):
    prediction: int

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    prediction = model.predict([req.feature_vector])[0]
    return PredictResponse(prediction=int(prediction))
