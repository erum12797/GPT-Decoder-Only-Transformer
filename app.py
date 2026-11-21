from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="GPT Next Token Prediction API")

class PredictionRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "GPT API is running"}

@app.post("/predict")
def predict(request: PredictionRequest):
    return {
        "input": request.text,
        "message": "Model loading and prediction will be connected next."
    }
