
import json
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import GPTLanguageModel

app = FastAPI(title="GPT Next Character Prediction API")

# Load tokenizer
with open("tokenizer.json", "r", encoding="utf-8") as f:
    tokenizer = json.load(f)

stoi = tokenizer["stoi"]
itos = {int(k): v for k, v in tokenizer["itos"].items()}

# Load trained model
checkpoint = torch.load(
    "gpt_decoder_model.pt",
    map_location="cpu",
    weights_only=False
)

config = checkpoint["config"]
model = GPTLanguageModel(**config)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

class PredictionRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "GPT API is running"}

@app.post("/predict")
def predict(request: PredictionRequest):
    text = request.text

    unknown = [ch for ch in text if ch not in stoi]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail="Input contains characters not in the training vocabulary."
        )

    input_ids = torch.tensor(
        [[stoi[ch] for ch in text]],
        dtype=torch.long
    )

    with torch.no_grad():
        logits = model(input_ids)
        next_id = int(torch.argmax(logits[0, -1]).item())

    return {
        "input": text,
        "next_character": itos[next_id]
    }
