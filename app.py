
import os
import torch
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import GPTLanguageModel

app = FastAPI(title="GPT Next Character Prediction API")

# Allow the Vercel frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Find trained checkpoint
BASE_DIR = Path(__file__).resolve().parent
checkpoint_path = os.getenv(
    "CHECKPOINT_PATH",
    str(BASE_DIR / "gpt_decoder_model.pt")
)

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError("GPT checkpoint file not found.")

# Load trained model
device = "cpu"

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False
)

config = checkpoint["model_config"]

model = GPTLanguageModel(**config)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load tokenizer directly from checkpoint
stoi = {
    str(k): int(v)
    for k, v in checkpoint["stoi"].items()
}

itos = {
    int(k): str(v)
    for k, v in checkpoint["itos"].items()
}

class PredictionRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "GPT API is running successfully"}

@app.post("/predict")
def predict(request: PredictionRequest):

    text = request.text

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Please enter some text."
        )

    unknown = [ch for ch in text if ch not in stoi]

    if unknown:
        raise HTTPException(
            status_code=400,
            detail="Input contains characters outside the training vocabulary."
        )

    input_ids = [stoi[ch] for ch in text]
    input_ids = input_ids[-config["block_size"]:]

    input_tensor = torch.tensor(
        [input_ids],
        dtype=torch.long
    )

    with torch.no_grad():
        logits = model(input_tensor)
        next_id = int(logits[0, -1].argmax().item())

    return {
        "input": text,
        "next_character": itos[next_id]
    }
