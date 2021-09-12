from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
import torch


app = FastAPI()
MODEL_PATH = "model.pth"
lstm_model = torch.load(MODEL_PATH)
lstm_model.eval()

Vector = List[float]


class Request(BaseModel):
    sentences: List[str]


class Response(BaseModel):
    predictions: List[Vector]


@app.get("/")
def root():
    return {"message": "API running"}


@app.post("/model/predict")
def predict(body: Request):
    # will process 10 sentences max
    text = body.sentences[:10]

    probas = lstm_model.predict(text)
    return Response(predictions=probas.tolist())
