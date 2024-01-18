from http import HTTPStatus
from typing import List

import torch
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = FastAPI()
Instrumentator().instrument(app).expose(app)


model = AutoModelForSequenceClassification.from_pretrained("../models/financial_tweets_sentiment_model/")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-xsmall", use_fast=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    model = model.half()

model.to(device)
model.eval()


@app.get("/")
def root():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


def classify(texts: List[str]):
    encoded_input = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        logits = model(**encoded_input).logits
    scores = logits.softmax(dim=-1)
    best_ids = scores.argmax(dim=-1).tolist()
    preds = [model.config.id2label[best_id] for best_id in best_ids]
    return preds


@app.post("/predict_batch/")
def predict_sentiment(texts: List[str]):
    # Remove duplicates
    texts = list(set(texts))
    # Classify
    preds = classify(texts)

    response = {"input": texts, "message": HTTPStatus.OK.phrase, "status-code": HTTPStatus.OK, "output": preds}
    return response


# uvicorn --reload --port 8000 main:app
# reload => when saved, the server will reload
# curl -X 'POST' \
#   'http://127.0.0.1:8000/predict_batch/' \
#   -H 'accept: application/json' \
#   -H 'Content-Type: application/json' \
#   -d '[
#   "Tesla is going down",
#   "Apple is going up"
# ]'
