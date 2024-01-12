from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


def predict(text: str) -> str:
    """Run prediction. """
    saved_model = "./models/financial_tweets_sentiment_model/"
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-xsmall")
    encoded_input = tokenizer(text, return_tensors="pt")
    id2label = {
        0: "Bearish",
        1: "Bullish",
        2: "Neutral"
    }
    label2id = {
        "Bearish": 0,
        "Bullish": 1,
        "Neutral": 2
    }
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        saved_model, num_labels=3, id2label=id2label, label2id=label2id
    )
    model.eval()
    with torch.no_grad():
        logits = model(**encoded_input).logits

    scores = logits.softmax(dim=-1)
    print(f"SCORES: {scores}")
    best_id = scores.argmax(dim=-1).item()
    prediction = id2label[best_id]
    print(f"Prediction: {prediction}")

    return prediction


if __name__ == "__main__":
    text = "I think $TSLA is going to the moon!"
    predict(text)
