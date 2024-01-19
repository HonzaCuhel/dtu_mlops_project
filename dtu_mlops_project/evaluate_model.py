from datasets import load_from_disk
from evaluate import TextClassificationEvaluator
from transformers import AutoTokenizer, pipeline


def eval_model(model_path: str, dataset_path: str) -> dict:
    """Evaluate trained model on test dataset."""
    # Load model
    m_pipeline = pipeline(
        "text-classification",
        model=model_path,
        tokenizer=AutoTokenizer.from_pretrained("microsoft/deberta-v3-xsmall", use_fast=False),
    )
    # Load dataset
    dataset = load_from_disk(dataset_path)
    # Evaluate
    evaluator = TextClassificationEvaluator()
    results = evaluator.compute(
        m_pipeline, dataset, metric="accuracy", label_mapping={"Bearish": 0, "Bullish": 1, "Neutral": 2}
    )
    print(f"Model accuracy: {results['accuracy']*100}%")

    return results


if __name__ == "__main__":
    model_path = "./models/financial_tweets_sentiment_model/"
    dataset_path = "./data/processed/test"
    eval_model(model_path, dataset_path)
