from evaluate import TextClassificationEvaluator
from transformers import pipeline, AutoTokenizer
from datasets import load_from_disk


def eval_model(model_path, dataset_path):
    """Evaluate model on dataset. """
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
        m_pipeline,
        dataset,
        metric="accuracy",
        label_mapping={
            "Bearish": 0,
            "Bullish": 1,
            "Neutral": 2
        }
    )
    #print(results)
    print(f"Accuracy: {results['accuracy']}")
    return results


if __name__ == "__main__":
    model_path = "./models/financial_tweets_sentiment_model_10_ep/"
    dataset_path = "./data/processed/test"
    eval_model(model_path, dataset_path)
