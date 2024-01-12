from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
import evaluate
import numpy as np
from datasets import load_dataset
import wandb


accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def main():
    # Hypereparameters
    model_id = "microsoft/deberta-v3-xsmall"
    lr = 1e-4
    batch_size = 32
    num_epochs = 1
    # import os
    # print(os.getcwd())
    output_dir = "models/financial_tweets_sentiment_model"
    train_log_dir = "./runs"
    weight_decay = 0.01
    eval_strategy = "epoch"
    save_strategy = "epoch"
    
    wandb.init(project="train", entity="dtu-mlops-financial-tweets")

    # Load data
    tw_fin = load_dataset("zeroshot/twitter-financial-news-sentiment")

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Preprocess data
    tokenized_tw_fin = tw_fin.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

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
        model_id, num_labels=3, id2label=id2label, label2id=label2id
    )
    # Specify training arguments
    training_args = TrainingArguments(
        output_dir=train_log_dir,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        evaluation_strategy=eval_strategy,
        save_strategy=save_strategy,
        load_best_model_at_end=True
    )
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_tw_fin["train"],
        eval_dataset=tokenized_tw_fin["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    # Train model
    trainer.train()
    # Save model
    model.save_pretrained(output_dir)
    # Evaluate model
    trainer.evaluate()


if __name__ == "__main__":
    main()
