from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
import evaluate
import numpy as np
from datasets import load_from_disk
import wandb
import logging
import sys
import hydra
from hydra.utils import get_original_cwd
import os

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)


accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


@hydra.main(config_path="config", config_name="train_config.yaml", version_base="1.1")
def main(cfg):
    # Hypereparameters
    model_id = cfg.hyperparameters.model_id
    lr = cfg.hyperparameters.lr
    batch_size = cfg.hyperparameters.batch_size
    num_epochs = cfg.hyperparameters.num_epochs
    output_dir = cfg.hyperparameters.output_dir
    train_log_dir = cfg.hyperparameters.train_log_dir
    weight_decay = cfg.hyperparameters.weight_decay
    eval_strategy = cfg.hyperparameters.eval_strategy
    save_strategy = cfg.hyperparameters.save_strategy

    # print(os.getcwd())
    original_working_dir = get_original_cwd() # because hydra changes the working directory

    train_set_path = os.path.join(original_working_dir, cfg.dataset.train_set_path)
    val_set_path = os.path.join(original_working_dir, cfg.dataset.val_set_path)
    output_dir = os.path.join(original_working_dir, output_dir)

    wandb.init(project="train", entity="dtu-mlops-financial-tweets")

    # Load data
    #tw_fin = load_dataset("zeroshot/twitter-financial-news-sentiment")
    train_set = load_from_disk(train_set_path)
    val_set = load_from_disk(val_set_path)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Preprocess data
    tokenized_train_set = train_set.map(preprocess_function, batched=True)
    tokenized_val_set = val_set.map(preprocess_function, batched=True)
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
        train_dataset=tokenized_train_set,
        eval_dataset=tokenized_val_set,
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
