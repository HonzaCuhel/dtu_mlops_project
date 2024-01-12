from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
import evaluate
import torch
import numpy as np

lr = 1e-3
batch_size = 16
num_epochs = 10
model_used = "microsoft/deberta-v3-xsmall"

from datasets import load_dataset
tw_fin = load_dataset("zeroshot/twitter-financial-news-sentiment")

seqeval = evaluate.load("seqeval")

label_list = [
    "Bearish", 
    "Bullish", 
    "Neutral",
]

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

tokenizer = AutoTokenizer.from_pretrained(model_used, add_prefix_space=True)

def tokenize_and_align_labels(exs):
    tokenized_inputs = tokenizer(exs["text"], truncation=True, is_split_into_words=False)

    labels = []
    for i, label in enumerate(exs[f"label"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_ids = None
        label_ids = []
        
        if word_ids is None:
            label_ids.append(-100)

        elif word_ids != previous_word_ids:
            label_ids.append(label)

        else:
            label_ids.append(-100)

        previous_word_ids = word_ids
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_tw_fin = tw_fin.map(tokenize_and_align_labels, batched=True)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

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

model = AutoModelForTokenClassification.from_pretrained(
    model_used, num_labels=3, id2label=id2label, label2id=label2id
)

peft_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1, bias="all"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"trainable params: 1855499 || all params: 355894283 || trainable%: 0.5213624069370061"

training_args = TrainingArguments(
    output_dir="token-classification",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset = tokenized_tw_fin["train"],
    eval_dataset = tokenized_tw_fin["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()


