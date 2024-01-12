from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
import evaluate
import torch
import numpy as np
from datasets import load_dataset

lr = 1e-3
batch_size = 16
num_epochs = 1
# model_used = "microsoft/deberta-v3-xsmall"
model_used = "distilbert-base-uncased"
# model_used = "microsoft/deberta-v3-xsmall"

imdb = load_dataset("imdb")
print(imdb)

tw_fin = load_dataset("zeroshot/twitter-financial-news-sentiment")
print(tw_fin)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

tokenizer = AutoTokenizer.from_pretrained(model_used)

def preprocess_function(examples):
    # print(examples['text'])
    # input_data = tokenizer(examples["text"], truncation=True, padding='max_length', max_length=256)
    return tokenizer(examples["text"], truncation=True)
    # print(input_data)
    # return input_data

tokenized_tw_fin = tw_fin.map(preprocess_function, batched=True)
# tokenized_tw_fin = imdb.map(preprocess_function, batched=True)

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

model = AutoModelForTokenClassification.from_pretrained(
    model_used, num_labels=3, id2label=id2label, label2id=label2id
)

# id2label = {0: "NEGATIVE", 1: "POSITIVE"}
# label2id = {"NEGATIVE": 0, "POSITIVE": 1}
# model = AutoModelForSequenceClassification.from_pretrained(
#     model_used, num_labels=2, id2label=id2label, label2id=label2id
# )

print(tokenized_tw_fin)

training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_tw_fin["train"],
    eval_dataset=tokenized_tw_fin["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_tw_fin["train"],
#     eval_dataset=tokenized_tw_fin["test"],
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
# )

# trainer.train()

# # peft_config = LoraConfig(
# #     task_type=TaskType.TOKEN_CLS, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1, bias="all"
# # )

# # model = get_peft_model(model, peft_config)
# # model.print_trainable_parameters()
# # "trainable params: 1855499 || all params: 355894283 || trainable%: 0.5213624069370061"

# training_args = TrainingArguments(
#     output_dir="financial_news_sentiment",
#     learning_rate=lr,
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     num_train_epochs=num_epochs,
#     weight_decay=0.01,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     load_best_model_at_end=True,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset = tokenized_tw_fin["train"],
#     eval_dataset = tokenized_tw_fin["validation"],
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
# )

# trainer.train()

# trainer.save_pretrained("financial_news_sentiment_model")

# trainer.evaluate()
# # trainer.train()


