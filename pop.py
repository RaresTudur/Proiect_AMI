import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from datasets import Dataset

from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    AdamW, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoConfig,
    Trainer,
    TrainingArguments
)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

df = pd.read_csv("data.csv")

x_train, x_val, y_train, y_val = train_test_split(
    df["headlines"],
    df["outcome"],
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=df["outcome"]
)

train_df = pd.DataFrame({"text": x_train, "label": y_train})
val_df = pd.DataFrame({"text": x_val, "label": y_val})

print(f"\nTrain size: {len(train_df)}, Val size: {len(val_df)}")

train_dataset = Dataset.from_pandas(train_df)
val_dataset   = Dataset.from_pandas(val_df)

model_name = "bert-base-uncased"

from transformers import AutoConfig

tokenizer = BertTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
   model_name,
   num_labels=2)

def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset   = val_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.remove_columns(["text"])
val_dataset   = val_dataset.remove_columns(["text"])

train_dataset.set_format("torch")
val_dataset.set_format("torch")

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

predictions = trainer.predict(val_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

print("\nRaport de clasificare:")
print(classification_report(y_true, y_pred, target_names=["Fake","Real"]))


def prezice_stire(stire: str) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    enc = tokenizer(
        stire,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors='pt'
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs.logits
        clasa = torch.argmax(logits, dim=1).cpu().numpy()[0]

    return "Real" if clasa == 1 else "Fake"

while 1 > 0:
    print("What news headline you want to find out?")
    test_text = input()
    eticheta_prezisa = prezice_stire(test_text)
    print(f"\nText: {test_text}\nPredicție: {eticheta_prezisa}")
