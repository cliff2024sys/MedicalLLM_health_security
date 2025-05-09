#!/usr/bin/env python
# fine_tune_biomistral_cupcase.py

"""
Fine-tune BioMistral-7B on the CUPCase dataset for diagnosis classification.

This script will:
 1. Load HF_API_TOKEN, HF_MODEL_ID, HF_PUSH_TO from .env.
 2. Load & sanitize the single-split CupCase dataset.
 3. Build label2id/id2label over all examples.
 4. Split 90/10 into train/validation.
 5. Tokenize with 512-token truncation.
 6. If CUDA is available:
      • Load BioMistral-7B in 8-bit (BitsAndBytesConfig) + SeqCls head.
      • Prepare for k-bit training.
    Else (CPU only):
      • Load BioMistral-7B in full precision with CPU offloading.
 7. Apply LoRA via PEFT.
 8. Train with HF Trainer (early stopping + metrics).
 9. Push your LoRA adapters to your HF_PUSH_TO repo.
"""

import os
import numpy as np
import torch
from dotenv import load_dotenv
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model
from peft.utils.other import prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ────────────────────────────────────────────────────────────────────────
# 1. Load environment variables
# ────────────────────────────────────────────────────────────────────────
load_dotenv()
HF_API_TOKEN  = os.getenv("HF_API_TOKEN")
BASE_MODEL_ID = os.getenv("HF_MODEL_ID")   # e.g. "mistralai/biomistral-7b"
PUSH_REPO_ID  = os.getenv("HF_PUSH_TO")    # e.g. "cnj3v/bio-mistral-cupcase"

if not HF_API_TOKEN or not BASE_MODEL_ID or not PUSH_REPO_ID:
    raise ValueError("Please set HF_API_TOKEN, HF_MODEL_ID, HF_PUSH_TO in your .env")

# ────────────────────────────────────────────────────────────────────────
# 2. Load & sanitize CupCase dataset (single split → Dataset)
# ────────────────────────────────────────────────────────────────────────
raw = load_dataset("ofir408/CupCase")
if isinstance(raw, dict):
    split = next(iter(raw.keys()))
    raw = raw[split]

def sanitize(ex):
    return {
        "text": str(ex.get("clean_case_presentation","") or ""),
        "label_txt": str(ex.get("correct_diagnosis","") or "")
    }

raw = raw.map(sanitize, remove_columns=raw.column_names)

# ────────────────────────────────────────────────────────────────────────
# 3. Build label2id / id2label over all examples
# ────────────────────────────────────────────────────────────────────────
all_labels = sorted(set(raw["label_txt"]))
label2id = {lbl: idx for idx, lbl in enumerate(all_labels)}
id2label = {idx: lbl for lbl, idx in label2id.items()}

def label_map(ex):
    ex["labels"] = label2id[ex["label_txt"]]
    return ex

raw = raw.map(label_map, remove_columns=["label_txt"])

# ────────────────────────────────────────────────────────────────────────
# 4. Train/validation split (90/10)
# ────────────────────────────────────────────────────────────────────────
split_ds = raw.train_test_split(test_size=0.1, seed=42)
ds = DatasetDict({"train": split_ds["train"], "validation": split_ds["test"]})

# ────────────────────────────────────────────────────────────────────────
# 5. Tokenization (512-token truncation)
# ────────────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
MAX_LENGTH = 512

def tokenize_fn(ex):
    return tokenizer(ex["text"], truncation=True, max_length=MAX_LENGTH)

ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

# ────────────────────────────────────────────────────────────────────────
# 6. Load model
#    – If GPU/CUDA: 8-bit quant + k-bit prep
#    – Else: full-precision with CPU offload
# ────────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    print("→ CUDA available: loading in 8-bit quantized mode")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id
    )
    model = prepare_model_for_kbit_training(model)
else:
    print("→ No CUDA: loading full precision with CPU offloading")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto",
        offload_folder="offload",
        offload_state_dict=True,
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id
    )

# ────────────────────────────────────────────────────────────────────────
# 7. Apply LoRA (PEFT)
# ────────────────────────────────────────────────────────────────────────
lora_cfg = LoraConfig(
    task_type="SEQ_CLS",
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, lora_cfg)

# ────────────────────────────────────────────────────────────────────────
# 8. Metrics & Trainer
# ────────────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f}

training_args = TrainingArguments(
    output_dir="./ft-biomistral-cupcase",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    evaluation_strategy="steps",
    eval_steps=200,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    num_train_epochs=3,
    learning_rate=5e-4,
    fp16=torch.cuda.is_available(),
    push_to_hub=True,
    hub_model_id=PUSH_REPO_ID,
    hub_token=HF_API_TOKEN
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
# ────────────────────────────────────────────────────────────────────────
# 9. Fine-tune & push LoRA adapters
# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("▶ Starting LoRA fine-tuning BioMistral-7B on CUPCase…")
    trainer.train()
    print("▶ Pushing LoRA adapters + config to the Hub…")
    trainer.push_to_hub()
    print(f"🎉 LoRA-fine-tuned model available at https://huggingface.co/{PUSH_REPO_ID}")
