"""
train_intent_classifier.py
────────────────────────────────────────────────────────────
Fine-tunes DistilBERT on our 9 call-center intent labels.
Uses HuggingFace Trainer — GPU-accelerated, ~10 min on your RTX.

Usage:
    python train_intent_classifier.py
    python train_intent_classifier.py --data data/intent_training --epochs 5
    python train_intent_classifier.py --test   # quick 2-epoch smoke test

Output:
    models/intent_classifier/          ← saved model (use in pipeline)
    models/intent_classifier/metrics.json
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path

try:
    import torch
    import pandas as pd
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
    )
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        classification_report,
        confusion_matrix,
    )
except ImportError as e:
    raise SystemExit(f"Missing dependency: {e}\nRun: pip install transformers datasets scikit-learn")

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

BASE_MODEL   = "distilbert-base-uncased"
MAX_LENGTH   = 128          # max tokens per utterance
BATCH_SIZE   = 32           # fits on 4GB+ GPU
LEARNING_RATE = 2e-5
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.1


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════════════════════

def load_splits(data_dir: str):
    """Load train/val/test CSV splits and label map."""
    data_path = Path(data_dir)

    train_df = pd.read_csv(data_path / "train.csv")
    val_df   = pd.read_csv(data_path / "val.csv")
    test_df  = pd.read_csv(data_path / "test.csv")

    with open(data_path / "label_map.json") as f:
        label_map = json.load(f)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Labels: {list(label_map['label2id'].keys())}")
    return train_df, val_df, test_df, label_map


def tokenize_dataset(df: pd.DataFrame, tokenizer, max_length: int) -> Dataset:
    """Convert a DataFrame to a HuggingFace Dataset with tokenized inputs."""
    dataset = Dataset.from_pandas(df[["text", "label_id"]].rename(
        columns={"label_id": "labels"}
    ))

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    dataset = dataset.map(tokenize, batched=True, batch_size=256)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset


# ═══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(eval_pred):
    """Called by Trainer after each eval epoch."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train(
    data_dir:   str  = "data/intent_training",
    output_dir: str  = "models/intent_classifier",
    epochs:     int  = 4,
    test_mode:  bool = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device.upper()}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    train_df, val_df, test_df, label_map = load_splits(data_dir)
    label2id = label_map["label2id"]
    id2label = {int(k): v for k, v in label_map["id2label"].items()}
    num_labels = len(label2id)

    # Quick smoke test
    if test_mode:
        print("\n[TEST MODE] Using 100 samples, 2 epochs")
        train_df = train_df.head(100)
        val_df   = val_df.head(50)
        epochs   = 2

    # Load tokenizer + model
    print(f"\nLoading {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Tokenize
    print("Tokenizing datasets...")
    train_dataset = tokenize_dataset(train_df, tokenizer, MAX_LENGTH)
    val_dataset   = tokenize_dataset(val_df,   tokenizer, MAX_LENGTH)
    test_dataset  = tokenize_dataset(test_df,  tokenizer, MAX_LENGTH)

    # Training arguments
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        # Evaluation & saving
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        # Logging
        logging_dir=str(out / "logs"),
        logging_steps=50,
        report_to="none",          # disable wandb
        # Performance
        fp16=torch.cuda.is_available(),   # use FP16 on GPU
        dataloader_num_workers=0,          # 0 = safe on Windows
        # Save disk space
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nTraining for up to {epochs} epochs...")
    train_result = trainer.train()

    print(f"\nTraining complete!")
    print(f"  Loss: {train_result.training_loss:.4f}")
    print(f"  Time: {train_result.metrics.get('train_runtime', 0):.0f}s")

    # ── Evaluate on test set ──────────────────────────────────────────────────
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(test_dataset)

    print(f"\n── Test Results ────────────────────────")
    print(f"  Accuracy  : {test_results.get('eval_accuracy', 0):.4f}")
    print(f"  F1 Macro  : {test_results.get('eval_f1_macro', 0):.4f}")
    print(f"  F1 Weighted: {test_results.get('eval_f1_weighted', 0):.4f}")

    # Full classification report
    preds_output = trainer.predict(test_dataset)
    preds        = np.argmax(preds_output.predictions, axis=-1)
    true_labels  = preds_output.label_ids

    label_names = [id2label[i] for i in range(num_labels)]
    report = classification_report(
        true_labels, preds,
        target_names=label_names,
        zero_division=0,
    )
    print(f"\n── Per-class Report ────────────────────\n{report}")

    # Confusion matrix
    cm = confusion_matrix(true_labels, preds)
    print(f"\n── Confusion Matrix ────────────────────")
    print(f"{'':25}", end="")
    for lbl in label_names:
        print(f"{lbl[:8]:>10}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"{label_names[i]:25}", end="")
        for val in row:
            print(f"{val:>10}", end="")
        print()

    # ── Save model ────────────────────────────────────────────────────────────
    print(f"\nSaving model to {out}/")
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))

    # Save metrics
    metrics = {
        "test_accuracy":    test_results.get("eval_accuracy", 0),
        "test_f1_macro":    test_results.get("eval_f1_macro", 0),
        "test_f1_weighted": test_results.get("eval_f1_weighted", 0),
        "train_loss":       train_result.training_loss,
        "epochs":           epochs,
        "base_model":       BASE_MODEL,
        "num_labels":       num_labels,
        "train_samples":    len(train_df),
        "label_names":      label_names,
        "classification_report": report,
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nAll outputs saved to: {out}/")
    print("  model files, tokenizer, metrics.json")
    print("\nTo use in pipeline, update config.yaml:")
    print(f'  nlp.intent_model: "{out}"')
    return str(out)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune intent classifier")
    parser.add_argument("--data",    default="data/intent_training",
                        help="Directory with train/val/test CSV splits")
    parser.add_argument("--output",  default="models/intent_classifier",
                        help="Where to save the fine-tuned model")
    parser.add_argument("--epochs",  type=int, default=4,
                        help="Max training epochs (early stopping applies)")
    parser.add_argument("--test",    action="store_true",
                        help="Quick smoke test with 100 samples, 2 epochs")
    args = parser.parse_args()

    train(
        data_dir=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        test_mode=args.test,
    )
