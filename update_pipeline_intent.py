"""
update_pipeline_intent.py
────────────────────────────────────────────────────────────
After training, this script:
  1. Tests the fine-tuned model on sample utterances
  2. Updates config.yaml to use the new model
  3. Re-runs the pipeline on synth data to compare before/after

Usage:
    python update_pipeline_intent.py
    python update_pipeline_intent.py --model models/intent_classifier --test-only
"""

import json
import argparse
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
#  QUICK INFERENCE TEST
# ═══════════════════════════════════════════════════════════════════════════════

TEST_UTTERANCES = [
    ("I was charged twice this month",                   "billing_dispute"),
    ("I want a refund for my broken product",            "refund_request"),
    ("My internet keeps disconnecting",                  "technical_support"),
    ("Please cancel my subscription",                    "account_cancellation"),
    ("My card payment is being declined",                "payment_issue"),
    ("What plans do you offer?",                         "product_inquiry"),
    ("I need to speak to a manager",                     "escalation_request"),
    ("Your agent was very rude to me",                   "complaint"),
    ("I just wanted to check my account balance",        "general_inquiry"),
]


def test_model(model_path: str) -> dict:
    """
    Run the fine-tuned model on test utterances.
    Returns accuracy dict.
    """
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        raise SystemExit("transformers not installed")

    print(f"\nLoading model from: {model_path}")
    classifier = hf_pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path,
        device=0 if _has_gpu() else -1,
    )

    print(f"\n{'─'*65}")
    print(f"{'Utterance':40} {'Expected':22} {'Got':22} OK?")
    print(f"{'─'*65}")

    correct = 0
    results = []
    for text, expected in TEST_UTTERANCES:
        out    = classifier(text, truncation=True, max_length=128)[0]
        pred   = out["label"]
        score  = out["score"]
        ok     = "✓" if pred == expected else "✗"
        if pred == expected:
            correct += 1

        short_text = text[:38] + ".." if len(text) > 40 else text
        print(f"{short_text:40} {expected:22} {pred:22} {ok}  ({score:.2f})")
        results.append({
            "text":     text,
            "expected": expected,
            "pred":     pred,
            "score":    score,
            "correct":  pred == expected,
        })

    accuracy = correct / len(TEST_UTTERANCES)
    print(f"{'─'*65}")
    print(f"Accuracy on test utterances: {correct}/{len(TEST_UTTERANCES)} = {accuracy:.0%}")

    # Load saved training metrics if available
    metrics_path = Path(model_path) / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            train_metrics = json.load(f)
        print(f"\nTraining metrics (from metrics.json):")
        print(f"  Test accuracy : {train_metrics.get('test_accuracy', '?'):.4f}")
        print(f"  F1 Macro      : {train_metrics.get('test_f1_macro', '?'):.4f}")
        print(f"  F1 Weighted   : {train_metrics.get('test_f1_weighted', '?'):.4f}")

    return {"accuracy": accuracy, "results": results}


def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  UPDATE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

def update_config(model_path: str, config_path: str = "config.yaml"):
    """
    Update config.yaml to point nlp.intent_model at the fine-tuned model.
    Also lowers intent_threshold since fine-tuned model is more confident.
    """
    try:
        import yaml
    except ImportError:
        print("pyyaml not installed, skipping config update")
        return

    config_file = Path(config_path)
    if not config_file.exists():
        print(f"Config not found: {config_path}")
        return

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Update intent model path and threshold
    cfg["nlp"]["intent_model"]     = str(Path(model_path).resolve())
    cfg["nlp"]["intent_threshold"] = 0.5   # higher threshold for fine-tuned model

    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    print(f"\nconfig.yaml updated:")
    print(f"  nlp.intent_model     → {model_path}")
    print(f"  nlp.intent_threshold → 0.5")


# ═══════════════════════════════════════════════════════════════════════════════
#  BEFORE / AFTER COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def compare_before_after(model_path: str):
    """
    Quick comparison: zero-shot MNLI vs fine-tuned model
    on the same 9 utterances.
    """
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        return

    LABELS = [
        "billing dispute", "refund request", "technical support",
        "account cancellation", "payment issue", "product inquiry",
        "escalation request", "complaint", "general inquiry",
    ]

    print("\n" + "═"*70)
    print("BEFORE vs AFTER COMPARISON")
    print("═"*70)

    # Zero-shot (before)
    print("\n[BEFORE] Zero-shot MNLI (typeform/distilbert-base-uncased-mnli)")
    try:
        zs = hf_pipeline(
            "zero-shot-classification",
            model="typeform/distilbert-base-uncased-mnli",
            device=0 if _has_gpu() else -1,
        )
        zs_correct = 0
        for text, expected in TEST_UTTERANCES[:5]:   # first 5 only (faster)
            out   = zs(text, candidate_labels=LABELS)
            pred  = out["labels"][0].replace(" ", "_")
            ok    = "✓" if pred == expected else "✗"
            if pred == expected:
                zs_correct += 1
            print(f"  {ok} [{pred:25}] {text[:45]}")
        print(f"  → {zs_correct}/5 correct")
    except Exception as e:
        print(f"  Failed to load zero-shot model: {e}")

    # Fine-tuned (after)
    print(f"\n[AFTER] Fine-tuned DistilBERT ({model_path})")
    try:
        ft = hf_pipeline(
            "text-classification",
            model=model_path,
            tokenizer=model_path,
            device=0 if _has_gpu() else -1,
        )
        ft_correct = 0
        for text, expected in TEST_UTTERANCES[:5]:
            out   = ft(text, truncation=True, max_length=128)[0]
            pred  = out["label"]
            ok    = "✓" if pred == expected else "✗"
            if pred == expected:
                ft_correct += 1
            print(f"  {ok} [{pred:25}] {text[:45]}")
        print(f"  → {ft_correct}/5 correct")
    except Exception as e:
        print(f"  Failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default="models/intent_classifier",
                        help="Path to fine-tuned model directory")
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--test-only", action="store_true",
                        help="Only test the model, don't update config")
    parser.add_argument("--compare",   action="store_true",
                        help="Show before/after comparison with zero-shot")
    args = parser.parse_args()

    model_path = args.model

    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        print("Run train_intent_classifier.py first.")
        return

    # Test the fine-tuned model
    results = test_model(model_path)

    # Before/after comparison
    if args.compare:
        compare_before_after(model_path)

    # Update config (unless --test-only)
    if not args.test_only:
        update_config(model_path, args.config)
        print("\nDone. Re-run the pipeline to see improved intent detection:")
        print("  python pipeline.py --input data/raw/synth/ --no-diarize")

    return results


if __name__ == "__main__":
    main()
