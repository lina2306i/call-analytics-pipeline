"""
prepare_intent_data.py
────────────────────────────────────────────────────────────
Downloads SLURP + MINDS-14 from HuggingFace, maps their intent
labels to our 9 pipeline categories, and saves clean
train / val / test CSV files ready for fine-tuning.

Usage:
    python prepare_intent_data.py
    python prepare_intent_data.py --output data/intent_training
"""

import re
import json
import argparse
from pathlib import Path

# ── dependency check ─────────────────────────────────────────────────────────
try:
    import pandas as pd
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split
except ImportError:
    raise SystemExit(
        "Missing dependencies. Run:\n"
        "  pip install datasets scikit-learn pandas"
    )

# ═══════════════════════════════════════════════════════════════════════════════
#  OUR 9 TARGET LABELS
# ═══════════════════════════════════════════════════════════════════════════════

TARGET_LABELS = [
    "billing_dispute",
    "refund_request",
    "technical_support",
    "account_cancellation",
    "payment_issue",
    "product_inquiry",
    "escalation_request",
    "complaint",
    "general_inquiry",
]

LABEL2ID = {lbl: i for i, lbl in enumerate(TARGET_LABELS)}
ID2LABEL = {i: lbl for i, lbl in enumerate(TARGET_LABELS)}

# ═══════════════════════════════════════════════════════════════════════════════
#  SLURP LABEL MAPPING  →  our 9 categories
#  SLURP has 18 "scenario" fields — we map them to our labels
# ═══════════════════════════════════════════════════════════════════════════════

SLURP_SCENARIO_MAP = {
    # billing / payment
    "banking":          "billing_dispute",
    "finance":          "billing_dispute",
    "payment":          "payment_issue",
    "credit_cards":     "payment_issue",
    # refund / returns
    "shopping":         "refund_request",
    "ecommerce":        "refund_request",
    # tech support
    "iot":              "technical_support",
    "smart_home":       "technical_support",
    "internet":         "technical_support",
    "computer":         "technical_support",
    "phone":            "technical_support",
    "email":            "technical_support",
    # cancellation
    "subscription":     "account_cancellation",
    "streaming":        "account_cancellation",
    # product inquiry
    "recommendation":   "product_inquiry",
    "lists":            "product_inquiry",
    "qa":               "product_inquiry",
    # general
    "general":          "general_inquiry",
    "datetime":         "general_inquiry",
    "weather":          "general_inquiry",
    "news":             "general_inquiry",
    "music":            "general_inquiry",
    "audio":            "general_inquiry",
    "play":             "general_inquiry",
    "social":           "general_inquiry",
    "transport":        "general_inquiry",
    "travel":           "general_inquiry",
    "calendar":         "general_inquiry",
    "alarm":            "general_inquiry",
}

# ═══════════════════════════════════════════════════════════════════════════════
#  MINDS-14 LABEL MAPPING  →  our 9 categories
#  MINDS-14 has 14 intent labels
# ═══════════════════════════════════════════════════════════════════════════════

MINDS14_INTENT_MAP = {
    "abroad":                      "general_inquiry",
    "address":                     "general_inquiry",
    "app_error":                   "technical_support",
    "atm_limit":                   "billing_dispute",
    "balance":                     "billing_dispute",
    "business_loan":               "product_inquiry",
    "card_issues":                 "payment_issue",
    "cash_deposit":                "payment_issue",
    "direct_debit":                "billing_dispute",
    "freeze_account":              "account_cancellation",
    "high_value_payment":          "payment_issue",
    "insurance":                   "product_inquiry",
    "insurance_car":               "product_inquiry",
    "interest_rate":               "product_inquiry",
    "internet_banking":            "technical_support",
    "joint_account":               "general_inquiry",
    "latest_transactions":         "billing_dispute",
    "lost_or_stolen":              "complaint",
    "make_purchase":               "payment_issue",
    "new_card":                    "product_inquiry",
    "online_banking":              "technical_support",
    "order":                       "refund_request",
    "pay_bill":                    "payment_issue",
    "payment":                     "payment_issue",
    "pin_change":                  "technical_support",
    "refund":                      "refund_request",
    "report_fraud":                "complaint",
    "report_lost_card":            "complaint",
    "routing":                     "general_inquiry",
    "savings":                     "product_inquiry",
    "spending_limit":              "billing_dispute",
    "transfer":                    "payment_issue",
    "visa":                        "general_inquiry",
    "withdraw":                    "payment_issue",
}

# ═══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC CALL TEMPLATES  (from our own scenarios — 100% label accuracy)
# ═══════════════════════════════════════════════════════════════════════════════

SYNTHETIC_EXAMPLES = {
    "billing_dispute": [
        "I was charged twice on my account this month.",
        "There's a duplicate charge on my bill I need resolved.",
        "My invoice shows an amount I don't recognize.",
        "I noticed an incorrect charge on my statement.",
        "I'm being billed for something I didn't purchase.",
        "My account balance doesn't match what I was told.",
        "I have a billing discrepancy I need you to look at.",
        "Why was my account debited twice for the same transaction?",
        "I need to dispute a charge from last month.",
        "The amount on my bill is wrong.",
    ],
    "refund_request": [
        "I want a full refund for the product I returned.",
        "Can I get my money back for this defective item?",
        "I'd like to request a refund for my last order.",
        "The product stopped working after two days, I want a refund.",
        "I returned the item two weeks ago and haven't received my refund.",
        "Please process my refund as soon as possible.",
        "I need a reimbursement for the service I cancelled.",
        "How do I get a refund for an online purchase?",
        "I was overcharged and I want the difference refunded.",
        "The item never arrived and I want my money back.",
    ],
    "technical_support": [
        "My internet connection keeps dropping every few minutes.",
        "The app is not working on my phone.",
        "I can't log in to my online account.",
        "My device is showing an error message.",
        "The website is not loading for me.",
        "I'm having trouble accessing my account online.",
        "My router keeps disconnecting from the network.",
        "The software update broke my settings.",
        "I need help setting up my new device.",
        "The service is down and I can't access anything.",
    ],
    "account_cancellation": [
        "I want to cancel my subscription.",
        "Please close my account immediately.",
        "I'd like to terminate my service.",
        "How do I cancel my membership?",
        "I no longer need the service and want to cancel.",
        "I'm moving abroad and need to close my account.",
        "Please cancel my plan at the end of the month.",
        "I want to stop my automatic renewal.",
        "I need to cancel before the next billing cycle.",
        "How do I delete my account completely?",
    ],
    "payment_issue": [
        "My payment is not going through.",
        "My card keeps getting declined.",
        "I'm having trouble making a payment.",
        "The payment failed even though my card is valid.",
        "I can't complete my transaction on the website.",
        "My payment was declined three times in a row.",
        "Why is my card being rejected?",
        "I need help with a failed payment.",
        "The checkout page shows a payment error.",
        "My bank transfer didn't go through.",
    ],
    "product_inquiry": [
        "What plans do you currently offer?",
        "Can you tell me more about your premium package?",
        "What's the difference between the basic and standard plan?",
        "Do you offer a student discount?",
        "I'd like to know about your business solutions.",
        "What are the features of your new product?",
        "How much does the annual subscription cost?",
        "Is there a free trial available?",
        "What's included in the enterprise plan?",
        "Can I upgrade my current plan?",
    ],
    "escalation_request": [
        "I want to speak to a manager right now.",
        "I've called three times and nobody has fixed this.",
        "This is unacceptable, I need to escalate this issue.",
        "Please transfer me to a supervisor.",
        "I'm not satisfied with this response, I want a manager.",
        "I've been waiting for a resolution for two weeks.",
        "This needs to go to a higher level of support.",
        "I want to make a formal complaint to management.",
        "Your agent couldn't help me, I need someone senior.",
        "I demand to speak with someone who can actually fix this.",
    ],
    "complaint": [
        "I want to make a formal complaint about your service.",
        "Your agent was extremely rude to me.",
        "I've had a terrible experience with your company.",
        "I'm very disappointed with how this was handled.",
        "I need to file a complaint about what happened.",
        "The service I received was completely unacceptable.",
        "I have been a loyal customer and this is how you treat me?",
        "I want to report the behavior of one of your employees.",
        "This is not the first time I've had problems with you.",
        "I expect better service than what I received today.",
    ],
    "general_inquiry": [
        "I just wanted to check on my account status.",
        "Can you tell me your business hours?",
        "I have a general question about my account.",
        "How do I update my contact information?",
        "What documents do I need to provide?",
        "I'd like to know more about how your service works.",
        "Can you send me a copy of my last statement?",
        "How long does processing usually take?",
        "I need help understanding my account details.",
        "What is your policy on late payments?",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_slurp(max_per_label: int = 300) -> list:
    """
    Load SLURP dataset and map to our 9 labels.
    Returns list of {"text": str, "label": str, "source": "slurp"}.
    """
    print("Downloading SLURP...")
    try:
        ds = load_dataset("slurp", trust_remote_code=True)
    except Exception as e:
        print(f"  SLURP load failed: {e}")
        return []

    records = []
    counts  = {lbl: 0 for lbl in TARGET_LABELS}

    for split in ["train", "validation"]:
        if split not in ds:
            continue
        for row in ds[split]:
            scenario = row.get("scenario", "").lower().strip()
            our_label = SLURP_SCENARIO_MAP.get(scenario)
            if our_label is None:
                continue
            if counts[our_label] >= max_per_label:
                continue
            text = row.get("sentence", "").strip()
            if len(text) < 5:
                continue
            records.append({"text": text, "label": our_label, "source": "slurp"})
            counts[our_label] += 1

    print(f"  SLURP: {len(records)} records loaded")
    for lbl, cnt in counts.items():
        print(f"    {lbl:25} {cnt}")
    return records


def load_minds14(max_per_label: int = 100) -> list:
    """
    Load MINDS-14 (en-US) and map to our 9 labels.
    """
    print("Downloading MINDS-14...")
    try:
        ds = load_dataset("PolyAI/minds14", name="en-US", trust_remote_code=True)
    except Exception as e:
        print(f"  MINDS-14 load failed: {e}")
        return []

    # Get intent label names
    intent_feature = ds["train"].features.get("intent_class")
    if intent_feature is None:
        print("  MINDS-14: intent_class field not found")
        return []

    intent_names = intent_feature.names  # list of label strings

    records = []
    counts  = {lbl: 0 for lbl in TARGET_LABELS}

    for split in ["train", "validation"] if "validation" in ds else ["train"]:
        if split not in ds:
            continue
        for row in ds[split]:
            intent_id   = row.get("intent_class", -1)
            intent_name = intent_names[intent_id].lower().strip() if intent_id >= 0 else ""
            our_label   = MINDS14_INTENT_MAP.get(intent_name)
            if our_label is None:
                continue
            if counts[our_label] >= max_per_label:
                continue
            # MINDS-14 has transcription field
            text = row.get("transcription", "").strip()
            if len(text) < 5:
                continue
            records.append({"text": text, "label": our_label, "source": "minds14"})
            counts[our_label] += 1

    print(f"  MINDS-14: {len(records)} records loaded")
    return records


def load_synthetic() -> list:
    """Load hand-crafted synthetic examples — 100% accurate labels."""
    records = []
    for label, texts in SYNTHETIC_EXAMPLES.items():
        for text in texts:
            records.append({"text": text, "label": label, "source": "synthetic"})
    print(f"  Synthetic: {len(records)} records")
    return records


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def build_dataset(output_dir: str = "data/intent_training", max_per_label: int = 300):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load all sources
    all_records = []
    all_records += load_synthetic()          # always include (high quality)
    all_records += load_slurp(max_per_label)
    all_records += load_minds14(100)

    df = pd.DataFrame(all_records)
    df = df.drop_duplicates(subset=["text"])
    df["label_id"] = df["label"].map(LABEL2ID)
    df = df.dropna(subset=["label_id"])
    df["label_id"] = df["label_id"].astype(int)

    print(f"\nTotal unique records: {len(df)}")
    print("\nLabel distribution:")
    print(df["label"].value_counts().to_string())
    print(f"\nSources: {df['source'].value_counts().to_dict()}")

    # Split: 70% train / 15% val / 15% test  (stratified)
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42
    )

    # Save splits
    train_df.to_csv(out / "train.csv", index=False)
    val_df.to_csv(out  / "val.csv",   index=False)
    test_df.to_csv(out / "test.csv",  index=False)

    # Save label map
    with open(out / "label_map.json", "w") as f:
        json.dump({"label2id": LABEL2ID, "id2label": ID2LABEL}, f, indent=2)

    print(f"\nSplit sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"Saved to: {out}/")
    print("  train.csv, val.csv, test.csv, label_map.json")
    return str(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",        default="data/intent_training")
    parser.add_argument("--max_per_label", type=int, default=300)
    args = parser.parse_args()
    build_dataset(args.output, args.max_per_label)
