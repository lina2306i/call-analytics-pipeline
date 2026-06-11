"""
generate_synthetic_calls.py
────────────────────────────────────────────────────────────
Generates synthetic call-center audio recordings using gTTS.
Produces both .mp3 audio files AND a ground_truth.json file
containing the expected transcript, intent, and compliance flags
for each call — so we can measure pipeline accuracy later.

Usage:
    python generate_synthetic_calls.py
    python generate_synthetic_calls.py --n 50 --out data/raw/synth
"""

import os
import re
import sys
import json
import time
import argparse
from pathlib import Path

# ── dependency check ──────────────────────────────────────────────────────────
try:
    from gtts import gTTS
except ImportError:
    sys.exit("gTTS not found. Run: pip install gTTS")

sys.path.insert(0, str(Path(__file__).parent))
from pipeline.utils import load_config, get_logger

# ═══════════════════════════════════════════════════════════════════════════════
#  DIALOGUE TEMPLATES
#  Each scenario has: dialogue turns, intent label, expected compliance flags
# ═══════════════════════════════════════════════════════════════════════════════

SCENARIOS = {

    "billing_dispute": {
        "intent": "billing_dispute",
        "compliance_issues": ["MISSING_DISCLOSURE"],  # no recording notice
        "dialogue": (
            "Agent: Thank you for calling customer support, how can I help you today? "
            "Customer: Hi, I need to talk about my bill. I was charged twice for the same month "
            "and I need this fixed immediately. "
            "Agent: I understand your frustration. Can I have your account number please? "
            "Customer: Yes, it is 4 8 2 1 9. "
            "Agent: Thank you. I can see two charges on the fifteenth and the sixteenth. "
            "One of them looks like a duplicate. I will raise a refund request for you right now. "
            "Customer: How long will that take? "
            "Agent: Within five to seven business days the amount will be back in your account. "
            "Customer: Alright, thank you."
        ),
    },

    "billing_dispute_compliant": {
        "intent": "billing_dispute",
        "compliance_issues": [],
        "dialogue": (
            "Agent: Thank you for calling support. Please note this call may be recorded "
            "for quality and training purposes. How can I help you today? "
            "Customer: Hello. I have a billing problem. There is a charge I do not recognize "
            "on my account from last week. "
            "Agent: I am sorry to hear that. Let me verify your identity first. "
            "Can you confirm your full name and the last four digits of your card? "
            "Customer: My name is John Smith and the last four digits are seven seven three two. "
            "Agent: Thank you John. I can see an unauthorized charge of forty nine dollars and ninety nine cents. "
            "I will escalate this to our billing team for investigation. "
            "Customer: Great, thank you for your help."
        ),
    },

    "refund_request": {
        "intent": "refund_request",
        "compliance_issues": ["MISSING_DISCLOSURE", "PROHIBITED_LANGUAGE"],
        "dialogue": (
            "Agent: Hello how can I assist you? "
            "Customer: I want a refund for the product I bought last week. It stopped working after two days. "
            "Agent: I completely understand. I can guarantee you will get your money back right away. "
            "Customer: How soon? "
            "Agent: I promise it will be processed today. Just give me your order number. "
            "Customer: It is order number 9 9 3 4 5 7. "
            "Agent: Perfect, I have submitted the refund. "
            "Customer: Thank you."
        ),
    },

    "refund_request_compliant": {
        "intent": "refund_request",
        "compliance_issues": [],
        "dialogue": (
            "Agent: Thank you for calling. This call may be recorded for quality purposes. "
            "How can I help you today? "
            "Customer: I purchased a product online five days ago and it is defective. I want a full refund. "
            "Agent: I am sorry about that experience. Let me verify your identity. "
            "Can you provide your email address and order number? "
            "Customer: Sure. My email is john at example dot com and the order is nine nine three four five seven. "
            "Agent: Thank you. Your purchase qualifies for our thirty day return policy. "
            "I have initiated a full refund of sixty five dollars. "
            "You will receive a confirmation email within twenty four hours. "
            "Customer: Perfect, I appreciate that."
        ),
    },

    "tech_support": {
        "intent": "technical_support",
        "compliance_issues": ["MISSING_DISCLOSURE"],
        "dialogue": (
            "Agent: Support line, how can I help? "
            "Customer: My internet connection keeps dropping every few minutes. It has been happening since yesterday. "
            "Agent: I am sorry to hear that. Let me run a remote diagnostic on your router. "
            "Can you tell me the model number on the back of your device? "
            "Customer: It says model A C eighteen hundred. "
            "Agent: Great. I am sending a reset signal to your device now. "
            "Can you unplug the router for thirty seconds and plug it back in? "
            "Customer: Okay, done. It seems more stable now. "
            "Agent: The diagnostic shows your connection is stable. "
            "If the issue returns within forty eight hours please call back and we will escalate to a technician. "
            "Customer: Sounds good, thank you."
        ),
    },

    "tech_support_escalation": {
        "intent": "escalation_request",
        "compliance_issues": ["MISSING_DISCLOSURE", "PROHIBITED_LANGUAGE"],
        "dialogue": (
            "Agent: Thank you for calling, how can I help? "
            "Customer: I have called three times about the same problem and nobody has fixed it. "
            "I want to speak to a manager right now. "
            "Agent: I understand you are frustrated. Let me see what I can do. "
            "Customer: I have been a customer for ten years and this is unacceptable. "
            "If this is not resolved today I will take legal action. "
            "Agent: Sir please calm down. I will escalate this to a senior agent immediately. "
            "Can I put you on a brief hold? "
            "Customer: Fine. But I want this resolved today. "
            "Agent: Absolutely, I understand. Let me transfer you right now."
        ),
    },

    "cancellation": {
        "intent": "account_cancellation",
        "compliance_issues": ["MISSING_DISCLOSURE"],
        "dialogue": (
            "Agent: Customer service, how can I help you? "
            "Customer: I would like to cancel my subscription please. "
            "Agent: I am sorry to hear you want to leave. Can I ask what the reason is? "
            "Customer: I am moving abroad and will not need the service anymore. "
            "Agent: I understand. Before I cancel, I should let you know we have an international plan "
            "that might work for you. Would you like to hear more? "
            "Customer: No thank you, I just want to cancel. "
            "Agent: Of course. Let me verify your identity first. "
            "What is the email address on the account? "
            "Customer: It is sarah at mymail dot com. "
            "Agent: Thank you Sarah. I have processed the cancellation. "
            "Your service will remain active until the end of the billing cycle. "
            "Customer: Thank you, goodbye."
        ),
    },

    "payment_issue": {
        "intent": "payment_issue",
        "compliance_issues": ["MISSING_DISCLOSURE", "PII_DETECTED"],
        "dialogue": (
            "Agent: Hello, how can I assist you today? "
            "Customer: My payment is not going through. I tried three times and it keeps declining. "
            "Agent: I can help with that. Can you read me your card number? "
            "Customer: Sure, it is four one one one two two two two three three three three four four four four. "
            "Agent: Thank you. The expiry date? "
            "Customer: Zero nine twenty six. "
            "Agent: I can see the issue. Your card was flagged for a security check. "
            "I will clear that flag now. Can you try the payment again in a few minutes? "
            "Customer: Okay I will try again. Thank you."
        ),
    },

    "payment_issue_compliant": {
        "intent": "payment_issue",
        "compliance_issues": [],
        "dialogue": (
            "Agent: Thank you for calling. This call may be recorded for quality and training purposes. "
            "How can I help you today? "
            "Customer: My payment keeps failing. I need to update my billing details. "
            "Agent: I can help with that. To keep your information secure, "
            "I will send you a secure link to update your card details. "
            "Can I confirm your email address? "
            "Customer: Yes, it is mike at example dot com. "
            "Agent: Thank you. I have sent the secure link. Please do not share "
            "your card number over the phone. The link will expire in fifteen minutes. "
            "Customer: Got it, I see the email. I will do it now. "
            "Agent: Great. Is there anything else I can help you with today? "
            "Customer: No, that is all. Thanks."
        ),
    },

    "general_inquiry": {
        "intent": "general_inquiry",
        "compliance_issues": [],
        "dialogue": (
            "Agent: Thank you for calling. This call may be recorded. How can I help you today? "
            "Customer: I just wanted to know what plans you currently offer. "
            "Agent: Of course. We have three main plans. "
            "The basic plan is nine ninety nine per month and includes ten gigabytes of storage. "
            "The standard plan is nineteen ninety nine and includes fifty gigabytes. "
            "The premium plan is thirty four ninety nine and gives you unlimited storage. "
            "Customer: What is the difference between standard and premium besides storage? "
            "Agent: Premium also includes priority support and access to our mobile app. "
            "Customer: Great, I think I will go with standard for now. "
            "Agent: Excellent choice. Shall I sign you up right now? "
            "Customer: Yes please."
        ),
    },

    "complaint": {
        "intent": "complaint",
        "compliance_issues": ["MISSING_DISCLOSURE"],
        "dialogue": (
            "Agent: Customer support, how can I help? "
            "Customer: I want to make a formal complaint. Your agent last week was extremely rude to me. "
            "Agent: I sincerely apologize for that experience. "
            "Can you tell me more about what happened? "
            "Customer: I called about a billing issue and the agent spoke over me and dismissed my concerns. "
            "I have been a loyal customer for five years and I felt completely disrespected. "
            "Agent: You are absolutely right to raise this. I will file a formal complaint on your behalf. "
            "A supervisor will contact you within twenty four hours. "
            "Can I have your name and account number? "
            "Customer: My name is Maria Garcia and my account is seven seven four two one. "
            "Agent: Thank you Maria. I have logged the complaint. "
            "We take feedback very seriously and this will be investigated. "
            "Customer: Thank you, I hope it gets resolved."
        ),
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
#  GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def clean_text_for_tts(text: str) -> str:
    """Remove TTS-unfriendly characters."""
    text = re.sub(r'[^\w\s\.,!?\'\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def generate_call(
    scenario_name: str,
    scenario: dict,
    output_dir: Path,
    call_index: int,
    lang: str = "en",
    slow: bool = False,
    logger=None,
) -> dict:
    """
    Generate a single synthetic call audio file + its ground truth record.
    Returns the ground truth dict.
    """
    call_id  = f"SYNTH-{call_index:04d}-{scenario_name.upper()}"
    filename = f"{call_id}.mp3"
    out_path = output_dir / filename

    dialogue  = clean_text_for_tts(scenario["dialogue"])
    log = logger.info if logger else print

    try:
        tts = gTTS(text=dialogue, lang=lang, slow=slow)
        tts.save(str(out_path))
        log(f"  ✓ {filename}")
    except Exception as e:
        if logger:
            logger.error(f"  ✗ {filename} — {e}")
        else:
            print(f"  ✗ {filename} — {e}")
        return {}

    # Build ground truth record
    ground_truth = {
        "call_id":            call_id,
        "audio_file":         str(out_path),
        "scenario":           scenario_name,
        "expected_intent":    scenario["intent"],
        "expected_flags":     scenario["compliance_issues"],
        "transcript_ref":     scenario["dialogue"],   # reference for WER calc
        "generated_at":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    return ground_truth


def generate_all(output_dir: str, n_repeats: int = 2, lang: str = "en", slow: bool = False, logger=None):
    """
    Generate all scenarios × n_repeats calls.
    Saves audio to output_dir and writes ground_truth.json alongside.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    log = logger.info if logger else print
    log(f"Generating synthetic calls → {out}")
    log(f"Scenarios: {len(SCENARIOS)} | Repeats each: {n_repeats}")

    all_ground_truth = []
    call_idx = 1

    for repeat in range(n_repeats):
        for scenario_name, scenario in SCENARIOS.items():
            gt = generate_call(
                scenario_name=scenario_name,
                scenario=scenario,
                output_dir=out,
                call_index=call_idx,
                lang=lang,
                slow=slow,
                logger=logger,
            )
            if gt:
                all_ground_truth.append(gt)
            call_idx += 1

            # Small delay to avoid gTTS rate limiting
            time.sleep(0.4)

    # Save ground truth
    gt_path = out / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(all_ground_truth, f, indent=2)

    log(f"\nDone. {len(all_ground_truth)} calls generated.")
    log(f"Ground truth saved → {gt_path}")
    return all_ground_truth


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic call recordings")
    parser.add_argument("--out",     default="data/raw/synth", help="Output directory")
    parser.add_argument("--repeats", type=int, default=2,      help="How many times to repeat each scenario")
    parser.add_argument("--lang",    default="en",             help="gTTS language code")
    parser.add_argument("--slow",    action="store_true",      help="Slow TTS speed")
    args = parser.parse_args()

    try:
        cfg    = load_config()
        logger = get_logger("synthetic_generator", cfg)
    except Exception:
        logger = None

    generate_all(
        output_dir=args.out,
        n_repeats=args.repeats,
        lang=args.lang,
        slow=args.slow,
        logger=logger,
    )
