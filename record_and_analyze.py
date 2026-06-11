"""
record_and_analyze.py  —  Simulate Call + Instant Analysis (CLI)
────────────────────────────────────────────────────────────
Generates a synthetic call audio on the fly using gTTS,
then immediately runs the full pipeline on it.

This is the CLI version of the "Simulate Call" tab in dashboard_v2.py.

Usage:
    python record_and_analyze.py
    python record_and_analyze.py --scenario billing_dispute
    python record_and_analyze.py --text "I want to cancel my subscription"
    python record_and_analyze.py --list-scenarios
"""

import re
import sys
import json
import time
import tempfile
import argparse
from pathlib import Path

# ── Scenarios ─────────────────────────────────────────────────────────────────

SCENARIOS = {
    "billing_dispute": {
        "intent":  "billing_dispute",
        "flags":   ["MISSING_DISCLOSURE"],
        "dialogue": (
            "Agent: Thank you for calling, this call may be recorded. How can I help? "
            "Customer: I have a double charge on my account from last week. "
            "Agent: I can see that. Let me verify your identity first. "
            "Customer: My name is Sarah and my account ends in four eight two one. "
            "Agent: Thank you Sarah. I confirm the duplicate and will process a refund."
        ),
    },
    "refund_request": {
        "intent":  "refund_request",
        "flags":   ["MISSING_DISCLOSURE", "PROHIBITED_LANGUAGE"],
        "dialogue": (
            "Agent: Hello, how can I help? "
            "Customer: I want a refund for the product I bought last week. It's broken. "
            "Agent: I guarantee you will get your money back right away. "
            "Customer: How soon? Agent: I promise it will be today. "
            "Customer: Okay thank you. My order is nine nine three four five seven."
        ),
    },
    "tech_support": {
        "intent":  "technical_support",
        "flags":   ["MISSING_DISCLOSURE"],
        "dialogue": (
            "Agent: Support line, how can I help? "
            "Customer: My internet keeps dropping every few minutes since yesterday. "
            "Agent: I am sorry about that. Let me run a remote diagnostic on your router. "
            "Customer: Okay it looks more stable now. "
            "Agent: Good. Call back if it drops again within 48 hours."
        ),
    },
    "cancellation": {
        "intent":  "account_cancellation",
        "flags":   ["MISSING_DISCLOSURE"],
        "dialogue": (
            "Agent: Hello, this call may be recorded. How can I help you today? "
            "Customer: I would like to cancel my subscription please. "
            "Agent: I am sorry to hear that. What is the reason? "
            "Customer: I am moving abroad and will not need this service. "
            "Agent: Your account has been cancelled effective end of month."
        ),
    },
    "payment_issue": {
        "intent":  "payment_issue",
        "flags":   [],
        "dialogue": (
            "Agent: Thank you for calling. This call may be recorded. How can I help? "
            "Customer: My payment keeps failing. I have tried three times. "
            "Agent: I understand. To keep things secure I will send a payment link. "
            "Customer: I see the email, I will use it now. "
            "Agent: Great, your payment should process within a few minutes."
        ),
    },
    "complaint": {
        "intent":  "complaint",
        "flags":   ["MISSING_DISCLOSURE"],
        "dialogue": (
            "Agent: Customer support, how can I help? "
            "Customer: I want to make a formal complaint. Your agent last week was rude. "
            "Agent: I sincerely apologize for that experience. "
            "Customer: I called about billing and felt completely dismissed. "
            "Agent: I have filed a formal complaint. A supervisor will call you today."
        ),
    },
    "escalation": {
        "intent":  "escalation_request",
        "flags":   ["MISSING_DISCLOSURE", "PROHIBITED_LANGUAGE"],
        "dialogue": (
            "Agent: Thank you for calling, how can I help? "
            "Customer: I have called three times about the same problem. Nobody fixed it. "
            "I want to speak to a manager right now. "
            "Agent: I understand you are frustrated. Let me escalate this immediately. "
            "Customer: If this is not resolved today I will take legal action. "
            "Agent: I am transferring you to a senior agent now."
        ),
    },
    "general_inquiry": {
        "intent":  "general_inquiry",
        "flags":   [],
        "dialogue": (
            "Agent: Thank you for calling. This call may be recorded. How can I help? "
            "Customer: I just want to know what plans you offer and the prices. "
            "Agent: Of course. We have three plans. Basic at nine ninety nine, "
            "Standard at nineteen ninety nine, and Premium at thirty four ninety nine. "
            "Customer: I will go with Standard. Agent: Excellent, signing you up now."
        ),
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
#  GENERATE AUDIO
# ═══════════════════════════════════════════════════════════════════════════════

def generate_audio(text: str, output_path: str, lang: str = "en") -> str:
    """Use gTTS to synthesize text to an MP3 file."""
    try:
        from gtts import gTTS
    except ImportError:
        raise SystemExit("gTTS not installed. Run: pip install gTTS")

    print(f"Generating audio ({len(text.split())} words)…")
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(output_path)
    print(f"Audio saved: {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
#  RUN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(audio_path: str, model_size: str = "base", agent_id: str = "") -> dict:
    """Run the full pipeline on the generated audio file."""
    try:
        from pipeline.utils import load_config, generate_call_id
        from pipeline import (
            AudioPreprocessor, Transcriber, Diarizer,
            NLPAnalyzer, ComplianceChecker, ReportBuilder
        )
    except ImportError as e:
        raise SystemExit(f"Pipeline import failed: {e}\n"
                         f"Make sure you're running from the project root directory.")

    cfg = load_config("config.yaml")
    cfg["whisper"]["model"]       = model_size
    cfg["diarization"]["enabled"] = False   # no HF token needed

    call_id = generate_call_id(audio_path)
    t0      = time.perf_counter()

    print("\nRunning pipeline…")

    # Stage 1
    print("  [1/5] Preprocessing audio…")
    proc  = AudioPreprocessor(cfg)
    audio = proc.preprocess(audio_path)
    print(f"        Duration: {audio.duration_sec:.1f}s  |  SNR: {audio.snr_estimate:.1f}dB")

    # Stage 2
    print(f"  [2/5] Transcribing with whisper-{model_size}…")
    trans      = Transcriber(cfg)
    transcript = trans.transcribe(audio)
    print(f"        Language: {transcript.language}  |  "
          f"Inference: {transcript.inference_sec:.1f}s")

    # Stage 3
    print("  [3/5] Diarization (fallback mode)…")
    diar     = Diarizer(cfg)
    diarized = diar.diarize(audio_path, transcript, audio.duration_sec)

    # Stage 4
    print("  [4/5] NLP analysis…")
    nlp        = NLPAnalyzer(cfg)
    nlp_result = nlp.analyze(diarized)
    print(f"        Sentiment: {nlp_result.overall_sentiment}  |  "
          f"Intent: {nlp_result.primary_intent}")

    # Stage 5
    print("  [5/5] Compliance check…")
    checker    = ComplianceChecker(cfg)
    compliance = checker.check(nlp_result, transcript.text)

    # Stage 6
    builder = ReportBuilder(cfg)
    record  = builder.build(call_id, audio, transcript,
                            diarized, nlp_result, compliance,
                            agent_id=agent_id)
    builder.export(record)
    elapsed = time.perf_counter() - t0

    return {
        "record":     record,
        "transcript": transcript,
        "elapsed":    elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPUTE WER
# ═══════════════════════════════════════════════════════════════════════════════

def compute_wer(reference: str, hypothesis: str) -> dict:
    try:
        from jiwer import wer, cer
        def norm(t):
            t = t.lower()
            t = re.sub(r"[^\w\s]", " ", t)
            return re.sub(r"\s+", " ", t).strip()
        ref = norm(reference)
        hyp = norm(hypothesis)
        return {"wer": round(wer(ref, hyp), 4), "cer": round(cer(ref, hyp), 4)}
    except ImportError:
        return {"wer": None, "cer": None, "note": "pip install jiwer"}


# ═══════════════════════════════════════════════════════════════════════════════
#  PRINT RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

def print_results(result: dict, reference: str = "", scenario: str = ""):
    record  = result["record"]
    elapsed = result["elapsed"]
    transcript = result["transcript"]

    print(f"\n{'='*60}")
    print(f"  CALL ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"  Call ID    : {record.call_id}")
    print(f"  Duration   : {record.duration_sec:.1f}s")
    print(f"  Model      : {record.whisper_model}")
    print(f"  Pipeline   : {elapsed:.1f}s total")
    print(f"\n  TRANSCRIPT :")
    print(f"  {record.full_transcript[:300]}{'…' if len(record.full_transcript) > 300 else ''}")
    print(f"\n  NLP RESULTS :")
    print(f"  Sentiment  : {record.overall_sentiment} ({record.overall_sentiment_score:.2f})")
    print(f"  Intent     : {record.primary_intent} ({record.primary_intent_score:.2f})")
    if record.all_entities:
        ents = [(e["text"], e["label"]) for e in record.all_entities[:5]]
        print(f"  Entities   : {ents}")

    print(f"\n  COMPLIANCE : {'PASS' if record.compliance_passed else 'FAIL'}")
    if record.compliance_flags:
        for flag in record.compliance_flags:
            severity = flag.get("severity","?").upper()
            print(f"    [{severity:8}] {flag.get('flag_type','')} — {flag.get('description','')}")
    else:
        print(f"    No issues detected.")

    # WER vs script
    if reference:
        wer_result = compute_wer(reference, record.full_transcript)
        print(f"\n  ASR ACCURACY (vs synthesized script):")
        if wer_result["wer"] is not None:
            wer_pct = wer_result["wer"]
            cer_pct = wer_result["cer"]
            status  = "PASS" if wer_pct < 0.15 else "FAIL (> 15% target)"
            print(f"    WER : {wer_pct:.2%}  [{status}]")
            print(f"    CER : {cer_pct:.2%}")
        else:
            print(f"    {wer_result.get('note')}")

    print(f"\n  Reports saved to: reports/")
    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Simulate a call and run the full ASR pipeline on it"
    )
    parser.add_argument("--scenario",  default="billing_dispute",
                        choices=list(SCENARIOS.keys()),
                        help="Which scenario to simulate")
    parser.add_argument("--text",      default="",
                        help="Custom dialogue text (overrides scenario)")
    parser.add_argument("--model",     default="base",
                        choices=["tiny","base","small","medium"])
    parser.add_argument("--agent",     default="SIM-001")
    parser.add_argument("--lang",      default="en")
    parser.add_argument("--keep-audio",action="store_true",
                        help="Keep generated MP3 in data/raw/synth/")
    parser.add_argument("--list-scenarios", action="store_true",
                        help="Show available scenarios and exit")
    args = parser.parse_args()

    # List scenarios
    if args.list_scenarios:
        print("\nAvailable scenarios:")
        for name, info in SCENARIOS.items():
            print(f"  {name:25} → intent: {info['intent']}")
            if info["flags"]:
                print(f"  {'':25}   flags:  {info['flags']}")
        return

    # Get dialogue
    if args.text.strip():
        dialogue      = args.text.strip()
        scenario_name = "custom"
        reference     = dialogue
        expected_flags = []
    else:
        sc            = SCENARIOS[args.scenario]
        dialogue      = sc["dialogue"]
        scenario_name = args.scenario
        reference     = dialogue
        expected_flags = sc.get("flags", [])

    print(f"\nScenario : {scenario_name}")
    print(f"Model    : whisper-{args.model}")
    print(f"Dialogue : {dialogue[:80]}…\n")

    # Generate audio
    if args.keep_audio:
        Path("data/raw/synth").mkdir(parents=True, exist_ok=True)
        audio_path = f"data/raw/synth/simulated_{scenario_name}.mp3"
        generate_audio(dialogue, audio_path, lang=args.lang)
        cleanup = False
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            audio_path = tmp.name
        generate_audio(dialogue, audio_path, lang=args.lang)
        cleanup = True

    # Run pipeline
    try:
        result = run_pipeline(audio_path, model_size=args.model, agent_id=args.agent)
        print_results(result, reference=reference, scenario=scenario_name)

        # Check expected vs detected flags
        if expected_flags:
            detected = [f["flag_type"] for f in result["record"].compliance_flags]
            print("  Expected flags  :", expected_flags)
            print("  Detected flags  :", detected)
            hit  = [f for f in expected_flags if f in detected]
            miss = [f for f in expected_flags if f not in detected]
            if miss:
                print(f"  Missed flags    : {miss}")
            else:
                print(f"  All expected flags correctly detected!")

    finally:
        if cleanup:
            try:
                import os
                os.unlink(audio_path)
            except Exception:
                pass


if __name__ == "__main__":
    main()
