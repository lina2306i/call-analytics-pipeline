"""
evaluate.py  —  ASR Evaluation Script
────────────────────────────────────────────────────────────
Compares Whisper transcriptions against ground_truth.json.
Calculates WER, CER, and the impact of audio denoising.

Usage:
    python evaluate.py
    python evaluate.py --ground-truth data/raw/synth/ground_truth.json
    python evaluate.py --model base --denoise-compare

Output:
    evaluation/wer_results.csv
    evaluation/evaluation_report.json
"""

import os
import re
import json
import argparse
import time
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
    from jiwer import wer, cer
except ImportError:
    raise SystemExit("Run: pip install jiwer pandas numpy")

# ═══════════════════════════════════════════════════════════════════════════════
#  TEXT NORMALIZATION
#  Normalize text before WER/CER computation (remove punctuation, lowercase)
# ═══════════════════════════════════════════════════════════════════════════════

def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD GROUND TRUTH
# ═══════════════════════════════════════════════════════════════════════════════

def load_ground_truth(gt_path: str) -> dict:
    """
    Load ground_truth.json.
    Returns dict: {audio_filename: {transcript, intent, flags}}
    """
    with open(gt_path, encoding="utf-8") as f:
        gt_list = json.load(f)

    gt = {}
    for item in gt_list:
        audio_file = Path(item["audio_file"]).name
        gt[audio_file] = {
            "transcript_ref": item.get("transcript_ref", ""),
            "expected_intent": item.get("expected_intent", ""),
            "expected_flags":  item.get("expected_flags", []),
            "scenario":        item.get("scenario", ""),
            "call_id":         item.get("call_id", ""),
        }
    print(f"Loaded {len(gt)} ground truth records from {gt_path}")
    return gt


# ═══════════════════════════════════════════════════════════════════════════════
#  TRANSCRIBE + EVALUATE
# ═══════════════════════════════════════════════════════════════════════════════

def transcribe_and_evaluate(
    audio_files: list,
    ground_truth: dict,
    model_size:   str  = "base",
    denoise:      bool = True,
) -> list:
    """
    Transcribe each audio file with Whisper and compute WER/CER
    against the ground truth reference.
    Returns list of result dicts.
    """
    try:
        import whisper
        import librosa
        import numpy as np
    except ImportError:
        raise SystemExit("Run: pip install openai-whisper librosa")

    try:
        import noisereduce as nr
        HAS_NR = True
    except ImportError:
        HAS_NR = False
        if denoise:
            print("  noisereduce not found — denoising disabled")

    print(f"\nLoading Whisper model: {model_size}...")
    model = whisper.load_model(model_size)

    results = []
    total   = len(audio_files)

    for i, audio_path in enumerate(audio_files, 1):
        filename = Path(audio_path).name
        gt_entry = ground_truth.get(filename)

        if gt_entry is None:
            print(f"  [{i}/{total}] SKIP (no ground truth): {filename}")
            continue

        print(f"  [{i}/{total}] {filename}")

        # Load audio
        try:
            audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        except Exception as e:
            print(f"    Load error: {e}")
            continue

        # Denoise
        if denoise and HAS_NR:
            noise_sample = audio[:int(sr * 0.5)]
            audio_clean  = nr.reduce_noise(y=audio, sr=sr, y_noise=noise_sample,
                                           prop_decrease=0.85, stationary=False)
        else:
            audio_clean = audio

        # Transcribe
        t0 = time.perf_counter()
        try:
            result = model.transcribe(audio_clean, language="en",
                                      word_timestamps=False, verbose=False)
            hypothesis = result["text"].strip()
        except Exception as e:
            print(f"    Transcription error: {e}")
            continue
        inference_sec = time.perf_counter() - t0

        # Reference text
        reference = gt_entry["transcript_ref"]

        # Normalize both
        hyp_norm = normalize(hypothesis)
        ref_norm = normalize(reference)

        # Compute WER / CER
        try:
            wer_score = wer(ref_norm, hyp_norm)
            cer_score = cer(ref_norm, hyp_norm)
        except Exception:
            wer_score = 1.0
            cer_score = 1.0

        # Real-time factor (RTF = inference_time / audio_duration)
        duration = len(audio) / sr
        rtf      = inference_sec / max(duration, 0.001)

        results.append({
            "filename":       filename,
            "call_id":        gt_entry["call_id"],
            "scenario":       gt_entry["scenario"],
            "model":          model_size,
            "denoised":       denoise and HAS_NR,
            "duration_sec":   round(duration, 2),
            "inference_sec":  round(inference_sec, 2),
            "rtf":            round(rtf, 4),
            "wer":            round(wer_score, 4),
            "cer":            round(cer_score, 4),
            "hypothesis":     hypothesis[:200],
            "reference":      reference[:200],
            "expected_intent":gt_entry["expected_intent"],
            "expected_flags": "|".join(gt_entry["expected_flags"]),
        })

        print(f"    WER={wer_score:.2%}  CER={cer_score:.2%}  RTF={rtf:.2f}x  ({inference_sec:.1f}s)")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  DENOISE COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def denoise_impact_comparison(
    audio_files: list,
    ground_truth: dict,
    model_size:   str = "base",
    max_files:    int = 5,
) -> pd.DataFrame:
    """
    Compare WER with and without denoising on first N files.
    """
    print(f"\nDenoise impact comparison (first {max_files} files)...")
    sample_files = audio_files[:max_files]

    rows = []
    for denoise in [False, True]:
        label   = "With denoising" if denoise else "Without denoising"
        results = transcribe_and_evaluate(
            sample_files, ground_truth, model_size, denoise=denoise
        )
        for r in results:
            rows.append({
                "filename":     r["filename"],
                "condition":    label,
                "wer":          r["wer"],
                "cer":          r["cer"],
                "inference_sec":r["inference_sec"],
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        summary = df.groupby("condition")[["wer", "cer"]].mean()
        print("\nDenoise Impact Summary:")
        print(summary.to_string())
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(results: list, output_dir: str = "evaluation") -> dict:
    """Save CSV + JSON evaluation report."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    if df.empty:
        print("No results to report.")
        return {}

    # Save CSV
    csv_path = out / "wer_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    # Aggregate stats
    report = {
        "total_files":      len(df),
        "model":            df["model"].iloc[0] if not df.empty else "?",
        "mean_wer":         round(df["wer"].mean(), 4),
        "median_wer":       round(df["wer"].median(), 4),
        "min_wer":          round(df["wer"].min(), 4),
        "max_wer":          round(df["wer"].max(), 4),
        "mean_cer":         round(df["cer"].mean(), 4),
        "mean_rtf":         round(df["rtf"].mean(), 4),
        "mean_duration_sec":round(df["duration_sec"].mean(), 2),
        "wer_by_scenario":  df.groupby("scenario")["wer"]
                              .mean().round(4).to_dict(),
        "target_wer":       0.15,
        "passes_target":    bool(df["wer"].mean() < 0.15),
    }

    # Save JSON report
    json_path = out / "evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'═'*55}")
    print(f"  EVALUATION REPORT  —  Whisper-{report['model']}")
    print(f"{'═'*55}")
    print(f"  Files evaluated  : {report['total_files']}")
    print(f"  Mean WER         : {report['mean_wer']:.2%}  "
          f"(target < 15% — {'PASS' if report['passes_target'] else 'FAIL'})")
    print(f"  Median WER       : {report['median_wer']:.2%}")
    print(f"  Best WER         : {report['min_wer']:.2%}")
    print(f"  Worst WER        : {report['max_wer']:.2%}")
    print(f"  Mean CER         : {report['mean_cer']:.2%}")
    print(f"  Mean RTF         : {report['mean_rtf']:.2f}x real-time")
    print(f"\n  WER by scenario:")
    for scenario, wer_val in sorted(report["wer_by_scenario"].items(),
                                     key=lambda x: x[1]):
        bar = "█" * int(wer_val * 20)
        print(f"    {scenario:30} {wer_val:.2%}  {bar}")
    print(f"{'─'*55}")
    print(f"  Saved: {csv_path}")
    print(f"  Saved: {json_path}")

    return report


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluate ASR pipeline WER/CER")
    parser.add_argument("--ground-truth", default="data/raw/synth/ground_truth.json")
    parser.add_argument("--audio-dir",    default="data/raw/synth")
    parser.add_argument("--model",        default="base",
                        choices=["tiny","base","small","medium","large"])
    parser.add_argument("--output",       default="evaluation")
    parser.add_argument("--no-denoise",   action="store_true")
    parser.add_argument("--denoise-compare", action="store_true",
                        help="Compare WER with vs without denoising")
    args = parser.parse_args()

    # Load ground truth
    gt = load_ground_truth(args.ground_truth)

    # Collect audio files that have ground truth
    audio_dir   = Path(args.audio_dir)
    audio_files = sorted([
        f for f in audio_dir.glob("*.mp3")
        if f.name in gt
    ])

    if not audio_files:
        print(f"No matching audio files found in {audio_dir}")
        return

    print(f"Found {len(audio_files)} audio files with ground truth")

    # Optional denoise comparison
    if args.denoise_compare:
        dc_df = denoise_impact_comparison(audio_files, gt, args.model)
        dc_path = Path(args.output) / "denoise_comparison.csv"
        Path(args.output).mkdir(parents=True, exist_ok=True)
        dc_df.to_csv(dc_path, index=False, encoding="utf-8")
        print(f"Denoise comparison saved: {dc_path}")

    # Main evaluation
    results = transcribe_and_evaluate(
        audio_files, gt,
        model_size=args.model,
        denoise=not args.no_denoise,
    )

    # Generate report
    generate_report(results, args.output)


if __name__ == "__main__":
    main()
