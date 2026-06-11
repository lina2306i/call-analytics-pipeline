"""
benchmark_models.py  —  Whisper Model Benchmarking
────────────────────────────────────────────────────────────
Compares Whisper tiny / base / small / medium on:
  - WER  (Word Error Rate)
  - CER  (Character Error Rate)
  - RTF  (Real-Time Factor = inference / audio duration)
  - Inference speed (seconds per call)

Usage:
    python benchmark_models.py
    python benchmark_models.py --models tiny base small
    python benchmark_models.py --max-files 5   # quick test

Output:
    evaluation/benchmark_results.csv
    evaluation/benchmark_summary.json
"""

import re
import json
import time
import argparse
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
    from jiwer import wer, cer
    import whisper
    import librosa
except ImportError as e:
    raise SystemExit(f"Missing dependency: {e}\nRun: pip install openai-whisper librosa jiwer pandas")


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_ground_truth(gt_path: str) -> dict:
    with open(gt_path, encoding="utf-8") as f:
        gt_list = json.load(f)
    return {
        Path(item["audio_file"]).name: normalize(item.get("transcript_ref", ""))
        for item in gt_list
        if item.get("transcript_ref")
    }


def load_audio(path: str):
    audio, sr = librosa.load(str(path), sr=16000, mono=True)
    return audio, sr


# ═══════════════════════════════════════════════════════════════════════════════
#  BENCHMARK SINGLE MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_one_model(
    model_size:   str,
    audio_files:  list,
    ground_truth: dict,
) -> list:
    """
    Run Whisper model_size on all audio_files.
    Returns list of result dicts.
    """
    print(f"\n  Loading whisper-{model_size}...")
    t_load = time.perf_counter()
    model  = whisper.load_model(model_size)
    load_time = time.perf_counter() - t_load
    print(f"  Loaded in {load_time:.1f}s")

    results = []
    for audio_path in audio_files:
        filename = Path(audio_path).name
        ref      = ground_truth.get(filename, "")
        if not ref:
            continue

        audio, sr    = load_audio(str(audio_path))
        duration_sec = len(audio) / sr

        t0 = time.perf_counter()
        try:
            out = model.transcribe(audio, language="en",
                                   word_timestamps=False, verbose=False)
            hyp = normalize(out["text"].strip())
        except Exception as e:
            print(f"    Error on {filename}: {e}")
            continue
        inference_sec = time.perf_counter() - t0

        wer_score = wer(ref, hyp) if ref and hyp else 1.0
        cer_score = cer(ref, hyp) if ref and hyp else 1.0
        rtf       = inference_sec / max(duration_sec, 0.001)

        results.append({
            "model":          model_size,
            "filename":       filename,
            "duration_sec":   round(duration_sec, 2),
            "inference_sec":  round(inference_sec, 3),
            "rtf":            round(rtf, 4),
            "wer":            round(wer_score, 4),
            "cer":            round(cer_score, 4),
            "load_time_sec":  round(load_time, 2),
        })

    # Free model from GPU memory
    del model
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  BENCHMARK ALL MODELS
# ═══════════════════════════════════════════════════════════════════════════════

def run_benchmark(
    models:       list,
    audio_dir:    str,
    gt_path:      str,
    output_dir:   str,
    max_files:    int = 0,
) -> pd.DataFrame:

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load ground truth
    gt = load_ground_truth(gt_path)

    # Collect audio files
    audio_files = sorted([
        f for f in Path(audio_dir).glob("*.mp3")
        if f.name in gt
    ])
    if max_files > 0:
        audio_files = audio_files[:max_files]

    print(f"\nBenchmarking {len(models)} models on {len(audio_files)} files")
    print(f"Models: {models}")

    all_results = []
    for model_size in models:
        print(f"\n{'─'*50}")
        print(f"  MODEL: whisper-{model_size.upper()}")
        print(f"{'─'*50}")
        results = benchmark_one_model(model_size, audio_files, gt)
        all_results.extend(results)
        if results:
            avg_wer = sum(r["wer"] for r in results) / len(results)
            avg_rtf = sum(r["rtf"] for r in results) / len(results)
            print(f"  Avg WER: {avg_wer:.2%}  |  Avg RTF: {avg_rtf:.2f}x")

    df = pd.DataFrame(all_results)

    if df.empty:
        print("No results collected.")
        return df

    # ── Save raw results ──────────────────────────────────────────────────────
    csv_path = out / "benchmark_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    # ── Summary per model ─────────────────────────────────────────────────────
    summary = df.groupby("model").agg(
        files_evaluated  =("filename",       "count"),
        mean_wer         =("wer",            "mean"),
        median_wer       =("wer",            "median"),
        mean_cer         =("cer",            "mean"),
        mean_rtf         =("rtf",            "mean"),
        mean_inference_sec=("inference_sec", "mean"),
        load_time_sec    =("load_time_sec",  "first"),
    ).round(4)

    print(f"\n{'═'*70}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'═'*70}")
    print(f"\n{'Model':10} {'WER':>8} {'CER':>8} {'RTF':>8} {'Inf(s)':>8} {'Load(s)':>8}")
    print(f"{'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    model_order = ["tiny", "base", "small", "medium", "large"]
    for model_size in model_order:
        if model_size not in summary.index:
            continue
        row = summary.loc[model_size]
        wer_pass = "✓" if row["mean_wer"] < 0.15 else " "
        print(f"  {model_size:8} "
              f"{row['mean_wer']:>7.2%} "
              f"{row['mean_cer']:>8.2%} "
              f"{row['mean_rtf']:>8.2f}x "
              f"{row['mean_inference_sec']:>7.1f}s "
              f"{row['load_time_sec']:>7.1f}s  {wer_pass}")

    print(f"\n  WER target < 15%")
    print(f"\n  Recommendation:")

    # Find best accuracy/speed balance
    if "base" in summary.index and summary.loc["base", "mean_wer"] < 0.15:
        print(f"  Use whisper-BASE — good accuracy (<15% WER) + fast RTF")
    elif "small" in summary.index and summary.loc["small", "mean_wer"] < 0.15:
        print(f"  Use whisper-SMALL — meets WER target with acceptable speed")
    else:
        print(f"  Use whisper-MEDIUM for best accuracy on this domain")

    # ── Save summary ──────────────────────────────────────────────────────────
    summary_dict = summary.reset_index().to_dict(orient="records")
    json_path    = out / "benchmark_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    print(f"\n  Results saved:")
    print(f"    {csv_path}")
    print(f"    {json_path}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Whisper models")
    parser.add_argument("--models",   nargs="+",
                        default=["tiny", "base", "small"],
                        choices=["tiny","base","small","medium","large"],
                        help="Models to benchmark")
    parser.add_argument("--audio-dir",    default="data/raw/synth")
    parser.add_argument("--ground-truth", default="data/raw/synth/ground_truth.json")
    parser.add_argument("--output",       default="evaluation")
    parser.add_argument("--max-files",    type=int, default=0,
                        help="Limit number of files (0 = all)")
    args = parser.parse_args()

    run_benchmark(
        models=args.models,
        audio_dir=args.audio_dir,
        gt_path=args.ground_truth,
        output_dir=args.output,
        max_files=args.max_files,
    )
