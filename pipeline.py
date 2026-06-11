"""
pipeline.py  —  Main Orchestrator
────────────────────────────────────────────────────────────
Wires all 6 modules together in sequence.
Handles errors per-stage without crashing the full batch.

CLI Usage:
    python pipeline.py --input data/raw/synth/ --output reports/
    python pipeline.py --input call.mp3 --agent A-117
    python pipeline.py --input data/raw/ --model medium --no-diarize

Programmatic usage:
    from pipeline import CallAnalyticsPipeline
    pipeline = CallAnalyticsPipeline()
    record   = pipeline.run("call.mp3")
"""

import sys
import glob
import time
import argparse
from pathlib import Path

from pipeline.utils        import load_config, get_logger, generate_call_id, ensure_dirs
from pipeline.audio_preprocessor import AudioPreprocessor
from pipeline.transcriber        import Transcriber
from pipeline.diarizer           import Diarizer
from pipeline.nlp_analyzer       import NLPAnalyzer
from pipeline.compliance_checker import ComplianceChecker
from pipeline.report_builder     import ReportBuilder


# ═══════════════════════════════════════════════════════════════════════════════
#  PIPELINE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class CallAnalyticsPipeline:
    """
    End-to-end call analytics pipeline.

    Stages (all independently skippable via config):
        1. Audio Preprocessing
        2. Transcription (Whisper)
        3. Speaker Diarization (pyannote)
        4. NLP Analysis (sentiment + intent + NER)
        5. Compliance Checking
        6. Report Building & Export

    Each stage failure is caught, logged, and the call record
    still gets written with partial results + error metadata.
    """

    VERSION = "1.0.0"

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.logger = get_logger("pipeline", self.config)
        ensure_dirs(self.config)

        self.logger.info(f"Call Analytics Pipeline v{self.VERSION} initialising…")

        # Instantiate all modules (models loaded lazily on first call)
        self.preprocessor = AudioPreprocessor(self.config)
        self.transcriber  = Transcriber(self.config)
        self.diarizer     = Diarizer(self.config)
        self.nlp          = NLPAnalyzer(self.config)
        self.compliance   = ComplianceChecker(self.config)
        self.reporter     = ReportBuilder(self.config)

        self.logger.info("All modules ready.")

    # ── Single call ───────────────────────────────────────────────────────────

    def run(self, audio_path: str, agent_id: str = "") -> dict:
        """
        Process a single audio file end-to-end.

        Returns a dict with:
            record    : CallRecord (full results)
            outputs   : {format: file_path}
            success   : bool
            elapsed   : float (total seconds)
            errors    : list of {stage, message}
        """
        call_id = generate_call_id(audio_path)
        t0      = time.perf_counter()
        errors  = []

        self.logger.info(f"\n{'─'*60}")
        self.logger.info(f"Processing: {Path(audio_path).name}  [{call_id}]")

        # ── Stage 1: Preprocessing ────────────────────────────────────────
        audio = None
        try:
            audio = self.preprocessor.preprocess(audio_path)
            if audio.warnings:
                for w in audio.warnings:
                    self.logger.warning(f"  Audio warning: {w}")
        except Exception as e:
            msg = f"Audio preprocessing failed: {e}"
            self.logger.error(f"  ✗ Stage 1 — {msg}")
            errors.append({"stage": "preprocessing", "message": str(e)})
            return self._failed_result(call_id, audio_path, errors, t0)

        # ── Stage 2: Transcription ────────────────────────────────────────
        transcript = None
        try:
            transcript = self.transcriber.transcribe(audio)
        except Exception as e:
            msg = f"Transcription failed: {e}"
            self.logger.error(f"  ✗ Stage 2 — {msg}")
            errors.append({"stage": "transcription", "message": str(e)})
            return self._failed_result(call_id, audio_path, errors, t0)

        # ── Stage 3: Diarization ──────────────────────────────────────────
        diarized = None
        try:
            diarized = self.diarizer.diarize(audio_path, transcript, audio.duration_sec)
        except Exception as e:
            self.logger.warning(f"  ⚠ Stage 3 — Diarization failed: {e}. Continuing.")
            errors.append({"stage": "diarization", "message": str(e)})
            # Fallback: run single-speaker diarization so pipeline continues
            diarized = self.diarizer._fallback_diarization(transcript, audio.duration_sec)

        # ── Stage 4: NLP Analysis ─────────────────────────────────────────
        nlp_result = None
        try:
            nlp_result = self.nlp.analyze(diarized)
        except Exception as e:
            self.logger.warning(f"  ⚠ Stage 4 — NLP failed: {e}. Continuing.")
            errors.append({"stage": "nlp_analysis", "message": str(e)})
            # Create empty NLP result so pipeline continues
            from pipeline.nlp_analyzer import NLPResult
            nlp_result = NLPResult(annotated_turns=[])

        # ── Stage 5: Compliance ───────────────────────────────────────────
        compliance_result = None
        try:
            compliance_result = self.compliance.check(nlp_result, transcript.text)
        except Exception as e:
            self.logger.warning(f"  ⚠ Stage 5 — Compliance failed: {e}. Continuing.")
            errors.append({"stage": "compliance", "message": str(e)})
            from pipeline.compliance_checker import ComplianceResult
            compliance_result = ComplianceResult(flags=[], passed=True, score=1.0,
                                                  summary="Compliance check skipped due to error")

        # ── Stage 6: Report Building ──────────────────────────────────────
        record  = None
        outputs = {}
        try:
            record = self.reporter.build(
                call_id, audio, transcript,
                diarized, nlp_result, compliance_result,
                agent_id=agent_id,
            )
            for err in errors:
                record.add_error(err["stage"], err["message"])

            outputs = self.reporter.export(record)
        except Exception as e:
            self.logger.error(f"  ✗ Stage 6 — Report building failed: {e}")
            errors.append({"stage": "report_building", "message": str(e)})

        elapsed = time.perf_counter() - t0
        status  = "✓" if not errors else f"⚠ ({len(errors)} errors)"
        self.logger.info(
            f"  {status} Completed in {elapsed:.1f}s | "
            f"intent={record.primary_intent if record else '?'} | "
            f"compliance={'PASS' if (compliance_result and compliance_result.passed) else 'FAIL'}"
        )

        return {
            "record":  record,
            "outputs": outputs,
            "success": len([e for e in errors if e["stage"] in ("preprocessing","transcription")]) == 0,
            "elapsed": elapsed,
            "errors":  errors,
        }

    # ── Batch processing ──────────────────────────────────────────────────────

    def run_batch(
        self,
        input_dir:  str,
        agent_id:   str = "",
        extensions: list = None,
    ) -> dict:
        """
        Process all audio files in a directory.

        Returns:
            {
                results    : list of per-call result dicts
                records    : list of CallRecord objects
                total      : int
                succeeded  : int
                failed     : int
                elapsed    : float
            }
        """
        if extensions is None:
            extensions = [".mp3", ".wav", ".flac", ".ogg", ".m4a"]

        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Collect audio files
        audio_files = []
        for ext in extensions:
            audio_files.extend(input_path.glob(f"*{ext}"))
            audio_files.extend(input_path.glob(f"*{ext.upper()}"))

        audio_files = sorted(set(audio_files))

        if not audio_files:
            self.logger.warning(f"No audio files found in: {input_dir}")
            return {"results": [], "records": [], "total": 0,
                    "succeeded": 0, "failed": 0, "elapsed": 0.0}

        self.logger.info(f"\n{'═'*60}")
        self.logger.info(f"Batch processing: {len(audio_files)} files in {input_dir}")
        self.logger.info(f"{'═'*60}")

        t0       = time.perf_counter()
        results  = []
        records  = []
        succeeded = 0
        failed    = 0

        for i, audio_file in enumerate(audio_files, 1):
            self.logger.info(f"\n[{i}/{len(audio_files)}]")
            result = self.run(str(audio_file), agent_id=agent_id)
            results.append(result)

            if result["success"] and result["record"]:
                records.append(result["record"])
                succeeded += 1
            else:
                failed += 1

        # Write batch Excel report
        if records:
            self.reporter.export_batch(records)

        elapsed = time.perf_counter() - t0
        self.logger.info(f"\n{'═'*60}")
        self.logger.info(
            f"Batch complete in {elapsed:.1f}s | "
            f"✓ {succeeded} succeeded | ✗ {failed} failed"
        )
        self._print_batch_summary(records)

        return {
            "results":   results,
            "records":   records,
            "total":     len(audio_files),
            "succeeded": succeeded,
            "failed":    failed,
            "elapsed":   elapsed,
        }

    # ── Summary ───────────────────────────────────────────────────────────────

    def _print_batch_summary(self, records):
        if not records:
            return

        total        = len(records)
        passed       = sum(1 for r in records if r.compliance_passed)
        failed_comp  = total - passed
        intents      = {}
        sentiments   = {}

        for r in records:
            intents[r.primary_intent]        = intents.get(r.primary_intent, 0) + 1
            sentiments[r.overall_sentiment]  = sentiments.get(r.overall_sentiment, 0) + 1

        self.logger.info("\n── Batch Summary ────────────────────────────────")
        self.logger.info(f"  Total calls     : {total}")
        self.logger.info(f"  Compliance pass : {passed}/{total} ({passed/total:.0%})")
        self.logger.info(f"  Compliance fail : {failed_comp}/{total}")
        self.logger.info(f"  Top intents     : {sorted(intents.items(), key=lambda x:-x[1])[:3]}")
        self.logger.info(f"  Sentiments      : {sentiments}")
        self.logger.info("─────────────────────────────────────────────────")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _failed_result(self, call_id, audio_path, errors, t0):
        """Return a minimal failure result dict."""
        from pipeline.utils import CallRecord
        record = CallRecord(call_id=call_id, audio_path=audio_path)
        for e in errors:
            record.add_error(e["stage"], e["message"])
        return {
            "record":  record,
            "outputs": {},
            "success": False,
            "elapsed": time.perf_counter() - t0,
            "errors":  errors,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Call Analytics Pipeline — process call recordings end-to-end"
    )
    p.add_argument("--input",     required=True,
                   help="Path to audio file OR directory of audio files")
    p.add_argument("--output",    default="reports",
                   help="Output directory for reports (default: reports/)")
    p.add_argument("--agent",     default="",
                   help="Agent ID to tag in reports")
    p.add_argument("--model",     default=None,
                   help="Whisper model size: tiny|base|small|medium|large")
    p.add_argument("--no-diarize",action="store_true",
                   help="Skip speaker diarization (faster, no pyannote needed)")
    p.add_argument("--config",    default="config.yaml",
                   help="Path to config.yaml")
    p.add_argument("--generate",  action="store_true",
                   help="Generate synthetic calls before running pipeline")
    return p.parse_args()


def main():
    args = parse_args()

    # Optionally generate synthetic data first
    if args.generate:
        print("Generating synthetic calls…")
        from generate_synthetic_calls import generate_all
        generate_all("data/raw/synth")

    # Override config with CLI flags
    import yaml
    #with open(args.config) as f:
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.model:
        cfg["whisper"]["model"] = args.model
    if args.no_diarize:
        cfg["diarization"]["enabled"] = False
    if args.output:
        cfg["paths"]["reports"] = args.output

    # Write patched config to a temp file
    tmp_cfg = ".pipeline_run.yaml"
    with open(tmp_cfg, "w") as f:
        yaml.dump(cfg, f)

    pipeline = CallAnalyticsPipeline(config_path=tmp_cfg)
    Path(tmp_cfg).unlink(missing_ok=True)

    input_path = Path(args.input)

    if input_path.is_dir():
        result = pipeline.run_batch(str(input_path), agent_id=args.agent)
        print(f"\nBatch done: {result['succeeded']}/{result['total']} succeeded "
              f"in {result['elapsed']:.1f}s")
    elif input_path.is_file():
        result = pipeline.run(str(input_path), agent_id=args.agent)
        rec = result["record"]
        print(f"\nCall ID   : {rec.call_id}")
        print(f"Sentiment : {rec.overall_sentiment}")
        print(f"Intent    : {rec.primary_intent}")
        print(f"Compliance: {'PASS' if rec.compliance_passed else 'FAIL'}")
        print(f"Flags     : {len(rec.compliance_flags)}")
        print(f"Reports   → {result['outputs']}")
    else:
        print(f"Error: --input path not found: {args.input}")
        sys.exit(1)


if __name__ == "__main__":
    main()
