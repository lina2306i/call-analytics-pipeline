"""
pipeline/report_builder.py  —  Module 6
────────────────────────────────────────────────────────────
Assembles all module outputs into a CallRecord and exports:
  • Per-call  JSON   (reports/calls/CALL_ID.json)
  • Daily     CSV    (reports/batch_YYYY-MM-DD.csv)
  • Excel workbook   (reports/summary_YYYY-MM-DD.xlsx)
    with sheets: Calls, Flags, Intents, Sentiment, Entities

Input  : all module results + audio metadata
Output : CallRecord dataclass  +  files on disk
"""

import json
import time
import sqlite3
from datetime import datetime, date
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, List

from pipeline.utils import load_config, get_logger, CallRecord, ComplianceFlag
from pipeline.audio_preprocessor import PreprocessedAudio
from pipeline.transcriber import TranscriptionResult
from pipeline.diarizer import DiarizationResult
from pipeline.nlp_analyzer import NLPResult
from pipeline.compliance_checker import ComplianceResult


# ═══════════════════════════════════════════════════════════════════════════════
#  REPORT BUILDER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class ReportBuilder:
    """
    Assembles pipeline outputs into a CallRecord and writes reports.

    Example:
        builder = ReportBuilder(config)
        record  = builder.build(call_id, audio, transcript,
                                diarization, nlp_result, compliance)
        builder.export(record)
    """

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()

        self.cfg    = config.get("reports", {})
        self.logger = get_logger("report_builder", config)

        self.formats      = self.cfg.get("formats", ["json", "csv", "xlsx"])
        self.inc_transcript = self.cfg.get("include_transcript", True)
        self.inc_segments   = self.cfg.get("include_segments",   True)

        self.reports_dir = Path(config.get("paths", {}).get("reports", "reports"))
        self.db_path     = config.get("paths", {}).get("database", "reports/calls.db")

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        (self.reports_dir / "calls").mkdir(exist_ok=True)

    # ── Build CallRecord ──────────────────────────────────────────────────────

    def build(
        self,
        call_id:     str,
        audio:       PreprocessedAudio,
        transcript:  TranscriptionResult,
        diarization: DiarizationResult,
        nlp_result:  NLPResult,
        compliance:  ComplianceResult,
        agent_id:    str = "",
    ) -> CallRecord:
        """
        Assemble all module outputs into a single CallRecord.
        """
        # Build speaker turns list (plain dicts for JSON safety)
        turns = []
        for ann in nlp_result.annotated_turns:
            turns.append({
                "speaker":         ann.speaker,
                "role":            ann.role,
                "text":            ann.text,
                "start_sec":       ann.start_sec,
                "end_sec":         ann.end_sec,
                "sentiment_label": ann.sentiment_label,
                "sentiment_score": ann.sentiment_score,
                "intent_label":    ann.intent_label,
                "intent_score":    ann.intent_score,
                "entities":        [{"text": e[0], "label": e[1]}
                                    if isinstance(e, tuple) else e
                                    for e in ann.entities],
            })

        # Build compliance flags list
        flags = []
        for f in compliance.flags:
            flags.append({
                "flag_type":    f.flag_type,
                "description":  f.description,
                "matched_text": f.matched_text,
                "speaker":      f.speaker,
                "role":         f.role,
                "start_sec":    f.start_sec,
                "end_sec":      f.end_sec,
                "severity":     f.severity,
            })

        record = CallRecord(
            call_id=call_id,
            audio_path=audio.file_path,
            duration_sec=audio.duration_sec,
            sample_rate=audio.sample_rate,
            num_channels=audio.original_channels,
            full_transcript=transcript.text if self.inc_transcript else "",
            language_detected=transcript.language,
            whisper_model=transcript.model_used,
            turns=turns if self.inc_segments else [],
            overall_sentiment=nlp_result.overall_sentiment,
            overall_sentiment_score=nlp_result.overall_sentiment_score,
            primary_intent=nlp_result.primary_intent,
            primary_intent_score=nlp_result.primary_intent_score,
            all_entities=nlp_result.all_entities,
            compliance_flags=flags,
            compliance_passed=compliance.passed,
            agent_id=agent_id,
            call_date=date.today().isoformat(),
        )

        self.logger.info(
            f"  Record built: {call_id} | "
            f"compliance={'PASS' if compliance.passed else 'FAIL'} | "
            f"intent={nlp_result.primary_intent}"
        )
        return record

    # ── Export ────────────────────────────────────────────────────────────────

    def export(self, record: CallRecord) -> dict:
        """
        Export a CallRecord to all configured formats.
        Returns dict of {format: file_path}.
        """
        outputs = {}

        if "json" in self.formats:
            outputs["json"] = self._write_json(record)
        if "csv" in self.formats:
            outputs["csv"]  = self._write_csv(record)
        if "xlsx" in self.formats:
            pass   # xlsx written per-batch (see write_batch_excel)

        self._write_sqlite(record)
        return outputs

    def export_batch(self, records: List[CallRecord]) -> dict:
        """
        Export a list of CallRecords. Also writes the Excel batch report.
        """
        outputs = {}
        for rec in records:
            outputs[rec.call_id] = self.export(rec)

        if "xlsx" in self.formats and records:
            xlsx_path = self._write_batch_excel(records)
            for cid in outputs:
                outputs[cid]["xlsx"] = xlsx_path

        self.logger.info(f"Batch export complete — {len(records)} records")
        return outputs

    # ── JSON ──────────────────────────────────────────────────────────────────

    def _write_json(self, record: CallRecord) -> str:
        path = self.reports_dir / "calls" / f"{record.call_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2, ensure_ascii=False)
        self.logger.debug(f"  JSON → {path}")
        return str(path)

    # ── CSV ───────────────────────────────────────────────────────────────────

    def _write_csv(self, record: CallRecord) -> str:
        """Append a one-row summary to the daily CSV."""
        try:
            import pandas as pd
        except ImportError:
            self.logger.warning("pandas not installed — CSV export skipped")
            return ""

        path = self.reports_dir / f"batch_{date.today().isoformat()}.csv"

        row = {
            "call_id":               record.call_id,
            "call_date":             record.call_date,
            "agent_id":              record.agent_id,
            "duration_sec":          record.duration_sec,
            "language":              record.language_detected,
            "overall_sentiment":     record.overall_sentiment,
            "sentiment_score":       record.overall_sentiment_score,
            "primary_intent":        record.primary_intent,
            "intent_score":          record.primary_intent_score,
            "compliance_passed":     record.compliance_passed,
            "num_flags":             len(record.compliance_flags),
            "flag_types":            "|".join(set(f["flag_type"] for f in record.compliance_flags)),
            "num_turns":             len(record.turns),
            "num_entities":          len(record.all_entities),
            "whisper_model":         record.whisper_model,
            "processed_at":          record.processed_at,
        }

        df_new = pd.DataFrame([row])

        if path.exists():
            df_existing = pd.read_csv(path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new

        df_combined.to_csv(path, index=False)
        self.logger.debug(f"  CSV  → {path}")
        return str(path)

    # ── Excel ─────────────────────────────────────────────────────────────────

    def _write_batch_excel(self, records: List[CallRecord]) -> str:
        """
        Write a multi-sheet Excel workbook from a batch of records.
        Sheets: Summary, Flags, Intents, Sentiment, Entities
        """
        try:
            import pandas as pd
        except ImportError:
            self.logger.warning("pandas not installed — Excel export skipped")
            return ""

        path = self.reports_dir / f"summary_{date.today().isoformat()}.xlsx"

        # ── Sheet 1: Summary ──────────────────────────────────────────────
        summary_rows = []
        for r in records:
            summary_rows.append({
                "call_id":           r.call_id,
                "call_date":         r.call_date,
                "agent_id":          r.agent_id,
                "duration_sec":      r.duration_sec,
                "overall_sentiment": r.overall_sentiment,
                "primary_intent":    r.primary_intent,
                "compliance_passed": r.compliance_passed,
                "num_flags":         len(r.compliance_flags),
                "language":          r.language_detected,
            })
        df_summary = pd.DataFrame(summary_rows)

        # ── Sheet 2: Flags ────────────────────────────────────────────────
        flag_rows = []
        for r in records:
            for f in r.compliance_flags:
                flag_rows.append({
                    "call_id":     r.call_id,
                    "agent_id":    r.agent_id,
                    "flag_type":   f["flag_type"],
                    "severity":    f["severity"],
                    "description": f["description"],
                    "matched_text":f["matched_text"],
                    "speaker":     f["speaker"],
                    "role":        f["role"],
                    "start_sec":   f["start_sec"],
                })
        df_flags = pd.DataFrame(flag_rows) if flag_rows else pd.DataFrame(
            columns=["call_id","agent_id","flag_type","severity","description","matched_text","role"])

        # ── Sheet 3: Intents ──────────────────────────────────────────────
        intent_rows = []
        for r in records:
            intent_rows.append({
                "call_id":     r.call_id,
                "agent_id":    r.agent_id,
                "intent":      r.primary_intent,
                "score":       r.primary_intent_score,
                "call_date":   r.call_date,
            })
        df_intents = pd.DataFrame(intent_rows)

        # ── Sheet 4: Sentiment ────────────────────────────────────────────
        sentiment_rows = []
        for r in records:
            for turn in r.turns:
                sentiment_rows.append({
                    "call_id":   r.call_id,
                    "agent_id":  r.agent_id,
                    "speaker":   turn.get("speaker"),
                    "role":      turn.get("role"),
                    "sentiment": turn.get("sentiment_label"),
                    "score":     turn.get("sentiment_score"),
                    "start_sec": turn.get("start_sec"),
                })
        df_sentiment = pd.DataFrame(sentiment_rows)

        # ── Sheet 5: Entities ─────────────────────────────────────────────
        entity_rows = []
        for r in records:
            for ent in r.all_entities:
                entity_rows.append({
                    "call_id":  r.call_id,
                    "agent_id": r.agent_id,
                    "text":     ent.get("text"),
                    "label":    ent.get("label"),
                })
        df_entities = pd.DataFrame(entity_rows)

        # ── Write workbook ────────────────────────────────────────────────
        with pd.ExcelWriter(str(path), engine="openpyxl") as writer:
            df_summary.to_excel(writer,  sheet_name="Summary",   index=False)
            df_flags.to_excel(writer,    sheet_name="Flags",      index=False)
            df_intents.to_excel(writer,  sheet_name="Intents",    index=False)
            df_sentiment.to_excel(writer,sheet_name="Sentiment",  index=False)
            df_entities.to_excel(writer, sheet_name="Entities",   index=False)

            # ── Pivot: compliance rate by agent ──────────────────────────
            if not df_summary.empty and "agent_id" in df_summary.columns:
                pivot = df_summary.pivot_table(
                    values="compliance_passed",
                    index="agent_id",
                    aggfunc=["count", "sum", "mean"],
                )
                pivot.columns = ["total_calls", "passed_calls", "pass_rate"]
                pivot.to_excel(writer, sheet_name="Agent_Pivot")

        self.logger.info(f"  XLSX → {path}")
        return str(path)

    # ── SQLite ────────────────────────────────────────────────────────────────

    def _write_sqlite(self, record: CallRecord):
        """
        Persist call record to SQLite for ad-hoc querying.
        Creates the table on first run.
        """
        try:
            import pandas as pd
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path)

            row = {
                "call_id":           record.call_id,
                "call_date":         record.call_date,
                "agent_id":          record.agent_id,
                "duration_sec":      record.duration_sec,
                "language":          record.language_detected,
                "overall_sentiment": record.overall_sentiment,
                "sentiment_score":   record.overall_sentiment_score,
                "primary_intent":    record.primary_intent,
                "intent_score":      record.primary_intent_score,
                "compliance_passed": int(record.compliance_passed),
                "num_flags":         len(record.compliance_flags),
                "num_turns":         len(record.turns),
                "whisper_model":     record.whisper_model,
                "processed_at":      record.processed_at,
            }

            pd.DataFrame([row]).to_sql(
                "calls", conn,
                if_exists="append",
                index=False
            )
            conn.close()
            self.logger.debug(f"  SQLite ← {record.call_id}")

        except Exception as e:
            self.logger.warning(f"  SQLite write failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import glob, sys
    from pathlib import Path
    from pipeline.audio_preprocessor import AudioPreprocessor
    from pipeline.transcriber import Transcriber
    from pipeline.diarizer import Diarizer
    from pipeline.nlp_analyzer import NLPAnalyzer
    from pipeline.compliance_checker import ComplianceChecker
    from pipeline.utils import generate_call_id

    cfg     = load_config()
    proc    = AudioPreprocessor(cfg)
    trans   = Transcriber(cfg)
    diar    = Diarizer(cfg)
    nlp     = NLPAnalyzer(cfg)
    checker = ComplianceChecker(cfg)
    builder = ReportBuilder(cfg)

    files = sorted(glob.glob("data/raw/synth/*.mp3"))[:3]
    if not files:
        print("No synth files found.")
        sys.exit(0)

    records = []
    for f in files:
        print(f"\nProcessing: {Path(f).name}")
        call_id    = generate_call_id(f)
        audio      = proc.preprocess(f)
        transcript = trans.transcribe(audio)
        diarized   = diar.diarize(f, transcript, audio.duration_sec)
        nlp_result = nlp.analyze(diarized)
        compliance = checker.check(nlp_result)
        record     = builder.build(call_id, audio, transcript,
                                   diarized, nlp_result, compliance)
        records.append(record)
        builder.export(record)
        print(f"  → {call_id} | intent={record.primary_intent} | "
              f"compliance={'PASS' if record.compliance_passed else 'FAIL'}")

    # Batch Excel
    builder.export_batch(records)
    print(f"\nDone. Reports in: {builder.reports_dir}/")
