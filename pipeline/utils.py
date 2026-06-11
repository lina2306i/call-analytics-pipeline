"""
utils.py — Shared utilities for the Call Analytics Pipeline.
Provides: config loading, logger setup, CallRecord dataclass.
"""

import os
import json
import logging
import yaml
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


# ════════════════════════════════════════════════════════════
#  CONFIG LOADER
# ════════════════════════════════════════════════════════════

def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load and return the master YAML config.
    Merges environment variables for sensitive fields (HF_TOKEN).
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    #with open(config_path, "r") as f:
    with open(config_path, "r", encoding="utf-8") as f:

        cfg = yaml.safe_load(f)

    # Allow HF_TOKEN to be set via environment variable
    env_token = os.environ.get("HF_TOKEN", "")
    if env_token:
        cfg["diarization"]["hf_token"] = env_token

    return cfg


# ════════════════════════════════════════════════════════════
#  LOGGER SETUP
# ════════════════════════════════════════════════════════════

def get_logger(name: str, config: Optional[dict] = None) -> logging.Logger:
    """
    Return a configured logger for the given module name.
    Logs to console and optionally to file based on config.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on re-import
    if logger.handlers:
        return logger

    level_str  = "INFO"
    log_file   = None
    to_console = True
    to_file    = False

    if config:
        log_cfg    = config.get("logging", {})
        level_str  = log_cfg.get("level", "INFO")
        to_console = log_cfg.get("console", True)
        to_file    = log_cfg.get("file", False)
        log_file   = config.get("paths", {}).get("logs", "logs/pipeline.log")

    level = getattr(logging, level_str.upper(), logging.INFO)
    logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    if to_console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    if to_file and log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ════════════════════════════════════════════════════════════
#  CALL RECORD DATACLASS
# ════════════════════════════════════════════════════════════

@dataclass
class ComplianceFlag:
    flag_type: str             # MISSING_DISCLOSURE | PROHIBITED_LANGUAGE | PII_DETECTED
    description: str
    matched_text: str = ""
    start_sec: float = 0.0
    end_sec: float = 0.0


@dataclass
class SpeakerTurn:
    speaker: str               # e.g. "SPEAKER_00", "SPEAKER_01"
    text: str
    start_sec: float
    end_sec: float
    sentiment_label: str = ""  # POSITIVE | NEGATIVE | NEUTRAL
    sentiment_score: float = 0.0
    intent_label: str = ""
    intent_score: float = 0.0
    entities: list = field(default_factory=list)  # [(text, label), ...]


@dataclass
class CallRecord:
    # ── Identity ──────────────────────────────────────────
    call_id: str
    audio_path: str
    processed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # ── Audio metadata ────────────────────────────────────
    duration_sec: float = 0.0
    sample_rate: int = 16000
    num_channels: int = 1

    # ── Transcription ─────────────────────────────────────
    full_transcript: str = ""
    language_detected: str = ""
    whisper_model: str = ""

    # ── Speaker turns ─────────────────────────────────────
    turns: list = field(default_factory=list)  # list of SpeakerTurn

    # ── Aggregate NLP ─────────────────────────────────────
    overall_sentiment: str = ""
    overall_sentiment_score: float = 0.0
    primary_intent: str = ""
    primary_intent_score: float = 0.0
    all_entities: list = field(default_factory=list)

    # ── Compliance ────────────────────────────────────────
    compliance_flags: list = field(default_factory=list)  # list of ComplianceFlag
    compliance_passed: bool = True

    # ── Agent info (populated externally if known) ────────
    agent_id: str = ""
    call_date: str = ""

    # ── Pipeline metadata ─────────────────────────────────
    pipeline_version: str = "1.0.0"
    errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to plain dict (JSON serializable)."""
        d = asdict(self)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def add_error(self, stage: str, message: str):
        self.errors.append({"stage": stage, "message": message, "time": datetime.utcnow().isoformat()})


# ════════════════════════════════════════════════════════════
#  PATH HELPERS
# ════════════════════════════════════════════════════════════

def ensure_dirs(config: dict):
    """Create all output directories defined in config.paths."""
    for key, path_str in config.get("paths", {}).items():
        p = Path(path_str)
        if p.suffix == "":          # it's a directory (no file extension)
            p.mkdir(parents=True, exist_ok=True)
        else:                        # it's a file path — create parent dir
            p.parent.mkdir(parents=True, exist_ok=True)


def generate_call_id(audio_path: str) -> str:
    """Generate a unique call ID from filename + timestamp."""
    stem = Path(audio_path).stem
    ts   = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return f"CC-{stem}-{ts}"
