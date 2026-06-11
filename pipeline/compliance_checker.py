"""
pipeline/compliance_checker.py  —  Module 5
────────────────────────────────────────────────────────────
Hybrid rule-based + pattern compliance checker.

Checks for:
  1. MISSING_DISCLOSURE  — required phrases not spoken by agent
  2. PROHIBITED_LANGUAGE — banned words/phrases spoken by anyone
  3. PII_DETECTED        — credit cards, SSN, phone numbers, emails
                           spoken aloud in transcript

Each flag includes: type, description, matched_text, and
the timestamp of the offending turn (for QA audio playback).

Input  : NLPResult (annotated turns) + full transcript string
Output : ComplianceResult(flags, passed, score)
"""

import re
import time
from dataclasses import dataclass, field
from typing import Optional, List

from pipeline.utils import load_config, get_logger
from pipeline.nlp_analyzer import NLPResult, AnnotatedTurn


# ═══════════════════════════════════════════════════════════════════════════════
#  OUTPUT DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ComplianceFlag:
    flag_type:    str    # MISSING_DISCLOSURE | PROHIBITED_LANGUAGE | PII_DETECTED
    description:  str
    matched_text: str  = ""
    speaker:      str  = ""
    role:         str  = ""
    start_sec:    float = 0.0
    end_sec:      float = 0.0
    severity:     str  = "medium"   # low | medium | high | critical


@dataclass
class ComplianceResult:
    flags:         List[ComplianceFlag]
    passed:        bool    # True if zero flags
    score:         float   # 0.0 (fail) – 1.0 (perfect)
    summary:       str     = ""
    inference_sec: float   = 0.0

    @property
    def critical_flags(self):
        return [f for f in self.flags if f.severity == "critical"]

    @property
    def flag_types(self):
        return list(set(f.flag_type for f in self.flags))


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPLIANCE CHECKER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class ComplianceChecker:
    """
    Checks call transcripts for compliance issues.

    Example:
        checker = ComplianceChecker(config)
        result  = checker.check(nlp_result, full_transcript)
        if not result.passed:
            for flag in result.flags:
                print(flag.flag_type, flag.description)
    """

    # Default PII patterns (overridden by config)
    DEFAULT_PII_PATTERNS = {
        "credit_card": (
            r'\b(?:\d[ \-]?){13,16}\b',
            "critical"
        ),
        "ssn": (
            r'\b\d{3}[-\s]\d{2}[-\s]\d{4}\b',
            "critical"
        ),
        "phone": (
            r'\b(?:\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b',
            "high"
        ),
        "email": (
            r'[a-zA-Z0-9._%+\-]+\s*(?:at|@)\s*[a-zA-Z0-9.\-]+\s*(?:dot|\.)\s*[a-zA-Z]{2,}',
            "medium"
        ),
    }

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()

        self.cfg    = config.get("compliance", {})
        self.logger = get_logger("compliance_checker", config)

        self.required_phrases  = [p.lower() for p in self.cfg.get("required_phrases", [
            "this call may be recorded",
            "for quality and training purposes",
            "verify your identity",
            "how can i help you today",
        ])]

        self.prohibited_phrases = [p.lower() for p in self.cfg.get("prohibited_phrases", [
            "guaranteed",
            "i promise",
            "sue you",
            "legal action will be taken",
            "we will take you to court",
            "pay or else",
        ])]

        # Build PII patterns from config or defaults
        cfg_pii = self.cfg.get("pii_patterns", {})
        self.pii_patterns = {}
        for name, (pattern, severity) in self.DEFAULT_PII_PATTERNS.items():
            cfg_pattern = cfg_pii.get(name, pattern)
            self.pii_patterns[name] = (re.compile(cfg_pattern, re.IGNORECASE), severity)

    # ── Public API ────────────────────────────────────────────────────────────

    def check(self, nlp_result: NLPResult, full_transcript: str = "") -> ComplianceResult:
        """
        Run all compliance checks. Returns ComplianceResult with list of flags.
        """
        t0    = time.perf_counter()
        flags = []

        # If full_transcript not provided, build from turns
        if not full_transcript:
            full_transcript = " ".join(t.text for t in nlp_result.annotated_turns)

        self.logger.info(f"  Compliance check — {len(nlp_result.annotated_turns)} turns")

        # 1. Check required disclosures
        flags += self._check_required_phrases(
            nlp_result.annotated_turns,
            full_transcript
        )

        # 2. Check prohibited language
        flags += self._check_prohibited_phrases(nlp_result.annotated_turns)

        # 3. Check for PII spoken aloud
        flags += self._check_pii(nlp_result.annotated_turns)

        passed = len(flags) == 0
        score  = self._compute_score(flags)
        summary = self._build_summary(flags)

        elapsed = time.perf_counter() - t0
        self.logger.info(
            f"  Done in {elapsed:.3f}s | "
            f"{'PASS ✓' if passed else f'FAIL — {len(flags)} flags'}"
        )

        return ComplianceResult(
            flags=flags,
            passed=passed,
            score=score,
            summary=summary,
            inference_sec=elapsed,
        )

    # ── Check 1: Required phrases ─────────────────────────────────────────────

    def _check_required_phrases(
        self,
        turns: List[AnnotatedTurn],
        full_transcript: str
    ) -> List[ComplianceFlag]:
        """
        Required phrases must appear somewhere in the call transcript.
        Each missing phrase = one MISSING_DISCLOSURE flag.
        """
        flags = []
        transcript_lower = full_transcript.lower()

        # Find the first agent turn for context
        first_agent_turn = next(
            (t for t in turns if t.role == "agent"), None
        )

        for phrase in self.required_phrases:
            if phrase not in transcript_lower:
                flags.append(ComplianceFlag(
                    flag_type="MISSING_DISCLOSURE",
                    description=f"Required phrase not spoken: \"{phrase}\"",
                    matched_text=phrase,
                    speaker=first_agent_turn.speaker if first_agent_turn else "",
                    role="agent",
                    start_sec=first_agent_turn.start_sec if first_agent_turn else 0.0,
                    end_sec=first_agent_turn.end_sec   if first_agent_turn else 0.0,
                    severity="high",
                ))

        return flags

    # ── Check 2: Prohibited phrases ───────────────────────────────────────────

    def _check_prohibited_phrases(
        self,
        turns: List[AnnotatedTurn]
    ) -> List[ComplianceFlag]:
        """
        Scan every turn for prohibited language.
        Each match = one PROHIBITED_LANGUAGE flag with exact timestamp.
        """
        flags = []

        for turn in turns:
            text_lower = turn.text.lower()
            for phrase in self.prohibited_phrases:
                if phrase in text_lower:
                    # Find the match position
                    idx   = text_lower.find(phrase)
                    snip  = turn.text[max(0, idx-20): idx+len(phrase)+20].strip()

                    flags.append(ComplianceFlag(
                        flag_type="PROHIBITED_LANGUAGE",
                        description=f"Prohibited phrase detected: \"{phrase}\"",
                        matched_text=snip,
                        speaker=turn.speaker,
                        role=turn.role,
                        start_sec=turn.start_sec,
                        end_sec=turn.end_sec,
                        severity=self._prohibited_severity(phrase),
                    ))

        return flags

    def _prohibited_severity(self, phrase: str) -> str:
        """Assign severity based on phrase content."""
        critical_keywords = ["sue", "court", "legal action", "pay or else"]
        if any(kw in phrase for kw in critical_keywords):
            return "critical"
        return "high"

    # ── Check 3: PII detection ────────────────────────────────────────────────

    def _check_pii(self, turns: List[AnnotatedTurn]) -> List[ComplianceFlag]:
        """
        Scan every turn for PII spoken aloud (card numbers, SSN, etc.).
        """
        flags = []

        for turn in turns:
            for pii_type, (pattern, severity) in self.pii_patterns.items():
                matches = pattern.findall(turn.text)
                for match in matches:
                    # Redact for the flag description
                    redacted = self._redact(match)
                    flags.append(ComplianceFlag(
                        flag_type="PII_DETECTED",
                        description=f"Possible {pii_type.replace('_', ' ')} spoken aloud",
                        matched_text=redacted,
                        speaker=turn.speaker,
                        role=turn.role,
                        start_sec=turn.start_sec,
                        end_sec=turn.end_sec,
                        severity=severity,
                    ))

        return flags

    @staticmethod
    def _redact(text: str) -> str:
        """Replace all but last 4 characters with *."""
        digits = re.sub(r'\D', '', text)
        if len(digits) > 4:
            return '*' * (len(digits) - 4) + digits[-4:]
        return '****'

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _compute_score(self, flags: List[ComplianceFlag]) -> float:
        """
        Score from 0.0 to 1.0.
        Deduct points per flag severity:
          critical → -0.30
          high     → -0.15
          medium   → -0.08
          low      → -0.03
        """
        deductions = {"critical": 0.30, "high": 0.15, "medium": 0.08, "low": 0.03}
        score = 1.0
        for flag in flags:
            score -= deductions.get(flag.severity, 0.10)
        return max(0.0, round(score, 2))

    def _build_summary(self, flags: List[ComplianceFlag]) -> str:
        if not flags:
            return "PASS — No compliance issues detected."
        type_counts = {}
        for f in flags:
            type_counts[f.flag_type] = type_counts.get(f.flag_type, 0) + 1
        parts = [f"{count}× {ftype}" for ftype, count in type_counts.items()]
        return "FAIL — " + ", ".join(parts)

    def format_report(self, result: ComplianceResult) -> str:
        """Human-readable compliance report."""
        lines = [
            f"Compliance: {'✓ PASS' if result.passed else '✗ FAIL'}",
            f"Score     : {result.score:.0%}",
            f"Summary   : {result.summary}",
        ]
        if result.flags:
            lines.append("\nFlags:")
            for f in result.flags:
                ts = f"[{f.start_sec:.1f}s–{f.end_sec:.1f}s]"
                lines.append(
                    f"  [{f.severity.upper():8}] {f.flag_type:25} {ts:18} "
                    f"{f.role:8} | {f.description}"
                )
        return "\n".join(lines)


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

    cfg     = load_config()
    proc    = AudioPreprocessor(cfg)
    trans   = Transcriber(cfg)
    diar    = Diarizer(cfg)
    nlp     = NLPAnalyzer(cfg)
    checker = ComplianceChecker(cfg)

    files = sorted(glob.glob("data/raw/synth/*.mp3"))[:3]
    if not files:
        print("No synth files found.")
        sys.exit(0)

    for f in files:
        print(f"\n{'═'*60}\nFile: {Path(f).name}")
        audio      = proc.preprocess(f)
        transcript = trans.transcribe(audio)
        diarized   = diar.diarize(f, transcript, audio.duration_sec)
        nlp_result = nlp.analyze(diarized)
        compliance = checker.check(nlp_result)

        print(checker.format_report(compliance))
