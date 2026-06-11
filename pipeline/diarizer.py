"""
pipeline/diarizer.py  —  Module 3
────────────────────────────────────────────────────────────
Speaker diarization using pyannote.audio 3.1.
Aligns diarization output with Whisper word timestamps to produce
a speaker-attributed transcript (list of SpeakerTurn objects).

Input  : audio file path + TranscriptionResult
Output : list of SpeakerTurn with speaker labels + text + timestamps

Fallback: if pyannote is unavailable or HF token is missing,
          returns all segments under a single "SPEAKER_00" label
          so the rest of the pipeline still works.
"""

import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

from pipeline.utils import load_config, get_logger
from pipeline.transcriber import TranscriptionResult, Segment


# ═══════════════════════════════════════════════════════════════════════════════
#  OUTPUT DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DiarizedTurn:
    """A single speaker turn with aligned transcript text."""
    speaker:   str           # e.g. "SPEAKER_00", "SPEAKER_01"
    role:      str           # "agent" | "customer" | "unknown"
    text:      str
    start_sec: float
    end_sec:   float
    segments:  list = field(default_factory=list)   # original Segment objects


@dataclass
class DiarizationResult:
    turns:        List[DiarizedTurn]
    num_speakers: int
    speaker_map:  dict                # {"SPEAKER_00": "agent", ...}
    method:       str                 # "pyannote" | "fallback"
    duration_sec: float
    inference_sec: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  DIARIZER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class Diarizer:
    """
    Wraps pyannote.audio for speaker diarization.
    Aligns diarization output with Whisper timestamps.

    Example:
        diarizer = Diarizer(config)
        result   = diarizer.diarize(audio_path, transcription)
        for turn in result.turns:
            print(f"{turn.role.upper():10} [{turn.start_sec:.1f}s]: {turn.text}")
    """

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()

        self.cfg    = config.get("diarization", {})
        self.logger = get_logger("diarizer", config)

        self.enabled      = self.cfg.get("enabled",      True)
        self.model_name   = self.cfg.get("model",        "pyannote/speaker-diarization-3.1")
        self.num_speakers = self.cfg.get("num_speakers", 2)   # None = auto-detect
        self.min_speakers = self.cfg.get("min_speakers", 1)
        self.max_speakers = self.cfg.get("max_speakers", 4)
        self.hf_token     = self.cfg.get("hf_token",     "")

        self._pipeline = None   # lazy-loaded

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_pipeline(self) -> bool:
        """
        Load pyannote diarization pipeline.
        Returns True if successful, False if unavailable (triggers fallback).
        """
        if self._pipeline is not None:
            return True

        if not self.enabled:
            return False

        if not self.hf_token:
            self.logger.warning(
                "No HuggingFace token set for pyannote. "
                "Set 'hf_token' in config.yaml or export HF_TOKEN=... "
                "Using single-speaker fallback."
            )
            return False

        try:
            from pyannote.audio import Pipeline
            self.logger.info(f"Loading pyannote pipeline: {self.model_name}")
            t0 = time.perf_counter()
            self._pipeline = Pipeline.from_pretrained(
                self.model_name,
                use_auth_token=self.hf_token,
            )
            self.logger.info(f"  Loaded in {time.perf_counter()-t0:.1f}s")
            return True

        except ImportError:
            self.logger.warning("pyannote.audio not installed. pip install pyannote.audio")
            return False
        except Exception as e:
            self.logger.warning(f"Failed to load pyannote pipeline: {e}")
            return False

    # ── Public API ────────────────────────────────────────────────────────────

    def diarize(
        self,
        audio_path: str,
        transcription: TranscriptionResult,
        duration_sec: float = 0.0,
    ) -> DiarizationResult:
        """
        Run speaker diarization on audio_path and align with transcription.
        Falls back to single-speaker mode if pyannote is unavailable.
        """
        t0 = time.perf_counter()
        self.logger.info(f"  Diarizing: {Path(audio_path).name}")

        has_pyannote = self._load_pipeline()

        if not has_pyannote:
            result = self._fallback_diarization(transcription, duration_sec)
            result.inference_sec = time.perf_counter() - t0
            return result

        # Run pyannote
        try:
            raw_turns = self._run_pyannote(audio_path)
        except Exception as e:
            self.logger.error(f"  Pyannote failed: {e}. Using fallback.")
            result = self._fallback_diarization(transcription, duration_sec)
            result.inference_sec = time.perf_counter() - t0
            return result

        # Align with Whisper segments
        diarized_turns = self._align(raw_turns, transcription.segments)

        # Infer agent vs customer (heuristic: first speaker to talk is agent)
        speaker_map = self._infer_roles(diarized_turns)

        # Apply roles
        for turn in diarized_turns:
            turn.role = speaker_map.get(turn.speaker, "unknown")

        elapsed = time.perf_counter() - t0
        self.logger.info(
            f"  Done in {elapsed:.1f}s | {len(diarized_turns)} turns | "
            f"{len(set(t.speaker for t in diarized_turns))} speakers"
        )

        return DiarizationResult(
            turns=diarized_turns,
            num_speakers=len(set(t.speaker for t in diarized_turns)),
            speaker_map=speaker_map,
            method="pyannote",
            duration_sec=duration_sec or transcription.duration_sec,
            inference_sec=elapsed,
        )

    # ── Pyannote runner ───────────────────────────────────────────────────────

    def _run_pyannote(self, audio_path: str) -> list:
        """
        Run pyannote diarization.
        Returns list of {"speaker": str, "start": float, "end": float}.
        """
        kwargs = {
            "min_speakers": self.min_speakers,
            "max_speakers": self.max_speakers,
        }
        if self.num_speakers:
            kwargs["num_speakers"] = self.num_speakers

        diarization = self._pipeline(audio_path, **kwargs)

        turns = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            turns.append({
                "speaker": speaker,
                "start":   segment.start,
                "end":     segment.end,
            })
        return turns

    # ── Alignment ─────────────────────────────────────────────────────────────

    def _align(self, diar_turns: list, whisper_segments: List[Segment]) -> List[DiarizedTurn]:
        """
        Assign each Whisper segment to the speaker with maximum overlap.
        Then merge consecutive same-speaker segments into turns.
        """
        assigned = []   # (segment, speaker_label)

        for seg in whisper_segments:
            best_speaker = self._find_best_speaker(seg.start_sec, seg.end_sec, diar_turns)
            assigned.append((seg, best_speaker))

        # Merge consecutive segments from the same speaker
        merged_turns: List[DiarizedTurn] = []
        for seg, speaker in assigned:
            if merged_turns and merged_turns[-1].speaker == speaker:
                # Extend current turn
                merged_turns[-1].text    += " " + seg.text
                merged_turns[-1].end_sec  = seg.end_sec
                merged_turns[-1].segments.append(seg)
            else:
                merged_turns.append(DiarizedTurn(
                    speaker=speaker,
                    role="unknown",
                    text=seg.text,
                    start_sec=seg.start_sec,
                    end_sec=seg.end_sec,
                    segments=[seg],
                ))

        return merged_turns

    def _find_best_speaker(self, seg_start: float, seg_end: float, diar_turns: list) -> str:
        """
        For a given time window, find the diarization speaker with maximum overlap.
        Falls back to "SPEAKER_00" if no overlap found.
        """
        best_speaker = "SPEAKER_00"
        best_overlap = 0.0

        for turn in diar_turns:
            overlap_start = max(seg_start, turn["start"])
            overlap_end   = min(seg_end,   turn["end"])
            overlap       = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]

        return best_speaker

    # ── Role inference ────────────────────────────────────────────────────────

    def _infer_roles(self, turns: List[DiarizedTurn]) -> dict:
        """
        Heuristic: the first speaker to talk is the agent (they answer calls).
        All others are customers. Works for typical 2-speaker call recordings.
        """
        if not turns:
            return {}

        speakers_in_order = []
        for t in turns:
            if t.speaker not in speakers_in_order:
                speakers_in_order.append(t.speaker)

        roles = {}
        for i, spk in enumerate(speakers_in_order):
            roles[spk] = "agent" if i == 0 else "customer"

        return roles

    # ── Fallback ──────────────────────────────────────────────────────────────

    def _fallback_diarization(
        self,
        transcription: TranscriptionResult,
        duration_sec: float,
    ) -> DiarizationResult:
        """
        When pyannote is unavailable, put all segments under SPEAKER_00.
        Pipeline continues normally downstream.
        """
        self.logger.info("  Using single-speaker fallback (no diarization)")

        turns = []
        for seg in transcription.segments:
            if turns and turns[-1].speaker == "SPEAKER_00":
                turns[-1].text   += " " + seg.text
                turns[-1].end_sec = seg.end_sec
                turns[-1].segments.append(seg)
            else:
                turns.append(DiarizedTurn(
                    speaker="SPEAKER_00",
                    role="unknown",
                    text=seg.text,
                    start_sec=seg.start_sec,
                    end_sec=seg.end_sec,
                    segments=[seg],
                ))

        return DiarizationResult(
            turns=turns,
            num_speakers=1,
            speaker_map={"SPEAKER_00": "unknown"},
            method="fallback",
            duration_sec=duration_sec or transcription.duration_sec,
        )

    def format_transcript(self, result: DiarizationResult) -> str:
        """
        Format diarized turns as a readable transcript string.
        """
        lines = []
        for turn in result.turns:
            role  = turn.role.upper().ljust(10)
            start = f"{turn.start_sec:6.1f}s"
            lines.append(f"[{start}] {role}: {turn.text.strip()}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import glob, sys
    from pipeline.audio_preprocessor import AudioPreprocessor
    from pipeline.transcriber import Transcriber

    cfg          = load_config()
    preprocessor = AudioPreprocessor(cfg)
    transcriber  = Transcriber(cfg)
    diarizer     = Diarizer(cfg)

    files = sorted(glob.glob("data/raw/synth/*.mp3"))[:2]
    if not files:
        print("No synth files found.")
        sys.exit(0)

    for f in files:
        print(f"\n{'═'*60}")
        print(f"File: {Path(f).name}")
        audio    = preprocessor.preprocess(f)
        transcript = transcriber.transcribe(audio)
        diar     = diarizer.diarize(f, transcript, audio.duration_sec)

        print(f"Method  : {diar.method}")
        print(f"Speakers: {diar.num_speakers}")
        print(f"Roles   : {diar.speaker_map}")
        print(f"\nTranscript:\n{diarizer.format_transcript(diar)[:600]}")
