"""
pipeline/transcriber.py  —  Module 2
────────────────────────────────────────────────────────────
Wraps OpenAI Whisper (or faster-whisper) for transcription.

Input  : PreprocessedAudio (from audio_preprocessor.py)
Output : TranscriptionResult(text, segments, language, model_used, duration)

Features:
  • Supports whisper  (openai-whisper)
  • Supports faster-whisper (CTranslate2, 4× faster on CPU)
  • File-based caching — skips already-transcribed audio
  • Word-level timestamps for diarization alignment
  • WER / CER evaluation against a reference transcript
"""

import json
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from pipeline.utils import load_config, get_logger
from pipeline.audio_preprocessor import PreprocessedAudio

# ═══════════════════════════════════════════════════════════════════════════════
#  OUTPUT DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WordToken:
    word:      str
    start_sec: float
    end_sec:   float
    probability: float = 1.0


@dataclass
class Segment:
    segment_id:  int
    text:        str
    start_sec:   float
    end_sec:     float
    avg_logprob: float = 0.0
    words:       list  = field(default_factory=list)  # list of WordToken


@dataclass
class TranscriptionResult:
    text:          str               # full transcript (joined segments)
    segments:      list              # list of Segment
    language:      str               # detected or forced language
    model_used:    str               # e.g. "whisper-medium"
    duration_sec:  float
    inference_sec: float             # wall-clock time for transcription
    audio_hash:    str               # links back to PreprocessedAudio
    from_cache:    bool = False
    warnings:      list = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
#  TRANSCRIBER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class Transcriber:
    """
    Transcribes preprocessed audio using Whisper or faster-whisper.

    Example:
        transcriber = Transcriber(config)
        result = transcriber.transcribe(preprocessed_audio)
        print(result.text)
    """

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()

        self.cfg        = config.get("whisper", {})
        self.logger     = get_logger("transcriber", config)

        self.model_size      = self.cfg.get("model",            "base")
        self.language        = self.cfg.get("language",         "en")
        self.word_timestamps = self.cfg.get("word_timestamps",  True)
        self.task            = self.cfg.get("task",             "transcribe")
        self.device          = self.cfg.get("device",           "cpu")
        self.use_faster      = self.cfg.get("use_faster",       False)
        self.compute_type    = self.cfg.get("compute_type",     "int8")

        # Cache directory (avoids re-transcribing same file)
        self.cache_dir = Path(config.get("paths", {}).get("processed", "data/processed")) / "transcripts"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._model = None   # lazy-loaded on first use

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self):
        if self._model is not None:
            return

        self.logger.info(f"Loading {'faster-whisper' if self.use_faster else 'whisper'} model: {self.model_size}")
        t0 = time.perf_counter()

        if self.use_faster:
            try:
                from faster_whisper import WhisperModel
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                )
                self._backend = "faster-whisper"
            except ImportError:
                self.logger.warning("faster-whisper not installed, falling back to openai-whisper")
                self._load_openai_whisper()
        else:
            self._load_openai_whisper()

        self.logger.info(f"  Model loaded in {time.perf_counter()-t0:.1f}s  [{self._backend}]")

    def _load_openai_whisper(self):
        try:
            import whisper
            self._model   = whisper.load_model(self.model_size, device=self.device)
            self._backend = f"whisper-{self.model_size}"
        except ImportError:
            raise ImportError("openai-whisper not installed. Run: pip install openai-whisper")

    # ── Public API ────────────────────────────────────────────────────────────

    def transcribe(self, audio: PreprocessedAudio) -> TranscriptionResult:
        """
        Transcribe a PreprocessedAudio object.
        Checks cache first; loads model on first call.
        """
        # Check cache
        cached = self._load_cache(audio.file_hash)
        if cached:
            self.logger.info(f"  Cache hit → {Path(audio.file_path).name}")
            return cached

        self._load_model()

        t0 = time.perf_counter()
        self.logger.info(f"  Transcribing: {Path(audio.file_path).name}")

        if self._backend.startswith("faster"):
            result = self._run_faster_whisper(audio)
        else:
            result = self._run_openai_whisper(audio)

        result.inference_sec = time.perf_counter() - t0
        rtf = result.inference_sec / max(audio.duration_sec, 0.001)
        self.logger.info(
            f"  Done in {result.inference_sec:.1f}s | "
            f"RTF={rtf:.2f}x | lang={result.language} | "
            f"{len(result.segments)} segments"
        )

        # Cache result
        self._save_cache(audio.file_hash, result)
        return result

    def transcribe_batch(self, audio_list: list) -> list:
        """Transcribe a list of PreprocessedAudio. Returns list of TranscriptionResult (None on error)."""
        results = []
        for i, audio in enumerate(audio_list, 1):
            if audio is None:
                results.append(None)
                continue
            self.logger.info(f"[{i}/{len(audio_list)}] {Path(audio.file_path).name}")
            try:
                results.append(self.transcribe(audio))
            except Exception as e:
                self.logger.error(f"  Transcription failed: {e}")
                results.append(None)
        return results

    # ── Backend runners ───────────────────────────────────────────────────────

    def _run_openai_whisper(self, audio: PreprocessedAudio) -> TranscriptionResult:
        """Run openai-whisper on the audio array."""
        options = {
            "language":        self.language or None,
            "task":            self.task,
            "word_timestamps": self.word_timestamps,
            "verbose":         False,
        }
        raw = self._model.transcribe(audio.audio, **options)

        segments = []
        for i, seg in enumerate(raw.get("segments", [])):
            words = []
            if self.word_timestamps:
                for w in seg.get("words", []):
                    words.append(WordToken(
                        word=w["word"],
                        start_sec=w["start"],
                        end_sec=w["end"],
                        probability=w.get("probability", 1.0),
                    ))
            segments.append(Segment(
                segment_id=i,
                text=seg["text"].strip(),
                start_sec=seg["start"],
                end_sec=seg["end"],
                avg_logprob=seg.get("avg_logprob", 0.0),
                words=words,
            ))

        return TranscriptionResult(
            text=raw["text"].strip(),
            segments=segments,
            language=raw.get("language", self.language or ""),
            model_used=self._backend,
            duration_sec=audio.duration_sec,
            inference_sec=0.0,
            audio_hash=audio.file_hash,
        )

    def _run_faster_whisper(self, audio: PreprocessedAudio) -> TranscriptionResult:
        """Run faster-whisper on the audio array."""
        segs_iter, info = self._model.transcribe(
            audio.audio,
            language=self.language or None,
            task=self.task,
            word_timestamps=self.word_timestamps,
            beam_size=5,
        )

        segments = []
        full_text_parts = []
        for i, seg in enumerate(segs_iter):
            words = []
            if self.word_timestamps and seg.words:
                for w in seg.words:
                    words.append(WordToken(
                        word=w.word,
                        start_sec=w.start,
                        end_sec=w.end,
                        probability=w.probability,
                    ))
            segments.append(Segment(
                segment_id=i,
                text=seg.text.strip(),
                start_sec=seg.start,
                end_sec=seg.end,
                avg_logprob=seg.avg_logprob,
                words=words,
            ))
            full_text_parts.append(seg.text.strip())

        return TranscriptionResult(
            text=" ".join(full_text_parts),
            segments=segments,
            language=info.language,
            model_used=f"faster-whisper-{self.model_size}",
            duration_sec=audio.duration_sec,
            inference_sec=0.0,
            audio_hash=audio.file_hash,
        )

    # ── Caching ───────────────────────────────────────────────────────────────

    def _cache_path(self, file_hash: str) -> Path:
        return self.cache_dir / f"{file_hash}.json"

    def _save_cache(self, file_hash: str, result: TranscriptionResult):
        try:
            path = self._cache_path(file_hash)
            # Convert to dict manually (dataclasses with nested objects)
            data = {
                "text":         result.text,
                "language":     result.language,
                "model_used":   result.model_used,
                "duration_sec": result.duration_sec,
                "inference_sec":result.inference_sec,
                "audio_hash":   result.audio_hash,
                "segments": [
                    {
                        "segment_id":  s.segment_id,
                        "text":        s.text,
                        "start_sec":   s.start_sec,
                        "end_sec":     s.end_sec,
                        "avg_logprob": s.avg_logprob,
                        "words": [
                            {"word": w.word, "start_sec": w.start_sec,
                             "end_sec": w.end_sec, "probability": w.probability}
                            for w in s.words
                        ],
                    }
                    for s in result.segments
                ],
            }
            with open(path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            self.logger.warning(f"  Cache write failed: {e}")

    def _load_cache(self, file_hash: str) -> Optional[TranscriptionResult]:
        path = self._cache_path(file_hash)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)

            segments = []
            for s in data.get("segments", []):
                words = [
                    WordToken(w["word"], w["start_sec"], w["end_sec"], w.get("probability", 1.0))
                    for w in s.get("words", [])
                ]
                segments.append(Segment(
                    segment_id=s["segment_id"], text=s["text"],
                    start_sec=s["start_sec"], end_sec=s["end_sec"],
                    avg_logprob=s.get("avg_logprob", 0.0), words=words,
                ))
            return TranscriptionResult(
                text=data["text"], segments=segments,
                language=data["language"], model_used=data["model_used"],
                duration_sec=data["duration_sec"], inference_sec=data["inference_sec"],
                audio_hash=data["audio_hash"], from_cache=True,
            )
        except Exception as e:
            self.logger.warning(f"  Cache read failed: {e}")
            return None

    def clear_cache(self):
        """Delete all cached transcripts."""
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
        self.logger.info("Transcript cache cleared.")

    # ── Evaluation ────────────────────────────────────────────────────────────

    def compute_wer(self, hypothesis: str, reference: str) -> dict:
        """
        Compute Word Error Rate (WER) and Character Error Rate (CER).
        Requires: pip install jiwer
        """
        try:
            from jiwer import wer, cer
            return {
                "wer": round(wer(reference, hypothesis), 4),
                "cer": round(cer(reference, hypothesis), 4),
            }
        except ImportError:
            self.logger.warning("jiwer not installed — WER unavailable. pip install jiwer")
            return {"wer": None, "cer": None}


# ═══════════════════════════════════════════════════════════════════════════════
#  QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import glob, sys
    from pipeline.audio_preprocessor import AudioPreprocessor

    cfg          = load_config()
    preprocessor = AudioPreprocessor(cfg)
    transcriber  = Transcriber(cfg)

    files = sorted(glob.glob("data/raw/synth/*.mp3"))[:2]
    if not files:
        print("No synth files found. Run generate_synthetic_calls.py first.")
        sys.exit(0)

    for f in files:
        audio  = preprocessor.preprocess(f)
        result = transcriber.transcribe(audio)
        print(f"\n{'─'*60}")
        print(f"File     : {Path(f).name}")
        print(f"Language : {result.language}")
        print(f"Model    : {result.model_used}")
        print(f"Duration : {result.duration_sec:.1f}s")
        print(f"Inf.time : {result.inference_sec:.1f}s")
        print(f"Transcript:\n  {result.text[:200]}...")
