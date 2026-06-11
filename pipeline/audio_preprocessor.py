"""
pipeline/audio_preprocessor.py  —  Module 1
────────────────────────────────────────────────────────────
Handles all audio ingestion and preprocessing.

Input  : path to any audio file (.mp3, .wav, .flac, .ogg, .m4a)
Output : PreprocessedAudio(array, sample_rate, duration_sec, metadata)

Responsibilities:
  • Load audio from any common format
  • Resample to 16 kHz mono  (Whisper requirement)
  • Normalize amplitude
  • Optionally apply spectral noise reduction
  • Chunk long calls into overlapping segments
  • Validate audio quality (duration, SNR estimate)
"""

import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import librosa
    import soundfile as sf
except ImportError as e:
    raise ImportError(f"Missing audio library: {e}. Run: pip install librosa soundfile")

try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False

from pipeline.utils import load_config, get_logger

# ═══════════════════════════════════════════════════════════════════════════════
#  OUTPUT DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PreprocessedAudio:
    audio:        np.ndarray        # float32 waveform, shape (N,)
    sample_rate:  int               # always TARGET_SR after preprocessing
    duration_sec: float
    file_path:    str
    file_hash:    str               # md5 of original file, for caching
    original_sr:  int               # sample rate before resampling
    original_channels: int
    snr_estimate: float = 0.0      # rough signal-to-noise ratio in dB
    chunks:       list  = field(default_factory=list)  # if call was chunked
    warnings:     list  = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
#  PREPROCESSOR CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class AudioPreprocessor:
    """
    Stateless audio preprocessor. Load once, call preprocess() per file.

    Example:
        preprocessor = AudioPreprocessor(config)
        result = preprocessor.preprocess("data/raw/synth/call_001.mp3")
        print(result.duration_sec, result.snr_estimate)
    """

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()
        self.cfg    = config.get("audio", {})
        self.logger = get_logger("audio_preprocessor", config)

        self.target_sr       = self.cfg.get("target_sr",        16000)
        self.normalize       = self.cfg.get("normalize",         True)
        self.denoise         = self.cfg.get("denoise",           True)
        self.denoise_strength= self.cfg.get("denoise_strength",  0.85)
        self.max_duration    = self.cfg.get("max_duration_sec",  600)
        self.chunk_overlap   = self.cfg.get("chunk_overlap_sec", 10)

        if self.denoise and not HAS_NOISEREDUCE:
            self.logger.warning("noisereduce not installed — denoising disabled. pip install noisereduce")
            self.denoise = False

    # ── Public API ────────────────────────────────────────────────────────────

    def preprocess(self, audio_path: str) -> PreprocessedAudio:
        """
        Full preprocessing pipeline for a single audio file.
        Returns PreprocessedAudio with a clean 16kHz mono float32 array.
        """
        t0 = time.perf_counter()
        path = Path(audio_path)

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self.logger.info(f"Preprocessing: {path.name}")

        # Step 1 — load
        audio, orig_sr, orig_channels = self._load(str(path))

        # Step 2 — convert to mono
        audio = self._to_mono(audio, orig_channels)

        # Step 3 — resample to target SR
        audio = self._resample(audio, orig_sr)

        # Step 4 — denoise (optional)
        if self.denoise:
            audio = self._denoise(audio)

        # Step 5 — normalize
        if self.normalize:
            audio = self._normalize(audio)

        duration   = len(audio) / self.target_sr
        snr        = self._estimate_snr(audio)
        file_hash  = self._md5(str(path))
        warnings   = self._quality_check(audio, duration)

        # Step 6 — chunk if too long
        chunks = []
        if duration > self.max_duration:
            chunks = self._chunk(audio, duration)
            self.logger.info(f"  Chunked into {len(chunks)} segments (max {self.max_duration}s each)")

        elapsed = time.perf_counter() - t0
        self.logger.info(
            f"  Done in {elapsed:.2f}s | {duration:.1f}s audio | "
            f"SNR≈{snr:.1f}dB | {orig_sr}Hz→{self.target_sr}Hz"
        )

        return PreprocessedAudio(
            audio=audio,
            sample_rate=self.target_sr,
            duration_sec=duration,
            file_path=str(path),
            file_hash=file_hash,
            original_sr=orig_sr,
            original_channels=orig_channels,
            snr_estimate=snr,
            chunks=chunks,
            warnings=warnings,
        )

    def preprocess_batch(self, audio_paths: list) -> list:
        """
        Preprocess multiple files. Skips files with errors, logs them.
        Returns list of PreprocessedAudio (same order as input, None for failures).
        """
        results = []
        for i, path in enumerate(audio_paths, 1):
            self.logger.info(f"[{i}/{len(audio_paths)}] {Path(path).name}")
            try:
                results.append(self.preprocess(path))
            except Exception as e:
                self.logger.error(f"  Failed: {e}")
                results.append(None)
        return results

    # ── Private steps ─────────────────────────────────────────────────────────

    def _load(self, path: str):
        """
        Load audio from any format librosa supports.
        Returns (audio_array, original_sr, original_channels).
        """
        try:
            # Load at native sample rate first so we can log the original
            audio, sr = librosa.load(path, sr=None, mono=False)
        except Exception as e:
            raise RuntimeError(f"Could not load audio '{path}': {e}")

        # Determine channel count
        if audio.ndim == 1:
            channels = 1
        else:
            channels = audio.shape[0]

        return audio, sr, channels

    def _to_mono(self, audio: np.ndarray, channels: int) -> np.ndarray:
        """Average all channels down to mono."""
        if audio.ndim == 2:
            audio = np.mean(audio, axis=0)
        return audio.astype(np.float32)

    def _resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resample to self.target_sr if needed."""
        if orig_sr == self.target_sr:
            return audio
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=self.target_sr)

    def _denoise(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply spectral gating noise reduction.
        Uses the first 0.5s as a noise profile estimate.
        """
        try:
            # Use first 500ms as noise sample
            noise_sample = audio[: int(self.target_sr * 0.5)]
            reduced = nr.reduce_noise(
                y=audio,
                sr=self.target_sr,
                y_noise=noise_sample,
                prop_decrease=self.denoise_strength,
                stationary=False,
            )
            return reduced.astype(np.float32)
        except Exception as e:
            self.logger.warning(f"  Denoising failed ({e}), using raw audio")
            return audio

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """
        Peak normalize to [-1, 1]. Handles silent audio gracefully.
        """
        peak = np.max(np.abs(audio))
        if peak < 1e-6:
            self.logger.warning("  Audio appears silent (peak < 1e-6)")
            return audio
        return audio / peak

    def _estimate_snr(self, audio: np.ndarray) -> float:
        """
        Simple energy-based SNR estimate in dB.
        Assumes the quietest 10% of frames are noise, rest is signal.
        """
        frame_len = self.target_sr // 10  # 100ms frames
        if len(audio) < frame_len * 2:
            return 0.0

        # RMS energy per frame
        frames = librosa.util.frame(audio, frame_length=frame_len, hop_length=frame_len // 2)
        rms    = np.sqrt(np.mean(frames ** 2, axis=0))

        if len(rms) == 0:
            return 0.0

        rms_sorted  = np.sort(rms)
        noise_floor = np.mean(rms_sorted[: max(1, len(rms_sorted) // 10)]) + 1e-10
        signal_rms  = np.mean(rms_sorted[len(rms_sorted) // 2:]) + 1e-10

        snr_db = 20 * np.log10(signal_rms / noise_floor)
        return float(np.clip(snr_db, -10, 60))

    def _chunk(self, audio: np.ndarray, duration: float) -> list:
        """
        Split long audio into overlapping chunks.
        Returns list of (start_sec, end_sec, chunk_array) tuples.
        """
        chunk_samples   = int(self.max_duration   * self.target_sr)
        overlap_samples = int(self.chunk_overlap  * self.target_sr)
        step_samples    = chunk_samples - overlap_samples

        chunks  = []
        start   = 0
        while start < len(audio):
            end   = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]
            chunks.append({
                "start_sec": start / self.target_sr,
                "end_sec":   end   / self.target_sr,
                "audio":     chunk,
            })
            if end == len(audio):
                break
            start += step_samples
        return chunks

    def _quality_check(self, audio: np.ndarray, duration: float) -> list:
        """Return list of warning strings for low-quality audio."""
        warnings = []
        if duration < 2.0:
            warnings.append("Very short audio (< 2 seconds)")
        if duration > self.max_duration:
            warnings.append(f"Long audio ({duration:.0f}s) — will be chunked")
        if np.max(np.abs(audio)) < 0.01:
            warnings.append("Very low amplitude — possible silent recording")
        snr = self._estimate_snr(audio)
        if snr < 5:
            warnings.append(f"Low SNR ({snr:.1f} dB) — noisy recording")
        return warnings

    @staticmethod
    def _md5(file_path: str) -> str:
        h = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def save_processed(self, result: PreprocessedAudio, output_dir: str) -> str:
        """
        Save preprocessed audio as a 16kHz WAV file.
        Returns the output file path.
        """
        out     = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem    = Path(result.file_path).stem
        out_path = out / f"{stem}_16k.wav"
        sf.write(str(out_path), result.audio, self.target_sr)
        self.logger.info(f"  Saved processed audio → {out_path}")
        return str(out_path)


# ═══════════════════════════════════════════════════════════════════════════════
#  QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, glob
    cfg  = load_config()
    proc = AudioPreprocessor(cfg)

    files = glob.glob("data/raw/synth/*.mp3")
    if not files:
        print("No .mp3 files found in data/raw/synth/. Run generate_synthetic_calls.py first.")
        sys.exit(0)

    # Test on first 3 files
    for f in sorted(files)[:3]:
        result = proc.preprocess(f)
        print(f"\n{'─'*50}")
        print(f"  File:     {Path(result.file_path).name}")
        print(f"  Duration: {result.duration_sec:.2f}s")
        print(f"  SNR:      {result.snr_estimate:.1f} dB")
        print(f"  Shape:    {result.audio.shape}")
        print(f"  Warnings: {result.warnings or 'none'}")
