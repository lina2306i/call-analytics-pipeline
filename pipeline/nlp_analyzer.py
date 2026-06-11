"""
pipeline/nlp_analyzer.py  —  Module 4
────────────────────────────────────────────────────────────
NLP analysis on diarized transcript turns.

Per turn it runs:
  1. Sentiment analysis   — DistilBERT SST-2 (POSITIVE / NEGATIVE)
  2. Intent classification — zero-shot via MNLI (no fine-tuning needed to start)
  3. Named Entity Recognition — spaCy en_core_web_sm

Input  : DiarizationResult (from diarizer.py)
Output : NLPResult with per-turn annotations + call-level aggregates
"""

import time
from dataclasses import dataclass, field
from typing import Optional, List

from pipeline.utils import load_config, get_logger
from pipeline.diarizer import DiarizationResult, DiarizedTurn


# ═══════════════════════════════════════════════════════════════════════════════
#  OUTPUT DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AnnotatedTurn:
    """A DiarizedTurn enriched with NLP annotations."""
    speaker:           str
    role:              str
    text:              str
    start_sec:         float
    end_sec:           float
    # sentiment
    sentiment_label:   str   = ""     # POSITIVE | NEGATIVE | NEUTRAL
    sentiment_score:   float = 0.0
    # intent
    intent_label:      str   = ""
    intent_score:      float = 0.0
    # entities
    entities:          list  = field(default_factory=list)  # [(text, label), ...]


@dataclass
class NLPResult:
    annotated_turns:        List[AnnotatedTurn]
    # call-level aggregates
    overall_sentiment:      str   = ""
    overall_sentiment_score:float = 0.0
    customer_sentiment:     str   = ""     # sentiment of customer turns only
    primary_intent:         str   = ""
    primary_intent_score:   float = 0.0
    all_intents:            list  = field(default_factory=list)
    all_entities:           list  = field(default_factory=list)
    inference_sec:          float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  NLP ANALYZER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class NLPAnalyzer:
    """
    Runs sentiment, intent, and NER on each speaker turn.
    All models are loaded lazily and cached for reuse across calls.

    Example:
        analyzer = NLPAnalyzer(config)
        result   = analyzer.analyze(diarization_result)
        print(result.primary_intent, result.customer_sentiment)
    """

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()

        self.cfg    = config.get("nlp", {})
        self.logger = get_logger("nlp_analyzer", config)

        self.spacy_model      = self.cfg.get("spacy_model",   "en_core_web_sm")
        self.sentiment_model  = self.cfg.get("sentiment_model",
                                    "distilbert-base-uncased-finetuned-sst-2-english")
        self.intent_model     = self.cfg.get("intent_model",
                                    "typeform/distilbert-base-uncased-mnli")
        self.intent_labels    = self.cfg.get("intent_labels", [
            "billing dispute", "refund request", "technical support",
            "account cancellation", "payment issue", "product inquiry",
            "escalation request", "complaint", "general inquiry",
        ])
        self.intent_threshold = self.cfg.get("intent_threshold", 0.4)

        # Lazy-loaded models
        self._nlp       = None   # spaCy
        self._sentiment = None   # HF pipeline
        self._intent    = None   # HF zero-shot pipeline

    # ── Model loaders ─────────────────────────────────────────────────────────

    def _load_spacy(self):
        if self._nlp is not None:
            return
        try:
            import spacy
            self.logger.info(f"Loading spaCy model: {self.spacy_model}")
            self._nlp = spacy.load(self.spacy_model)
        except OSError:
            self.logger.warning(
                f"spaCy model '{self.spacy_model}' not found. "
                f"Run: python -m spacy download {self.spacy_model}"
            )
            self._nlp = None
        except ImportError:
            self.logger.warning("spaCy not installed. pip install spacy")
            self._nlp = None

    def _load_sentiment(self):
        if self._sentiment is not None:
            return
        try:
            from transformers import pipeline
            self.logger.info(f"Loading sentiment model: {self.sentiment_model}")
            self._sentiment = pipeline(
                "sentiment-analysis",
                model=self.sentiment_model,
                truncation=True,
                max_length=512,
            )
        except Exception as e:
            self.logger.warning(f"Sentiment model failed to load: {e}")
            self._sentiment = None

    def _load_intent(self):
        if self._intent is not None:
            return
        try:
            from transformers import pipeline
            self.logger.info(f"Loading intent model: {self.intent_model}")
            self._intent = pipeline(
                "zero-shot-classification",
                model=self.intent_model,
            )
        except Exception as e:
            self.logger.warning(f"Intent model failed to load: {e}")
            self._intent = None

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, diarization: DiarizationResult) -> NLPResult:
        """
        Run full NLP analysis on all turns in a DiarizationResult.
        """
        t0 = time.perf_counter()
        self.logger.info(f"  NLP analysis — {len(diarization.turns)} turns")

        # Load models once
        self._load_spacy()
        self._load_sentiment()
        self._load_intent()

        annotated_turns = []
        for turn in diarization.turns:
            ann = self._analyze_turn(turn)
            annotated_turns.append(ann)

        result = self._aggregate(annotated_turns)
        result.inference_sec = time.perf_counter() - t0

        self.logger.info(
            f"  Done in {result.inference_sec:.1f}s | "
            f"sentiment={result.overall_sentiment} | "
            f"intent={result.primary_intent}"
        )
        return result

    # ── Per-turn analysis ─────────────────────────────────────────────────────

    def _analyze_turn(self, turn: DiarizedTurn) -> AnnotatedTurn:
        text = turn.text.strip()

        ann = AnnotatedTurn(
            speaker=turn.speaker,
            role=turn.role,
            text=text,
            start_sec=turn.start_sec,
            end_sec=turn.end_sec,
        )

        if not text:
            return ann

        # 1. Sentiment
        sent_label, sent_score = self._run_sentiment(text)
        ann.sentiment_label = sent_label
        ann.sentiment_score = sent_score

        # 2. Intent (only for customer turns; or all if single-speaker)
        if turn.role in ("customer", "unknown"):
            intent_label, intent_score = self._run_intent(text)
            ann.intent_label = intent_label
            ann.intent_score = intent_score

        # 3. NER
        ann.entities = self._run_ner(text)

        return ann

    # ── Sentiment ─────────────────────────────────────────────────────────────

    def _run_sentiment(self, text: str):
        """
        Returns (label, score).
        Label: POSITIVE | NEGATIVE | NEUTRAL (neutral = low confidence)
        """
        if self._sentiment is None:
            return "NEUTRAL", 0.5

        try:
            out = self._sentiment(text[:512])[0]
            label = out["label"].upper()
            score = round(out["score"], 4)

            # Remap to NEUTRAL when confidence is low
            if score < 0.65:
                label = "NEUTRAL"

            return label, score
        except Exception as e:
            self.logger.debug(f"Sentiment failed: {e}")
            return "NEUTRAL", 0.5

    # ── Intent ────────────────────────────────────────────────────────────────

    def _run_intent(self, text: str):
        """
        Zero-shot intent classification.
        Returns (label, score) or ("unknown", 0.0) if below threshold.
        """
        if self._intent is None:
            return "unknown", 0.0

        try:
            out   = self._intent(text[:512], candidate_labels=self.intent_labels)
            label = out["labels"][0]
            score = round(out["scores"][0], 4)

            if score < self.intent_threshold:
                return "unknown", score

            # Normalize label (replace spaces with underscores)
            label = label.replace(" ", "_")
            return label, score
        except Exception as e:
            self.logger.debug(f"Intent failed: {e}")
            return "unknown", 0.0

    # ── NER ───────────────────────────────────────────────────────────────────

    def _run_ner(self, text: str) -> list:
        """
        Run spaCy NER. Returns list of (entity_text, entity_label) tuples.
        Common labels: PERSON, ORG, MONEY, DATE, CARDINAL, PRODUCT
        """
        if self._nlp is None:
            return []

        try:
            doc = self._nlp(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            return entities
        except Exception as e:
            self.logger.debug(f"NER failed: {e}")
            return []

    # ── Call-level aggregation ────────────────────────────────────────────────

    def _aggregate(self, turns: List[AnnotatedTurn]) -> NLPResult:
        """
        Compute call-level sentiment, primary intent, and entity list
        from all annotated turns.
        """
        if not turns:
            return NLPResult(annotated_turns=[])

        # ── Overall sentiment ─────────────────────────────────────────────
        # Weighted average of all turn scores
        scores = [t.sentiment_score for t in turns if t.sentiment_label]
        labels = [t.sentiment_label for t in turns if t.sentiment_label]

        overall_label, overall_score = self._majority_sentiment(labels, scores)

        # Customer-only sentiment
        cust_turns = [t for t in turns if t.role == "customer"]
        if cust_turns:
            c_labels = [t.sentiment_label for t in cust_turns if t.sentiment_label]
            c_scores = [t.sentiment_score for t in cust_turns if t.sentiment_label]
            cust_sentiment, _ = self._majority_sentiment(c_labels, c_scores)
        else:
            cust_sentiment = overall_label

        # ── Primary intent ────────────────────────────────────────────────
        # Take the highest-confidence intent across all customer turns
        intent_turns = [(t.intent_label, t.intent_score)
                        for t in turns if t.intent_label and t.intent_label != "unknown"]
        if intent_turns:
            intent_turns.sort(key=lambda x: x[1], reverse=True)
            primary_intent       = intent_turns[0][0]
            primary_intent_score = intent_turns[0][1]
            all_intents          = list(dict.fromkeys([i[0] for i in intent_turns]))
        else:
            primary_intent       = "unknown"
            primary_intent_score = 0.0
            all_intents          = []

        # ── All entities (deduplicated) ───────────────────────────────────
        seen     = set()
        entities = []
        for t in turns:
            for ent in t.entities:
                key = (ent[0].lower(), ent[1])
                if key not in seen:
                    seen.add(key)
                    entities.append({"text": ent[0], "label": ent[1]})

        return NLPResult(
            annotated_turns=turns,
            overall_sentiment=overall_label,
            overall_sentiment_score=overall_score,
            customer_sentiment=cust_sentiment,
            primary_intent=primary_intent,
            primary_intent_score=primary_intent_score,
            all_intents=all_intents,
            all_entities=entities,
        )

    def _majority_sentiment(self, labels: list, scores: list):
        """Return the most common sentiment label and its mean score."""
        if not labels:
            return "NEUTRAL", 0.5

        counts = {}
        for lbl in labels:
            counts[lbl] = counts.get(lbl, 0) + 1
        majority = max(counts, key=counts.get)

        mean_score = round(
            sum(s for l, s in zip(labels, scores) if l == majority) /
            max(counts[majority], 1), 4
        )
        return majority, mean_score


# ═══════════════════════════════════════════════════════════════════════════════
#  QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import glob, sys
    from pipeline.audio_preprocessor import AudioPreprocessor
    from pipeline.transcriber import Transcriber
    from pipeline.diarizer import Diarizer

    cfg   = load_config()
    proc  = AudioPreprocessor(cfg)
    trans = Transcriber(cfg)
    diar  = Diarizer(cfg)
    nlp   = NLPAnalyzer(cfg)

    files = sorted(glob.glob("data/raw/synth/*.mp3"))[:2]
    if not files:
        print("No synth files found.")
        sys.exit(0)

    for f in files:
        from pathlib import Path
        print(f"\n{'═'*60}\nFile: {Path(f).name}")
        audio     = proc.preprocess(f)
        transcript = trans.transcribe(audio)
        diarized  = diar.diarize(f, transcript, audio.duration_sec)
        result    = nlp.analyze(diarized)

        print(f"Overall sentiment : {result.overall_sentiment} ({result.overall_sentiment_score:.2f})")
        print(f"Customer sentiment: {result.customer_sentiment}")
        print(f"Primary intent    : {result.primary_intent} ({result.primary_intent_score:.2f})")
        print(f"Entities          : {result.all_entities[:5]}")
        print(f"\nPer-turn:")
        for t in result.annotated_turns:
            print(f"  [{t.role:8}] {t.sentiment_label:8} | {t.intent_label:25} | {t.text[:60]}")
