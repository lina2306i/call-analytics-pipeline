# 📞 Call Analytics Pipeline — ASR End-to-End

> Pipeline de reconnaissance vocale automatique (ASR) pour l'analyse d'appels de centres d'appels.
> De l'audio brut aux rapports structurés avec métriques ASR, NLP et conformité.
> End-to-end pipeline: raw audio → transcription → diarization → NLP → compliance → reports.


![Python](https://img.shields.io/badge/Python-3.11-3776ab?logo=python&logoColor=white)
![Whisper](https://img.shields.io/badge/Whisper-OpenAI-412991?logo=openai&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-ee4c2c?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-ff4b4b?logo=streamlit&logoColor=white)
![WER](https://img.shields.io/badge/WER-7.8%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Aperçu

Ce projet implémente la **chaîne complète ASR** :

```
Audio MP3/WAV
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  [1] Audio Preprocessor  — 16kHz, débruitage, SNR       │
│  [2] Whisper Transcriber — texte + word timestamps      │
│  [3] Speaker Diarizer    — Agent vs Customer            │
│  [4] NLP Analyzer        — sentiment + intent + NER     │
│  [5] Compliance Checker  — PII + phrases + flags        │
│  [6] Report Builder      — JSON + CSV + Excel + SQLite  │
└─────────────────────────────────────────────────────────┘
      │
      ▼
Dashboard Streamlit (4 onglets)
```

---

## Métriques ASR obtenues

| Modèle | WER ↓ | CER ↓ | RTF ↓ | Objectif |
|--------|-------|-------|-------|----------|
| whisper-tiny | 12.3% | 4.1% | 0.05x | ✅ < 15% |
| whisper-base | 7.8% | 2.6% | 0.10x | ✅ < 15% |
| whisper-small | 5.2% | 1.8% | 0.22x | ✅ < 15% |

> **RTF** = Real-Time Factor. RTF 0.10x = 1 min d'audio transcrit en 6 secondes.

---

## Installation

### Prérequis

- Python 3.11+
- GPU NVIDIA recommandé (CUDA 12.1) — fonctionne aussi en CPU
- ffmpeg installé et dans le PATH

### Étapes

```bash
# 1. Cloner le repo
git clone https://github.com/lina2306i/call-analytics-pipeline.git
cd call-analytics-pipeline

# 2. Environnement virtuel
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / Mac

# 3. Dépendances
pip install --upgrade pip setuptools wheel
pip install openai-whisper
pip install -r requirements.txt

# 4. Modèle spaCy
python -m spacy download en_core_web_sm

# 5. ffmpeg
winget install ffmpeg            # Windows
# brew install ffmpeg            # Mac
# sudo apt install ffmpeg        # Linux

# 6. Add HuggingFace token to config.yaml (for diarization)
# diarization.hf_token: "hf_..."
# OR: export HF_TOKEN=hf_...
```

---

## Démarrage rapide

```bash
# Générer 22 appels synthétiques de test (gTTS)
python generate_synthetic_calls.py

# Lancer le pipeline sur tous les appels
python pipeline.py --input data/raw/synth/ --no-diarize
# python pipeline.py --input data/raw/synth/ --output reports/

# Évaluer WER/CER/RTF
python evaluate.py

# Benchmark tiny vs base vs small
python benchmark_models.py --models tiny base small --max-files 5

# Dashboard interactif
#streamlit run dashboard_v2.py
streamlit run dashboard_v3.py
# → http://localhost:8501
```

---

## Dashboard — 4 onglets

### 📊 Call Analytics
KPIs temps réel, distribution des intentions, taux de conformité,
sentiment par appel, call log filtrable et téléchargeable.

### 🎯 ASR Performance
WER/CER par appel et par scénario, benchmark comparatif des modèles
Whisper, impact du débruitage sur le WER, RTF.

### 📁 Upload & Analyze
Glisser-déposer un fichier MP3/WAV → pipeline complet en 10-30s.
Player audio interactif avec transcript mot par mot et timestamps
cliquables. Chaque mot est synchronisé avec la position dans l'audio.

### 🎙️ Simulate Call
Choisir un scénario (billing dispute, refund, escalation...) →
gTTS génère l'audio → Whisper transcrit → NLP analyse → WER calculé
vs script de référence. Démo complète sans fichier externe.

---

## Structure du projet

```
call-analytics-pipeline/
│
├── .venv/
├── __pycache__
├── data/
├── reports/
├── evaluation/
├── logs/
│
├── pipeline/
│   ├── __init__.py
│   ├── audio_preprocessor.py   # Signal processing (librosa, noisereduce) :: # Module 1 — load, denoise, resample
│   ├── compliance_checker.py   # Règles + PII regex + scoring :: # Module 5 — rule-based + PII flagging
│   ├── diarizer.py             # pyannote.audio + fallback :: # Module 3 — pyannote speaker separation
│   ├── nlp_analyzer.py         # DistilBERT + spaCy + zero-shot ::  # Module 4 — sentiment, intent, NER
│   ├── report_builder.py       # JSON/CSV/Excel/SQLite :: # Module 6 — JSON / CSV / Excel / SQLite
│   ├── transcriber.py          # Whisper ASR + cache + WER :: # Module 2 — Whisper transcription + cache
│   └── utils.py                # Config, logger, CallRecord dataclass
│
├── pipeline.py                 # Orchestrateur + CLI
├── dashboard_v2.py             # Streamlit UI (4 onglets)
├── dashboard_v3.py             # Streamlit UI (4 onglets)
├── transcript_player.py        # Composant HTML/JS player interactif
│
├── evaluate.py                 # WER/CER/RTF + impact débruitage
├── benchmark_models.py         # Comparaison modèles Whisper
├── record_and_analyze.py       # Simulation CLI
├── generate_synthetic_calls.py # Génération données test (gTTS) ::  # Synthetic data generator
│
├── prepare_intent_data.py      # Download SLURP + MINDS-14
├── train_intent_classifier.py  # Fine-tuning DistilBERT intent
├── update_pipeline_intent.py   # Mise à jour config après training
│
├── config.yaml                 # Toute la configuration ici
├── requirements.txt
└── README.md
```

## CLI Reference

```bash
# Single file
python pipeline.py --input call.mp3 --agent A-117

# Batch directory
python pipeline.py --input data/raw/synth/ --model medium

# Skip diarization (no HF token needed)
python pipeline.py --input data/raw/ --no-diarize

# Generate synthetic data then run
python pipeline.py --input data/raw/synth/ --generate
```

## Module Interfaces

| Module | Input | Output |
|--------|-------|--------|
| AudioPreprocessor | audio file path | `PreprocessedAudio` |
| Transcriber | `PreprocessedAudio` | `TranscriptionResult` |
| Diarizer | audio path + `TranscriptionResult` | `DiarizationResult` |
| NLPAnalyzer | `DiarizationResult` | `NLPResult` |
| ComplianceChecker | `NLPResult` | `ComplianceResult` |
| ReportBuilder | all above | `CallRecord` + files |

## Output Files

```
reports/
├── calls/
│   └── CC-call001-20250519123456.json   # per-call JSON
├── batch_2025-05-19.csv                 # daily batch CSV
├── summary_2025-05-19.xlsx              # Excel workbook (5 sheets)
└── calls.db                             # SQLite database
```

---

## Configuration

All settings live in `config.yaml`. Key options:

```yaml
# config.yaml — toutes les options modifiables ici

whisper:
  model: "base"          # tiny | base | small | medium | large
  language: "en"         # null = détection automatique
  word_timestamps: true
  device: "cpu"          # cpu | cuda

diarization:
  enabled: true
  hf_token: ""           # Gratuit sur huggingface.co
  num_speakers: 2

nlp:
  spacy_model: "en_core_web_sm"
  intent_threshold: 0.4
  intent_labels:
    - "billing dispute"
    - "refund request"
    - "technical support"
    - "account cancellation"
    - "payment issue"
    - "product inquiry"
    - "escalation request"
    - "complaint"
    - "general inquiry"

compliance:
  required_phrases:
    - "this call may be recorded"
    - "verify your identity"
  prohibited_phrases:
    - "guaranteed"
    - "i promise"
    - "legal action"
```

---

## Démarches ASR implémentées

| # | Démarche | Technologie | Fichier |
|---|----------|-------------|---------|
| 1 | Acquisition signal — 16kHz mono | librosa, soundfile | preprocessor.py |
| 2 | Preprocessing — débruitage, normalisation, SNR | noisereduce, numpy | preprocessor.py |
| 3 | Feature extraction — Log-Mel Spectrogram 80D | STFT + filtres Mel (interne Whisper) | transcriber.py |
| 4 | Modèle acoustique — Encoder-Decoder Transformer | OpenAI Whisper | transcriber.py |
| 5 | Décodage + timestamps — word-level | DTW alignment | transcriber.py |
| 6 | Diarisation — qui parle quand | ECAPA-TDNN + clustering | diarizer.py |
| 7 | Évaluation — WER/CER/RTF, benchmark | jiwer | evaluate.py |

---

## Résultats sur données synthétiques (22 appels)

```
Total appels traités  : 22 / 22  (100%)
Compliance PASS       : 2 / 22   (9%)  — scénarios conçus pour échouer
Top intents détectés  : refund_request · payment_issue · billing_dispute
Sentiment dominant    : NEGATIVE (73%)
PII détecté           : 2 appels (numéros de carte parlés à voix haute)
WER moyen             : 7.8%   ✅ objectif < 15%
RTF moyen             : 0.10x  ✅ objectif < 1.0
```

---

## Stack technique

| Catégorie | Technologie |
|-----------|-------------|
| **ASR** | OpenAI Whisper (tiny/base/small/medium) |
| **Diarisation** | pyannote.audio 3.1 |
| **Sentiment** | DistilBERT SST-2 (HuggingFace) |
| **Intent** | Zero-shot MNLI + fine-tuning DistilBERT |
| **NER** | spaCy en_core_web_sm |
| **Audio** | librosa · noisereduce · soundfile |
| **Dashboard** | Streamlit + HTML/JS natif |
| **Rapports** | pandas · openpyxl · SQLite |
| **Deep Learning** | PyTorch + CUDA 12.1 |
| **Évaluation** | jiwer · scikit-learn |
| **Données synthétiques** | gTTS |

---

## Perspectives

- [ ] Mode streaming temps réel (faster-whisper + WebSocket)
- [ ] Fine-tuning Whisper sur dialecte arabe tunisien
- [ ] Analyse émotion acoustique (openSMILE / SpeechBrain)
- [ ] API REST (FastAPI) + déploiement Docker
- [ ] Support multilingue arabe / français

---

## Licence

MIT License — voir [LICENSE](LICENSE)

---

## 👩‍💻 Auteur

**Lina Labiadh** — *Ingénieure Data & AI*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profil-0077b5?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/linalabiadh/)
[![Email](https://img.shields.io/badge/Email-Contact-d14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:lina.hkl2306@gmail.com)
[![Email](https://img.shields.io/badge/Email-Contact-d14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:lina.hkl2306@gmail.com)
