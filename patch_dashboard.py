"""
patch_dashboard.py
────────────────────────────────────────────────────────────
Ce fichier montre les 2 fonctions à remplacer dans dashboard_v2.py.

INSTRUCTIONS :
  1. Ouvre dashboard_v2.py
  2. Trouve la fonction _display_call_result  (~ligne 350)
  3. Remplace-la par la version ci-dessous
  4. Ajoute l'import en haut du fichier

C'est le seul changement nécessaire — les 2 tabs (Upload + Simulate)
appellent tous les deux _display_call_result, donc un seul patch suffit.
"""

# ════════════════════════════════════════════════════════════════════════
#  ÉTAPE 1 — Ajouter cet import en haut de dashboard_v2.py
#  (après les autres imports, vers la ligne 15)
# ════════════════════════════════════════════════════════════════════════

IMPORT_TO_ADD = """
from transcript_player import show_transcript_player
"""

# ════════════════════════════════════════════════════════════════════════
#  ÉTAPE 2 — Remplacer la fonction _display_call_result
#  dans dashboard_v2.py par celle-ci :
# ════════════════════════════════════════════════════════════════════════

NEW_DISPLAY_FUNCTION = '''
def _display_call_result(record, elapsed: float, audio_bytes: bytes = None,
                         transcript_obj=None, diarization_obj=None):
    """
    Affiche un CallRecord avec player interactif + transcript AVANT
    les resultats NLP/compliance.
    """
    st.success(f"Analyse complete en {elapsed:.1f}s")
    st.markdown("---")

    # ── SECTION 1 : PLAYER INTERACTIF + TRANSCRIPT ────────────────────
    if audio_bytes:
        # Recupere les segments depuis le TranscriptionResult
        segments = []
        if transcript_obj and hasattr(transcript_obj, "segments"):
            segments = transcript_obj.segments
        elif transcript_obj and isinstance(transcript_obj, dict):
            segments = transcript_obj.get("segments", [])

        # Recupere les turns depuis la diarisation
        turns = []
        if diarization_obj and hasattr(diarization_obj, "turns"):
            turns = diarization_obj.turns
        elif diarization_obj and isinstance(diarization_obj, dict):
            turns = diarization_obj.get("turns", [])

        show_transcript_player(
            audio_bytes=audio_bytes,
            transcript_text=record.full_transcript,
            segments=segments,
            turns=turns,
        )
    else:
        # Fallback : affichage texte simple si pas d audio
        st.subheader("Transcription")
        st.text_area("", value=record.full_transcript, height=130,
                     disabled=True, label_visibility="collapsed")

    st.markdown("---")

    # ── SECTION 2 : KPIs ──────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Duree",      f"{record.duration_sec:.1f}s")
    with c2: st.metric("Sentiment",  record.overall_sentiment)
    with c3: st.metric("Intent",     record.primary_intent.replace("_"," ").title())
    with c4:
        status = "PASS" if record.compliance_passed else "FAIL"
        st.metric("Compliance", status)

    st.markdown("---")

    # ── SECTION 3 : COMPLIANCE FLAGS ──────────────────────────────────
    st.subheader("Compliance")
    if not record.compliance_flags:
        st.success("Aucun probleme de compliance detecte.")
    else:
        for flag in record.compliance_flags:
            severity = flag.get("severity", "medium")
            icon = {"critical": "🔴", "high": "🟠",
                    "medium": "🟡", "low": "🟢"}.get(severity, "🟡")
            st.markdown(
                f"{icon} **{flag.get('flag_type','')}** "
                f"— {flag.get('description','')} "
                f"`{flag.get('matched_text','')}`"
            )

    # ── SECTION 4 : ENTITES ───────────────────────────────────────────
    if record.all_entities:
        st.subheader("Entites detectees")
        import pandas as pd
        st.dataframe(pd.DataFrame(record.all_entities),
                     use_container_width=True)

    # ── SECTION 5 : DOWNLOAD ──────────────────────────────────────────
    st.download_button(
        "Download JSON Report",
        data=record.to_json().encode("utf-8"),
        file_name=f"{record.call_id}.json",
        mime="application/json"
    )
'''


# ════════════════════════════════════════════════════════════════════════
#  ÉTAPE 3 — Patcher _run_pipeline_on_file pour retourner
#  transcript_obj et diarization_obj
# ════════════════════════════════════════════════════════════════════════

NEW_RUN_PIPELINE = '''
def _run_pipeline_on_file(audio_path: str, agent_id: str, model_size: str):
    """Run the pipeline and return record + raw objects for the player."""
    try:
        import time
        from pipeline.utils import load_config, generate_call_id
        from pipeline import (
            AudioPreprocessor, Transcriber, Diarizer,
            NLPAnalyzer, ComplianceChecker, ReportBuilder
        )

        cfg = load_config("config.yaml")
        cfg["whisper"]["model"]       = model_size
        cfg["diarization"]["enabled"] = False

        call_id    = generate_call_id(audio_path)
        t0         = time.perf_counter()

        proc       = AudioPreprocessor(cfg)
        trans      = Transcriber(cfg)
        diar       = Diarizer(cfg)
        nlp        = NLPAnalyzer(cfg)
        checker    = ComplianceChecker(cfg)
        builder    = ReportBuilder(cfg)

        audio      = proc.preprocess(audio_path)
        transcript = trans.transcribe(audio)
        diarized   = diar.diarize(audio_path, transcript, audio.duration_sec)
        nlp_result = nlp.analyze(diarized)
        compliance = checker.check(nlp_result, transcript.text)
        record     = builder.build(call_id, audio, transcript,
                                   diarized, nlp_result, compliance,
                                   agent_id=agent_id)
        builder.export(record)
        elapsed    = time.perf_counter() - t0

        # Read audio bytes for the player
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        return {
            "record":       record,
            "elapsed":      elapsed,
            "success":      True,
            "audio_bytes":  audio_bytes,
            "transcript":   transcript,    # <-- NEW
            "diarization":  diarized,      # <-- NEW
        }

    except Exception as e:
        import streamlit as st
        st.error(f"Pipeline error: {e}")
        return None
'''


# ════════════════════════════════════════════════════════════════════════
#  ÉTAPE 4 — Dans tab_upload(), remplacer l'appel de _display_call_result
# ════════════════════════════════════════════════════════════════════════

OLD_CALL = """
        _display_call_result(record, result.get("elapsed", 0))
"""

NEW_CALL = """
        _display_call_result(
            record,
            result.get("elapsed", 0),
            audio_bytes=result.get("audio_bytes"),
            transcript_obj=result.get("transcript"),
            diarization_obj=result.get("diarization"),
        )
"""


# ════════════════════════════════════════════════════════════════════════
#  ÉTAPE 5 — Dans _run_simulation(), meme remplacement
# ════════════════════════════════════════════════════════════════════════

OLD_SIM_CALL = """
    _display_call_result(record, elapsed)
"""

NEW_SIM_CALL = """
    # Re-read audio for the player (was written to tmp_file)
    try:
        from gtts import gTTS
        import io
        buf = io.BytesIO()
        gTTS(text=dialogue, lang="en", slow=False).write_to_fp(buf)
        sim_audio_bytes = buf.getvalue()
    except Exception:
        sim_audio_bytes = None

    _display_call_result(
        record,
        elapsed,
        audio_bytes=sim_audio_bytes,
        transcript_obj=transcript,
        diarization_obj=diarized,
    )
"""


if __name__ == "__main__":
    print("Ce fichier est un guide de patch, pas un script a executer.")
    print("Suis les etapes ETAPE 1 a 5 pour patcher dashboard_v2.py")
    print()
    print("Etapes:")
    print("  1. Ajouter import transcript_player")
    print("  2. Remplacer _display_call_result")
    print("  3. Remplacer _run_pipeline_on_file")
    print("  4. Patcher tab_upload() appel")
    print("  5. Patcher _run_simulation() appel")
