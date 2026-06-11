"""
dashboard_v2.py  —  Upgraded Call Analytics Dashboard
────────────────────────────────────────────────────────────
4 tabs:
  Tab 1 — Call Analytics   (KPIs, compliance, intents, call log)
  Tab 2 — ASR Performance  (WER/CER charts, model comparison)
  Tab 3 — Upload & Analyze (drop an MP3 → instant pipeline output)
  Tab 4 — Simulate Call    (pick a scenario → generate + analyze live + transcript_player)

Run:
    streamlit run dashboard_v3.py
"""

import io
import json
import time
import sqlite3
import tempfile
from pathlib import Path
from datetime import date, timedelta
from transcript_player import show_transcript_player
try:
    import streamlit as st
    import pandas as pd
    import numpy as np
except ImportError:
    raise SystemExit("Run: pip install streamlit pandas numpy")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Call Analytics — ASR Pipeline",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DB_PATH        = "reports/calls.db"
REPORTS_DIR    = Path("reports/calls")
EVAL_DIR       = Path("evaluation")
SYNTH_DIR      = Path("data/raw/synth")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.kpi-card {
    background:#1e2433; border:1px solid #30363d; border-radius:8px;
    padding:18px 20px; text-align:center; margin-bottom:8px;
}
.kpi-label { font-size:11px; letter-spacing:.12em; text-transform:uppercase;
             color:#8b949e; margin-bottom:4px; }
.kpi-value { font-size:30px; font-weight:700; color:#e6edf3; }
.kpi-sub   { font-size:12px; color:#3fb950; margin-top:4px; }
.pass { color:#3fb950; font-weight:600; }
.fail { color:#f78166; font-weight:600; }
.tag  { background:#21262d; border:1px solid #30363d; border-radius:4px;
        padding:2px 8px; font-size:11px; font-family:monospace; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=30)
def load_calls() -> pd.DataFrame:
    if not Path(DB_PATH).exists():
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT * FROM calls ORDER BY processed_at DESC", conn)
    conn.close()
    if not df.empty:
        df["call_date"]         = pd.to_datetime(df["call_date"])
        df["processed_at"]      = pd.to_datetime(df["processed_at"])
        df["compliance_passed"] = df["compliance_passed"].astype(bool)
    return df


@st.cache_data(ttl=60)
def load_eval_results() -> pd.DataFrame:
    csv = EVAL_DIR / "wer_results.csv"
    if csv.exists():
        return pd.read_csv(csv)
    return pd.DataFrame()


@st.cache_data(ttl=60)
def load_benchmark() -> pd.DataFrame:
    csv = EVAL_DIR / "benchmark_results.csv"
    if csv.exists():
        return pd.read_csv(csv)
    return pd.DataFrame()


@st.cache_data(ttl=60)
def load_flags() -> pd.DataFrame:
    flags = []
    if not REPORTS_DIR.exists():
        return pd.DataFrame()
    for jf in REPORTS_DIR.glob("*.json"):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            for f in data.get("compliance_flags", []):
                f["call_id"]  = data["call_id"]
                f["agent_id"] = data.get("agent_id", "")
                flags.append(f)
        except Exception:
            pass
    return pd.DataFrame(flags) if flags else pd.DataFrame(
        columns=["call_id","flag_type","severity","description","agent_id"])


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — CALL ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

def tab_analytics():
    df       = load_calls()
    flags_df = load_flags()

    if df.empty:
        st.info("No data yet. Run the pipeline first:\n"
                "```\npython pipeline.py --input data/raw/synth/ --no-diarize\n```")
        return

    # ── Sidebar filters ───────────────────────────────────────────────────────
    st.sidebar.markdown("### Filters")

    min_date = df["call_date"].min().date()
    max_date = df["call_date"].max().date()

    if min_date == max_date:
        st.sidebar.info(f"Date: {min_date}")
        date_range = (min_date, max_date)
    else:
        default_start = max(min_date, max_date - timedelta(days=7))
        date_range = st.sidebar.date_input(
            "Date range",
            value=(default_start, max_date),
            min_value=min_date,
            max_value=max_date,
        )

    agents  = ["All"] + sorted(df["agent_id"].dropna().unique().tolist())
    agent   = st.sidebar.selectbox("Agent", agents)
    intents = ["All"] + sorted(df["primary_intent"].dropna().unique().tolist())
    intent  = st.sidebar.selectbox("Intent", intents)
    comp    = st.sidebar.radio("Compliance", ["All","Pass only","Fail only"])

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Total records: {len(df)}")

    # Apply filters
    fdf = df.copy()
    if len(date_range) == 2:
        fdf = fdf[(fdf["call_date"].dt.date >= date_range[0]) &
                  (fdf["call_date"].dt.date <= date_range[1])]
    if agent  != "All": fdf = fdf[fdf["agent_id"]       == agent]
    if intent != "All": fdf = fdf[fdf["primary_intent"] == intent]
    if comp == "Pass only": fdf = fdf[fdf["compliance_passed"] == True]
    if comp == "Fail only": fdf = fdf[fdf["compliance_passed"] == False]

    # ── KPIs ──────────────────────────────────────────────────────────────────
    if fdf.empty:
        st.warning("No records match the filters.")
        return

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.metric("Total Calls",      len(fdf))
    with c2: st.metric("Compliance Rate",  f"{fdf['compliance_passed'].mean():.0%}",
                       delta=f"{fdf['compliance_passed'].sum()} passed")
    with c3: st.metric("Top Intent",
                       fdf["primary_intent"].mode()[0].replace("_"," ").title()
                       if not fdf["primary_intent"].isna().all() else "—")
    with c4: st.metric("Avg Duration",     f"{fdf['duration_sec'].mean():.0f}s")
    with c5: st.metric("Total Flags",      int(fdf["num_flags"].sum()),
                       delta=f"{(~fdf['compliance_passed']).sum()} failed",
                       delta_color="inverse")

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Intent Distribution")
        ic = fdf["primary_intent"].value_counts()
        ic.index = [i.replace("_"," ").title() for i in ic.index]
        st.bar_chart(pd.DataFrame({"count": ic}))

    with col2:
        st.subheader("Overall Sentiment")
        sc = fdf["overall_sentiment"].value_counts()
        st.bar_chart(pd.DataFrame({"count": sc}))

    st.subheader("Compliance Pass Rate Over Time")
    daily = (fdf.set_index("call_date")
               .resample("D")["compliance_passed"]
               .agg(["sum","count"])
               .rename(columns={"sum":"passed","count":"total"}))
    daily["pass_rate"] = daily["passed"] / daily["total"].replace(0,1)
    st.line_chart(daily["pass_rate"])

    if not flags_df.empty:
        st.subheader("Compliance Flags by Type")
        fc = flags_df["flag_type"].value_counts()
        st.bar_chart(pd.DataFrame({"count": fc}))

    # ── Call Log ──────────────────────────────────────────────────────────────
    st.subheader("Call Log")
    search = st.text_input("Search call ID or intent…", "")
    disp   = fdf.copy()
    if search:
        mask = (disp["call_id"].str.contains(search, case=False, na=False) |
                disp["primary_intent"].str.contains(search, case=False, na=False))
        disp = disp[mask]

    disp["compliance_passed"] = disp["compliance_passed"].map(
        {True:"✓ PASS", False:"✗ FAIL"})
    cols = ["call_id","call_date","agent_id","duration_sec",
            "overall_sentiment","primary_intent","compliance_passed","num_flags"]
    st.dataframe(disp[[c for c in cols if c in disp.columns]].head(100),
                 use_container_width=True, height=380)
    st.download_button("⬇ Download CSV",
                       data=disp.to_csv(index=False).encode("utf-8"),
                       file_name=f"calls_{date.today()}.csv",
                       mime="text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — ASR PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════

def tab_asr_performance():
    st.header("ASR Performance — WER / CER / RTF")

    eval_df  = load_eval_results()
    bench_df = load_benchmark()

    if eval_df.empty and bench_df.empty:
        st.info(
            "No evaluation data yet. Run:\n\n"
            "```powershell\n"
            "# Basic WER evaluation\n"
            "python evaluate.py\n\n"
            "# Model benchmark (tiny vs base vs small)\n"
            "python benchmark_models.py --models tiny base small\n"
            "```"
        )
        return

    # ── WER Results ───────────────────────────────────────────────────────────
    if not eval_df.empty:
        st.subheader("WER / CER per Call")

        mean_wer = eval_df["wer"].mean()
        mean_cer = eval_df["cer"].mean()
        mean_rtf = eval_df["rtf"].mean()

        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("Mean WER",
                           f"{mean_wer:.2%}",
                           delta="PASS" if mean_wer < 0.15 else "FAIL — above 15% target",
                           delta_color="normal" if mean_wer < 0.15 else "inverse")
        with c2: st.metric("Mean CER",   f"{mean_cer:.2%}")
        with c3: st.metric("Mean RTF",   f"{mean_rtf:.2f}x")
        with c4: st.metric("Files",      len(eval_df))

        st.markdown("---")

        # WER per scenario
        st.subheader("WER by Scenario")
        by_scenario = (eval_df.groupby("scenario")["wer"]
                               .mean()
                               .sort_values()
                               .reset_index())
        by_scenario.columns = ["scenario","mean_wer"]
        by_scenario["scenario"] = by_scenario["scenario"].str.replace("_"," ").str.title()
        st.bar_chart(by_scenario.set_index("scenario"))

        # WER distribution
        st.subheader("WER Distribution per Call")
        wer_chart = eval_df[["filename","wer","cer"]].copy()
        wer_chart["filename"] = wer_chart["filename"].str[:30]
        st.bar_chart(wer_chart.set_index("filename")[["wer","cer"]])

        # Raw table
        with st.expander("See full WER table"):
            disp = eval_df[["filename","scenario","model","wer","cer",
                             "rtf","inference_sec","duration_sec"]].copy()
            disp["wer"] = disp["wer"].map("{:.2%}".format)
            disp["cer"] = disp["cer"].map("{:.2%}".format)
            st.dataframe(disp, use_container_width=True)

        # Download
        st.download_button(
            "⬇ Download WER CSV",
            data=eval_df.to_csv(index=False).encode("utf-8"),
            file_name="wer_results.csv",
            mime="text/csv"
        )

    # ── Benchmark Results ─────────────────────────────────────────────────────
    if not bench_df.empty:
        st.markdown("---")
        st.subheader("Model Benchmark — Accuracy vs Speed")

        summary = bench_df.groupby("model").agg(
            mean_wer=("wer","mean"),
            mean_cer=("cer","mean"),
            mean_rtf=("rtf","mean"),
            mean_inf=("inference_sec","mean"),
        ).round(4).reset_index()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**WER by Model (lower = better)**")
            st.bar_chart(summary.set_index("model")[["mean_wer"]])
        with c2:
            st.markdown("**RTF by Model (lower = faster)**")
            st.bar_chart(summary.set_index("model")[["mean_rtf"]])

        # Summary table
        st.markdown("**Full Comparison Table**")
        disp = summary.copy()
        disp["mean_wer"] = disp["mean_wer"].map("{:.2%}".format)
        disp["mean_cer"] = disp["mean_cer"].map("{:.2%}".format)
        disp["mean_rtf"] = disp["mean_rtf"].map("{:.2f}x".format)
        disp["mean_inf"] = disp["mean_inf"].map("{:.1f}s".format)
        disp.columns = ["Model","WER","CER","RTF","Avg Inf Time"]
        st.dataframe(disp, use_container_width=True)

        # Recommendation
        best = summary.loc[summary["mean_wer"].idxmin(), "model"]
        fastest = summary.loc[summary["mean_rtf"].idxmin(), "model"]
        st.success(
            f"**Best accuracy:** `whisper-{best}` "
            f"(WER={summary[summary['model']==best]['mean_wer'].values[0]:.2%})  |  "
            f"**Fastest:** `whisper-{fastest}` "
            f"(RTF={summary[summary['model']==fastest]['mean_rtf'].values[0]:.2f}x)"
        )

    # ── Denoise Impact ────────────────────────────────────────────────────────
    denoise_csv = EVAL_DIR / "denoise_comparison.csv"
    if denoise_csv.exists():
        st.markdown("---")
        st.subheader("Impact of Denoising on WER")
        dc = pd.read_csv(denoise_csv)
        if not dc.empty:
            pivot = dc.pivot_table(values="wer", index="filename",
                                   columns="condition", aggfunc="mean")
            st.bar_chart(pivot)
            by_cond = dc.groupby("condition")[["wer","cer"]].mean()
            st.dataframe(by_cond.style.format("{:.2%}"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — UPLOAD & ANALYZE
# ═══════════════════════════════════════════════════════════════════════════════

def tab_upload():
    st.header("Upload & Analyze a Call Recording")
    st.markdown(
        "Upload any `.mp3` or `.wav` file — the pipeline will transcribe it, "
        "run NLP, check compliance, and show the full analysis below."
    )

    uploaded = st.file_uploader(
        "Drop your audio file here",
        type=["mp3","wav","flac","ogg","m4a"],
        help="Supported formats: MP3, WAV, FLAC, OGG, M4A"
    )

    col1, col2 = st.columns(2)
    with col1:
        agent_id = st.text_input("Agent ID (optional)", placeholder="e.g. A-117")
    with col2:
        model_size = st.selectbox("Whisper model", ["base","tiny","small","medium"],
                                   index=0)

    if uploaded is None:
        st.info("Upload a file to start analysis.")
        return

    # Save to temp file
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.audio(uploaded)

    if st.button("Run Pipeline", type="primary", use_container_width=True):
        with st.spinner("Running pipeline… (this takes 10-30s depending on model)"):
            result = _run_pipeline_on_file(tmp_path, agent_id, model_size)

        if result is None:
            st.error("Pipeline failed. Check terminal for errors.")
            return

        record = result.get("record")
        if record is None:
            st.error("No record returned from pipeline.")
            return

        _display_call_result(
            record,
            result.get("elapsed", 0),
            audio_bytes=result.get("audio_bytes"),
            transcript_obj=result.get("transcript"),
            diarization_obj=result.get("diarization"),
        )
        #_display_call_result(record, result.get("elapsed", 0))


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
    
# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — SIMULATE CALL
# ═══════════════════════════════════════════════════════════════════════════════

SCENARIOS = {
    "Billing Dispute":         "billing_dispute",
    "Refund Request":          "refund_request",
    "Technical Support":       "tech_support",
    "Account Cancellation":    "cancellation",
    "Payment Issue":           "payment_issue",
    "General Inquiry":         "general_inquiry",
    "Complaint":               "complaint",
    "Escalation Request":      "tech_support_escalation",
}

CUSTOM_DIALOGUES = {
    "billing_dispute": (
        "Agent: Thank you for calling, this call may be recorded. How can I help? "
        "Customer: I have a double charge on my account from last week. "
        "Agent: I can see that. Let me verify your identity first. "
        "Customer: Sure, my name is Sarah and my account ends in 4821. "
        "Agent: Thank you. I can confirm the duplicate charge and will process a refund."
    ),
    "refund_request": (
        "Agent: Thank you for calling. This call may be recorded for quality. "
        "Customer: I want a refund for a product I bought last week. It stopped working. "
        "Agent: I understand. Can you give me your order number? "
        "Customer: It is order nine nine three four five seven. "
        "Agent: Thank you. I have initiated a full refund of sixty five dollars."
    ),
    "tech_support": (
        "Agent: Thank you for calling support. How can I help you today? "
        "Customer: My internet keeps dropping every few minutes since yesterday. "
        "Agent: I am sorry about that. Let me run a diagnostic on your router. "
        "Customer: Okay it seems more stable now after the reset. "
        "Agent: The connection looks stable. Please call back if the issue returns."
    ),
    "cancellation": (
        "Agent: Thank you for calling. This call may be recorded "
        "for quality and training purposes. How can I help you today? "
        "Customer: I would like to cancel my subscription please. "
        "Agent: I am sorry to hear that. Let me verify your identity first. "
        "Can you confirm your email address? "
        "Customer: Sure it is sarah at example dot com. "
        "Agent: Thank you Sarah. I have processed the cancellation. "
        "Your service remains active until end of billing cycle. "
        "Customer: Thank you goodbye."
    ),
    "payment_issue": (
        "Agent: Thank you for calling. This call may be recorded "
        "for quality and training purposes. How can I help you today? "
        "Customer: My payment keeps failing. I have tried three times already. "
        "Agent: I am sorry to hear that. Let me verify your identity first. "
        "Can you confirm your full name and account number? "
        "Customer: My name is James and my account ends in seven seven three two. "
        "Agent: Thank you James. To keep things secure I will send you a payment link. "
        "Customer: I see the email, using it now. "
        "Agent: Great, your payment should process within a few minutes. "
        "Is there anything else I can help you with today? "
        "Customer: No that is all, thank you very much."
    ),
    "general_inquiry": (
        "Agent: Thank you for calling. This call may be recorded "
        "for quality and training purposes. How can I help you today? "
        "Customer: I just want to know what plans you currently offer. "
        "Agent: Of course. We have three plans. "
        "Basic at nine ninety nine, Standard at nineteen ninety nine, "
        "and Premium at thirty four ninety nine per month. "
        "Customer: What is included in the standard plan? "
        "Agent: Standard includes fifty gigabytes of storage and email support. "
        "Customer: Perfect I will go with that one please. "
        "Agent: Excellent. Let me verify your identity and sign you up now."
    ),
    "complaint": (
        "Agent: Customer support, how can I help? "
        "Customer: I want to make a formal complaint. Your agent last week was rude. "
        "Agent: I sincerely apologize for that experience. Can you tell me more? "
        "Customer: I called about billing and the agent dismissed my concerns. "
        "Agent: I will file a formal complaint. A supervisor will contact you today."
    ),
    "tech_support_escalation": (
        "Agent: Thank you for calling, how can I help? "
        "Customer: I have called three times about the same issue. Nobody fixed it. "
        "I want to speak to a manager right now. "
        "Agent: I completely understand your frustration. Let me escalate this. "
        "Customer: I have been a customer for ten years and this is unacceptable. "
        "Agent: Absolutely, I will transfer you to a senior agent immediately."
    ),
}


def tab_simulate():
    st.header("Simulate a Call — Live Analysis")
    st.markdown(
        "Choose a scenario, generate a synthetic call with gTTS, "
        "and run the full pipeline in real time."
    )

    col1, col2 = st.columns(2)
    with col1:
        scenario_label = st.selectbox("Call Scenario", list(SCENARIOS.keys()))
        model_size     = st.selectbox("Whisper model", ["base","tiny","small"], index=0)
    with col2:
        agent_id       = st.text_input("Agent ID", value="SIM-001")
        custom_text    = st.text_area(
            "Custom dialogue (optional — leave blank to use template)",
            height=100,
            placeholder="Type your own call script here…"
        )

    # Show the dialogue that will be used
    scenario_key = SCENARIOS[scenario_label]
    dialogue     = custom_text.strip() if custom_text.strip() \
                   else CUSTOM_DIALOGUES.get(scenario_key, "Hello, how can I help you today?")

    with st.expander("Preview dialogue that will be synthesized"):
        st.text(dialogue)

    if st.button("Generate & Analyze Call", type="primary", use_container_width=True):
        _run_simulation(dialogue, scenario_key, agent_id, model_size)


def _run_simulation(dialogue: str, scenario: str, agent_id: str, model_size: str):
    """Generate audio from text and run the full pipeline."""

    progress = st.progress(0, text="Step 1/5 — Generating audio with gTTS…")

    # Step 1 — Generate audio
    try:
        from gtts import gTTS
        import tempfile, os

        tts      = gTTS(text=dialogue, lang="en", slow=False)
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp_file.name)
        audio_path = tmp_file.name
        progress.progress(20, text="Step 2/5 — Audio generated. Preprocessing…")
    except Exception as e:
        st.error(f"gTTS failed: {e}. Run: pip install gTTS")
        return

    # Step 2-5 — Pipeline
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

        call_id = generate_call_id(audio_path)
        t0      = time.perf_counter()

        progress.progress(30, text="Step 3/5 — Transcribing with Whisper…")
        proc       = AudioPreprocessor(cfg)
        trans      = Transcriber(cfg)
        audio      = proc.preprocess(audio_path)
        transcript = trans.transcribe(audio)

        progress.progress(55, text="Step 4/5 — Running NLP analysis…")
        diar       = Diarizer(cfg)
        nlp        = NLPAnalyzer(cfg)
        checker    = ComplianceChecker(cfg)
        diarized   = diar.diarize(audio_path, transcript, audio.duration_sec)
        nlp_result = nlp.analyze(diarized)
        compliance = checker.check(nlp_result, transcript.text)

        progress.progress(80, text="Step 5/5 — Building report…")
        builder    = ReportBuilder(cfg)
        record     = builder.build(call_id, audio, transcript,
                                   diarized, nlp_result, compliance,
                                   agent_id=agent_id)
        builder.export(record)
        elapsed    = time.perf_counter() - t0

        progress.progress(100, text="Done!")
        time.sleep(0.3)
        progress.empty()

    except Exception as e:
        st.error(f"Pipeline error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return
    finally:
        try:
            import os
            os.unlink(audio_path)
        except Exception:
            pass

    # ── Display results ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"Results — {scenario.replace('_',' ').title()} Scenario")

    # Play the synthesized audio (re-generate for playback)
    try:
        from gtts import gTTS
        import io
        tts_play = gTTS(text=dialogue, lang="en", slow=False)
        buf      = io.BytesIO()
        tts_play.write_to_fp(buf)
        buf.seek(0)
        st.audio(buf, format="audio/mp3")
    except Exception:
        pass

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
    #_display_call_result(record, elapsed)

    # WER against dialogue template (ground truth = our script)
    try:
        import re
        from jiwer import wer as compute_wer, cer as compute_cer

        def norm(t):
            t = t.lower()
            t = re.sub(r"[^\w\s]", " ", t)
            return re.sub(r"\s+", " ", t).strip()

        ref = norm(dialogue)
        hyp = norm(record.full_transcript)
        w   = compute_wer(ref, hyp)
        c   = compute_cer(ref, hyp)

        st.markdown("---")
        st.subheader("ASR Accuracy (vs synthesized script)")
        m1, m2, m3 = st.columns(3)
        with m1: st.metric("WER",  f"{w:.2%}",
                            delta="PASS" if w < 0.15 else "Above 15% target",
                            delta_color="normal" if w < 0.15 else "inverse")
        with m2: st.metric("CER",  f"{c:.2%}")
        with m3: st.metric("Model", f"whisper-{model_size}")
    except ImportError:
        st.info("Install jiwer for WER: `pip install jiwer`")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.title("📞 Call Analytics — ASR Pipeline")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Call Analytics",
        "🎯 ASR Performance",
        "📁 Upload & Analyze",
        "🎙️ Simulate Call",
    ])

    with tab1: tab_analytics()
    with tab2: tab_asr_performance()
    with tab3: tab_upload()
    with tab4: tab_simulate()


if __name__ == "__main__":
    main()
