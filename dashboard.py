"""
dashboard.py  —  Streamlit Analytics Dashboard  (Phase 5)
────────────────────────────────────────────────────────────
Reads from reports/calls.db (SQLite) produced by the pipeline.

Run:
    streamlit run dashboard.py

Shows:
    • KPI cards (calls today, pass rate, top intent, avg sentiment)
    • Compliance trend over time
    • Intent distribution bar chart
    • Sentiment breakdown pie
    • Per-agent performance table
    • Call log with search/filter
"""

import json
import sqlite3
from pathlib import Path
from datetime import date, timedelta

try:
    import streamlit as st
    import pandas as pd
except ImportError:
    raise SystemExit("Run: pip install streamlit pandas")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Call Analytics Dashboard",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
DB_PATH      = "reports/calls.db"
REPORTS_DIR  = Path("reports/calls")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #1e2433;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 16px 20px;
    text-align: center;
}
.metric-label { font-size: 11px; letter-spacing: 0.1em;
                text-transform: uppercase; color: #8b949e; margin-bottom: 4px; }
.metric-value { font-size: 28px; font-weight: 700; color: #e6edf3; }
.metric-delta { font-size: 12px; color: #3fb950; margin-top: 4px; }
.pass  { color: #3fb950; font-weight: 600; }
.fail  { color: #f78166; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=30)
def load_data() -> pd.DataFrame:
    """Load all call records from SQLite."""
    if not Path(DB_PATH).exists():
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT * FROM calls ORDER BY processed_at DESC", conn)
    conn.close()
    if not df.empty:
        df["call_date"]     = pd.to_datetime(df["call_date"])
        df["processed_at"]  = pd.to_datetime(df["processed_at"])
        df["compliance_passed"] = df["compliance_passed"].astype(bool)
    return df


@st.cache_data(ttl=30)
def load_flags() -> pd.DataFrame:
    """Load all compliance flags by reading individual call JSON files."""
    flags = []
    for json_file in REPORTS_DIR.glob("*.json"):
        try:
            data = json.loads(json_file.read_text())
            for f in data.get("compliance_flags", []):
                f["call_id"]   = data["call_id"]
                f["call_date"] = data.get("call_date", "")
                f["agent_id"]  = data.get("agent_id", "")
                flags.append(f)
        except Exception:
            pass
    return pd.DataFrame(flags) if flags else pd.DataFrame(
        columns=["call_id","flag_type","severity","description","agent_id","call_date"]
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def sidebar(df: pd.DataFrame):
    st.sidebar.title("📞 Call Analytics")
    st.sidebar.markdown("---")

    if df.empty:
        st.sidebar.info("No data yet. Run the pipeline first.")
        return df, None, None

    # Date range filter
    min_date = df["call_date"].min().date()
    max_date = df["call_date"].max().date()
    # NOUVEAU CODE — gère le cas où min == max
    if min_date == max_date:
        date_range = (min_date, max_date)
        st.sidebar.info(f"Date: {min_date}")
    else:
        default_start = max(min_date, max_date - timedelta(days=7))
        date_range = st.sidebar.date_input(
            "Date range",
            value=(default_start, max_date),
            min_value=min_date,
            max_value=max_date,
        )
    
    
    # Agent filter
    agents = ["All"] + sorted(df["agent_id"].dropna().unique().tolist())
    agent  = st.sidebar.selectbox("Agent", agents)

    # Intent filter
    intents = ["All"] + sorted(df["primary_intent"].dropna().unique().tolist())
    intent  = st.sidebar.selectbox("Intent", intents)

    # Compliance filter
    comp_filter = st.sidebar.radio("Compliance", ["All", "Pass only", "Fail only"])

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Total records: {len(df)}")
    st.sidebar.caption(f"DB: {DB_PATH}")

    # Apply filters
    if len(date_range) == 2:
        df = df[
            (df["call_date"].dt.date >= date_range[0]) &
            (df["call_date"].dt.date <= date_range[1])
        ]
    if agent != "All":
        df = df[df["agent_id"] == agent]
    if intent != "All":
        df = df[df["primary_intent"] == intent]
    if comp_filter == "Pass only":
        df = df[df["compliance_passed"] == True]
    elif comp_filter == "Fail only":
        df = df[df["compliance_passed"] == False]

    return df, agent, comp_filter


# ═══════════════════════════════════════════════════════════════════════════════
#  KPI CARDS
# ═══════════════════════════════════════════════════════════════════════════════

def kpi_cards(df: pd.DataFrame):
    if df.empty:
        st.warning("No records match the current filters.")
        return

    total      = len(df)
    pass_rate  = df["compliance_passed"].mean()
    top_intent = df["primary_intent"].mode()[0] if not df["primary_intent"].isna().all() else "—"
    avg_dur    = df["duration_sec"].mean()
    total_flags = df["num_flags"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total Calls", total)
    with c2:
        st.metric("Compliance Rate", f"{pass_rate:.0%}",
                  delta=f"{df['compliance_passed'].sum()} passed")
    with c3:
        st.metric("Top Intent", top_intent.replace("_", " ").title())
    with c4:
        st.metric("Avg Duration", f"{avg_dur:.0f}s")
    with c5:
        st.metric("Total Flags", int(total_flags),
                  delta=f"{df[df['compliance_passed']==False].shape[0]} failed calls",
                  delta_color="inverse")


# ═══════════════════════════════════════════════════════════════════════════════
#  CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def charts(df: pd.DataFrame, flags_df: pd.DataFrame):
    if df.empty:
        return

    col1, col2 = st.columns(2)

    # ── Intent distribution ───────────────────────────────────────────────────
    with col1:
        st.subheader("Intent Distribution")
        intent_counts = (
            df["primary_intent"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "intent", "primary_intent": "count"})
        )
        # Rename columns safely for different pandas versions
        intent_counts.columns = ["intent", "count"]
        intent_counts["intent"] = intent_counts["intent"].str.replace("_", " ").str.title()
        st.bar_chart(intent_counts.set_index("intent"))

    # ── Sentiment breakdown ───────────────────────────────────────────────────
    with col2:
        st.subheader("Overall Sentiment")
        sent_counts = df["overall_sentiment"].value_counts()
        st.bar_chart(sent_counts)

    # ── Compliance over time ──────────────────────────────────────────────────
    st.subheader("Compliance Pass Rate Over Time")
    daily = (
        df.set_index("call_date")
        .resample("D")["compliance_passed"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "passed", "count": "total"})
    )
    daily["pass_rate"] = daily["passed"] / daily["total"].replace(0, 1)
    if not daily.empty:
        st.line_chart(daily["pass_rate"])

    # ── Flag type breakdown ───────────────────────────────────────────────────
    if not flags_df.empty:
        st.subheader("Compliance Flags by Type")
        flag_counts = flags_df["flag_type"].value_counts()
        st.bar_chart(flag_counts)


def chart(df: pd.DataFrame, flags_df: pd.DataFrame):
    if df.empty:
        return

    col1, col2 = st.columns(2)

    # Intent distribution
    with col1:
        st.subheader("Intent Distribution")
        intent_counts = df["primary_intent"].value_counts()
        intent_df = pd.DataFrame({
            "intent": [i.replace("_", " ").title() for i in intent_counts.index],
            "count":  intent_counts.values
        }).set_index("intent")
        st.bar_chart(intent_df)

    # Sentiment breakdown
    with col2:
        st.subheader("Overall Sentiment")
        sent_df = pd.DataFrame({
            "sentiment": df["overall_sentiment"].value_counts().index,
            "count":     df["overall_sentiment"].value_counts().values
        }).set_index("sentiment")
        st.bar_chart(sent_df)

    # Compliance over time
    st.subheader("Compliance Pass Rate Over Time")
    daily = (
        df.set_index("call_date")
        .resample("D")["compliance_passed"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "passed", "count": "total"})
    )
    daily["pass_rate"] = daily["passed"] / daily["total"].replace(0, 1)
    if not daily.empty:
        st.line_chart(daily["pass_rate"])

    # Flag types
    if not flags_df.empty:
        st.subheader("Compliance Flags by Type")
        flag_df = pd.DataFrame({
            "type":  flags_df["flag_type"].value_counts().index,
            "count": flags_df["flag_type"].value_counts().values
        }).set_index("type")
        st.bar_chart(flag_df)
        
# ═══════════════════════════════════════════════════════════════════════════════
#  AGENT TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def agent_table(df: pd.DataFrame):
    if df.empty or df["agent_id"].isna().all():
        return

    st.subheader("Agent Performance")
    agent_stats = (
        df.groupby("agent_id")
        .agg(
            total_calls=("call_id",         "count"),
            pass_rate  =("compliance_passed","mean"),
            avg_flags  =("num_flags",        "mean"),
            avg_duration=("duration_sec",    "mean"),
            top_intent =("primary_intent",   lambda x: x.mode()[0] if len(x) else "—"),
        )
        .reset_index()
    )
    agent_stats["pass_rate"]    = (agent_stats["pass_rate"]    * 100).round(1).astype(str) + "%"
    agent_stats["avg_flags"]    = agent_stats["avg_flags"].round(2)
    agent_stats["avg_duration"] = agent_stats["avg_duration"].round(0).astype(int).astype(str) + "s"

    st.dataframe(agent_stats, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  CALL LOG TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def call_log(df: pd.DataFrame):
    st.subheader("Call Log")

    search = st.text_input("Search call ID or intent…", "")

    display = df.copy()
    if search:
        mask = (
            display["call_id"].str.contains(search, case=False, na=False) |
            display["primary_intent"].str.contains(search, case=False, na=False)
        )
        display = display[mask]

    display["compliance_passed"] = display["compliance_passed"].map(
        {True: "✓ PASS", False: "✗ FAIL"}
    )

    cols = ["call_id", "call_date", "agent_id", "duration_sec",
            "overall_sentiment", "primary_intent", "compliance_passed", "num_flags"]
    available = [c for c in cols if c in display.columns]

    st.dataframe(
        display[available].head(100),
        use_container_width=True,
        height=400,
    )

    # Download button
    csv = display[available].to_csv(index=False)
    st.download_button(
        "⬇ Download filtered CSV",
        data=csv,
        file_name=f"calls_filtered_{date.today()}.csv",
        mime="text/csv",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.title("📞 Call Analytics Dashboard")
    st.caption(f"Data source: {DB_PATH}")

    # Load data
    df       = load_data()
    flags_df = load_flags()

    if df.empty:
        st.info(
            "No call records found. Run the pipeline first:\n\n"
            "```bash\n"
            "python generate_synthetic_calls.py\n"
            "python pipeline.py --input data/raw/synth/ --generate\n"
            "```"
        )
        return

    # Sidebar filters (returns filtered df)
    df, agent_filter, comp_filter = sidebar(df)

    st.markdown("---")

    # KPI row
    kpi_cards(df)
    st.markdown("---")

    # Charts
    charts(df, flags_df)
    st.markdown("---")

    # Agent table
    agent_table(df)
    st.markdown("---")

    # Call log
    call_log(df)


if __name__ == "__main__":
    main()
