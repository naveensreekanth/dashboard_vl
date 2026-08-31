import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import io

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retest_ai.config.settings import (
    ALL_MODEL_FEATURES,
    ATE_COST_CURRENCY,
    ATE_COST_PER_HOUR,
    IDENTIFIER_COLS,
    MONTH_12_OUTCOMES_FILE,
    TARGET_COL,
)
from retest_ai.validation.outcome_validator import validate_recommendations_against_outcomes
from retest_ai.models.service import MLService
from retest_ai.decision.decision_policy import (
    DOCX_REFERENCE_THRESHOLD,
    POLICY_LABEL,
    RETEST_LABEL,
)
from retest_ai.kpis.business_impact import (
    ESTIMATED_TIME_COL,
    format_money,
    format_seconds,
    seconds_to_cost,
)
import importlib
from retest_ai.kpis.breakdowns import filter_month12_batch_table

try:
    from retest_ai.kpis.breakdowns import overview_recommendation_counts
except ImportError:
    # Streamlit can keep a stale kpis.breakdowns module from before this helper existed.
    import retest_ai.kpis.breakdowns as _kpis_breakdowns
    _kpis_breakdowns = importlib.reload(_kpis_breakdowns)
    overview_recommendation_counts = _kpis_breakdowns.overview_recommendation_counts

st.set_page_config(
    page_title="ATE Retest AI Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #070b14 !important;
        color: #f1f5f9 !important;
    }
    .stAppHeader { background-color: transparent !important; }
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1600px !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #0b1120 !important;
        border-right: 1px solid #1a263d !important;
        width: 280px !important;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 1.2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .sidebar-brand {
        display: flex; align-items: center; gap: 12px;
        padding: 8px 12px 20px 12px;
        border-bottom: 1px solid #1a263d; margin-bottom: 20px;
    }
    .brand-icon {
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
        width: 40px; height: 40px; border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 22px; color: #ffffff;
        box-shadow: 0 0 15px rgba(168, 85, 247, 0.4);
    }
    .brand-text-title { font-size: 16px; font-weight: 800; letter-spacing: 0.5px; color: #ffffff; line-height: 1.1; }
    .brand-text-sub { font-size: 12px; font-weight: 700; letter-spacing: 1.5px; color: #a855f7; margin-top: 2px; }
    .nav-header {
        font-size: 11px; font-weight: 700; text-transform: uppercase;
        letter-spacing: 1.2px; color: #64748b; margin-top: 16px; margin-bottom: 8px; padding-left: 10px;
    }
    .top-header {
        display: flex; justify-content: space-between; align-items: center;
        background: #0f172a; border: 1px solid #1e293b; border-radius: 12px;
        padding: 14px 24px; margin-bottom: 24px;
    }
    .top-title { font-size: 22px; font-weight: 800; color: #ffffff; letter-spacing: -0.3px; }
    .top-subtitle { font-size: 13px; color: #94a3b8; margin-top: 2px; }
    .top-badges { display: flex; align-items: center; gap: 12px; }
    .badge-pill-model {
        background: rgba(168, 85, 247, 0.15); border: 1px solid #a855f7; color: #d8b4fe;
        font-size: 12px; font-weight: 600; padding: 6px 14px; border-radius: 20px;
    }
    .badge-pill-status {
        background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; color: #6ee7b7;
        font-size: 12px; font-weight: 600; padding: 6px 14px; border-radius: 20px;
    }
    .dark-card {
        background: #111a2d; border: 1px solid #1e2c4a; border-radius: 12px;
        padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
    }
    .dark-card-header { font-size: 15px; font-weight: 700; color: #ffffff; margin-bottom: 14px; }
    .dark-card-compact {
        background: #111a2d; border: 1px solid #1e2c4a; border-radius: 12px;
        padding: 12px 16px; margin-bottom: 12px; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
    }
    .workflow-strip {
        display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
        font-size: 11px; font-weight: 700; letter-spacing: 0.6px;
        text-transform: uppercase; color: #94a3b8; margin-bottom: 6px;
    }
    .workflow-step { color: #e2e8f0; }
    .workflow-arrow { color: #64748b; }
    .wf-panel {
        background: #111a2d; border: 1px solid #1e2c4a; border-radius: 12px;
        padding: 22px 24px; min-height: 360px; margin-bottom: 12px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
        display: flex; flex-direction: column;
    }
    .wf-title {
        font-size: 18px; font-weight: 700; letter-spacing: 0.8px;
        text-transform: uppercase; color: #d8b4fe; margin-bottom: 16px;
    }
    .wf-steps {
        display: grid; grid-template-columns: 1fr 1fr; gap: 12px 16px;
        align-items: center; flex: 1;
    }
    .wf-item { font-size: 17px; font-weight: 600; }
    .wf-item.done { color: #6ee7b7; }
    .wf-item.now {
        color: #d8b4fe; border: 1px solid #a855f7; border-radius: 8px;
        padding: 8px 12px; background: rgba(168, 85, 247, 0.12);
        display: inline-block;
    }
    .wf-item.todo { color: #64748b; }
    .wf-arrow { color: #475569; font-size: 17px; margin-right: 6px; }
    .wf-rec { color: #10b981; font-weight: 700; }
    .wf-skip { color: #ef4444; font-weight: 700; }
    .wf-blurb { font-size: 14px; color: #94a3b8; margin-top: 16px; line-height: 1.5; }
    .data-source-pill {
        display: inline-block; background: rgba(168, 85, 247, 0.15);
        border: 1px solid #a855f7; color: #d8b4fe;
        font-size: 12px; font-weight: 600; padding: 6px 14px; border-radius: 20px;
    }
    .kpi-card {
        background: #111a2d; border: 1px solid #1e2c4a; border-radius: 12px;
        padding: 20px 22px; min-height: 148px; height: 100%;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .kpi-gap { margin: 0 8px; }
    .kpi-row-spacer { height: 18px; }
    .kpi-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
    .kpi-label { font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; color: #94a3b8; }
    .kpi-context {
        font-size: 10px; font-weight: 600; text-transform: uppercase;
        letter-spacing: 1px; color: #64748b; margin-bottom: 4px;
    }
    .kpi-value { font-size: 38px; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: #ffffff; line-height: 1.1; }
    .kpi-value-cost { font-size: 28px; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: #ffffff; line-height: 1.1; }
    .kpi-sub { font-size: 12px; color: #64748b; margin-top: 8px; }
    .kpi-pair { display: flex; align-items: flex-end; gap: 28px; }
    .kpi-pair-item { display: flex; flex-direction: column; }
    .kpi-pair-label { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; color: #94a3b8; margin-bottom: 4px; }
    .retest-banner {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 2px solid #10b981; border-radius: 12px; padding: 22px; text-align: center;
    }
    .skip-banner {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.05) 100%);
        border: 2px solid #ef4444; border-radius: 12px; padding: 22px; text-align: center;
    }
    .rec-heading { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 6px; }
    .rec-badge-retest { font-size: 38px; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: #10b981; }
    .rec-badge-skip { font-size: 38px; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: #ef4444; }
    .prob-focal-card {
        background: #111a2d; border: 1px solid #233554; border-radius: 12px; padding: 24px; text-align: center;
    }
    .prob-focal-label { font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; color: #a855f7; margin-bottom: 6px; }
    .prob-focal-value { font-size: 52px; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: #ffffff; }
    .param-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
    .param-item { background: #0d1526; border: 1px solid #1a263d; border-radius: 8px; padding: 12px 14px; }
    .param-item-label { font-size: 11px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }
    .param-item-val { font-size: 16px; font-weight: 700; color: #f1f5f9; font-family: 'JetBrains Mono', monospace; margin-top: 3px; }
    .sidebar-footer-card { background: #0d1526; border: 1px solid #1a263d; border-radius: 10px; padding: 14px; margin-top: 30px; }
    .status-indicator {
        display: inline-block; width: 8px; height: 8px; border-radius: 50%;
        background-color: #10b981; margin-right: 6px; box-shadow: 0 0 8px #10b981;
    }
    .bottom-info-bar {
        display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;
        background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 14px 20px; margin-top: 24px;
    }
    .info-bar-item { display: flex; align-items: center; gap: 10px; }
    .info-bar-icon { font-size: 18px; color: #a855f7; }
    .info-bar-text-label { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }
    .info-bar-text-val { font-size: 13px; font-weight: 700; color: #f1f5f9; }
    .policy-note { font-size: 12px; color: #94a3b8; margin-top: 8px; }
    .ol-row { display: flex; justify-content: space-between; gap: 12px; margin: 4px 0; }
    .ol-label { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: #64748b; }
    .ol-val { font-size: 13px; font-weight: 700; color: #f1f5f9; text-align: right; }
    .ol-behavior { font-size: 12px; color: #94a3b8; margin-top: 8px; line-height: 1.4; }
    div[data-baseweb="select"] > div {
        background-color: #0d1526 !important; border-color: #1e2c4a !important; color: #ffffff !important; border-radius: 8px !important;
    }
    div[data-baseweb="select"] * { color: #ffffff !important; }
    .stSelectbox label, .stMultiSelect label, .stSlider label {
        color: #94a3b8 !important; font-weight: 600 !important; font-size: 13px !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #9333ea 100%) !important;
        color: #ffffff !important; border: none !important; border-radius: 8px !important;
        padding: 8px 18px !important; font-weight: 600 !important;
    }
    .stDownloadButton > button {
        background: #1e293b !important; color: #38bdf8 !important;
        border: 1px solid #334155 !important; border-radius: 8px !important; font-weight: 600 !important;
    }
    [data-testid="stDataFrame"] {
        background: #0d1526 !important; border: 1px solid #1e2c4a !important; border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_ml_service():
    return MLService.get_instance()


ml_service = get_ml_service()
if type(ml_service) is not MLService or not hasattr(ml_service, "load_and_predict_pre_retest_workbook"):
    ml_service.__class__ = MLService
comparison_results = ml_service.comparison_results
explainer = ml_service.explainer
datasets = ml_service.datasets

EVENT_DETAIL_COLS = [
    "Device_ID", "Failure_Event", "Fail_Test", "Fail_Bin", "Wafer_ID", "ATE_Site",
    "Voltage_V", "Temperature_C", "First_Test_Time_sec", ESTIMATED_TIME_COL,
    "Estimated_Retest_Cost", "AI_Predicted_Retest_Cost",
    "P(RETEST_BENEFICIAL)", "AI_Recommendation", "Ground_Truth"
]


def kpi_card_html(label, value, sub, color="#ffffff", context_label=None):
    context = f'<div class="kpi-context">{context_label}</div>' if context_label else ""
    return (
        '<div class="kpi-card">'
        f"{context}"
        f'<div class="kpi-top"><div class="kpi-label">{label}</div></div>'
        f'<div class="kpi-value" style="color:{color};">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        "</div>"
    )


def kpi_events_devices_card_html(label, events, devices, color="#ffffff", context_label=None):
    context = f'<div class="kpi-context">{context_label}</div>' if context_label else ""
    return (
        '<div class="kpi-card">'
        f"{context}"
        f'<div class="kpi-top"><div class="kpi-label">{label}</div></div>'
        '<div class="kpi-pair">'
        '<div class="kpi-pair-item">'
        '<div class="kpi-pair-label">Events</div>'
        f'<div class="kpi-value" style="color:{color};">{events}</div>'
        "</div>"
        '<div class="kpi-pair-item">'
        '<div class="kpi-pair-label">Devices</div>'
        f'<div class="kpi-value" style="color:{color};">{devices}</div>'
        "</div>"
        "</div>"
        "</div>"
    )


def kpi_pct_events_devices_card_html(label, pct, events, devices, color="#ffffff"):
    return (
        '<div class="kpi-card">'
        f'<div class="kpi-top"><div class="kpi-label">{label}</div></div>'
        f'<div class="kpi-value" style="color:{color};">{pct}</div>'
        '<div class="kpi-pair" style="margin-top:10px;">'
        '<div class="kpi-pair-item">'
        '<div class="kpi-pair-label">Events</div>'
        f'<div class="kpi-value" style="color:{color};">{events}</div>'
        "</div>"
        '<div class="kpi-pair-item">'
        '<div class="kpi-pair-label">Devices</div>'
        f'<div class="kpi-value" style="color:{color};">{devices}</div>'
        "</div>"
        "</div>"
        "</div>"
    )


def kpi_cost_card_html(label, value, sub, color="#ffffff", context_label=None):
    context = f'<div class="kpi-context">{context_label}</div>' if context_label else ""
    return (
        '<div class="kpi-card">'
        f"{context}"
        f'<div class="kpi-top"><div class="kpi-label">{label}</div></div>'
        f'<div class="kpi-value-cost" style="color:{color};">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        "</div>"
    )


def _active_cost_per_hour():
    return max(float(st.session_state.get("ate_cost_per_hour", ATE_COST_PER_HOUR)), 0.0)


def _with_cost_columns(df, cost_per_hour=None):
    if df is None or len(df) == 0:
        return df
    out = df if ESTIMATED_TIME_COL in df.columns else ml_service.attach_estimated_retest_times(df)
    out = out.copy()
    rate = ATE_COST_PER_HOUR if cost_per_hour is None else cost_per_hour
    times = pd.to_numeric(out[ESTIMATED_TIME_COL], errors="coerce").fillna(0.0)
    out["Estimated_Retest_Cost"] = (times * (max(float(rate), 0.0) / 3600.0)).round(2)
    rec = out["AI_Recommendation"].astype(str).str.strip() if "AI_Recommendation" in out.columns else pd.Series("", index=out.index)
    out["AI_Predicted_Retest_Cost"] = out["Estimated_Retest_Cost"].where(rec == RETEST_LABEL, 0.0).round(2)
    return out


def _go_overview_view(view):
    st.session_state["overview_view"] = view
    st.rerun()


def _back_to_overview_button(key):
    if st.button("← Back to Overview", key=key):
        _go_overview_view("overview")


def render_analysis_workflow(current_stage):
    if current_stage == "learned":
        marks = ["done", "done", "done", "done", "done", "done", "done", "now"]
    elif current_stage == "validate":
        marks = ["done", "done", "done", "done", "done", "done", "now", "todo"]
    elif current_stage == "recommend":
        marks = ["done", "done", "now", "now", "todo", "todo", "todo", "todo"]
    else:
        marks = ["now", "todo", "todo", "todo", "todo", "todo", "todo", "todo"]
    labels = [
        "Upload Pre-Retest Data",
        "Analyze with AI",
        'AI Recommendation — <span class="wf-rec">RETEST</span> or <span class="wf-skip">DON\'T RETEST</span>',
        "Estimate Retest Cost — all-device vs AI",
        "Perform Actual Retest",
        "Upload Actual Outcomes",
        "Validate AI Recommendation",
        'Click <b>Learn from These Validated Outcomes</b> — RLS (not automatic)',
    ]
    parts = []
    for i, (mark, label) in enumerate(zip(marks, labels), start=1):
        prefix = "✓ " if mark == "done" else ("● " if mark == "now" else "○ ")
        arrow = '<span class="wf-arrow">→</span>' if i > 1 else ""
        parts.append(f'<span class="wf-item {mark}">{arrow}{prefix}{i}. {label}</span>')
    html = (
        '<div class="wf-panel">'
        '<div class="wf-title">How AI Analysis Works</div>'
        f'<div class="wf-steps">{"".join(parts)}</div>'
        '<div class="wf-blurb">The AI recommends whether a retest may be beneficial and estimates '
        "tester-time cost for all devices vs AI-selected retests. After actual testing, upload outcomes "
        "to validate the recommendations. Online (RLS) learning starts only if you explicitly click "
        "<b>Learn from These Validated Outcomes</b> — it does not run automatically after verification.</div>"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


REC_COLOR_MAP = {"RETEST": "#10b981", "DON'T RETEST": "#ef4444"}
OUTCOME_COLOR_MAP = {"RETEST_BENEFICIAL": "#a855f7", "PERSISTENT_FAILURE": "#f59e0b"}


def _style_inspect_fig(fig, height=230, title=None):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b"),
        font={"family": "Inter", "color": "#f1f5f9"},
        title=dict(text=title, font=dict(size=14, color="#f1f5f9")) if title else None,
        showlegend=False,
    )
    return fig


def _render_inspect_chart(df, empty_message, make_fig, height=230):
    if df is None or len(df) == 0:
        st.info(empty_message)
        return
    fig = make_fig(df)
    title = fig.layout.title.text if fig.layout.title and fig.layout.title.text else None
    _style_inspect_fig(fig, height=height, title=title)
    st.plotly_chart(fig, use_container_width=True)


def show_event_table(df, title):
    cols = [c for c in EVENT_DETAIL_COLS if c in df.columns]
    st.caption(title)
    if len(df) == 0:
        st.info("No events in this cell.")
        return
    display_df = df[cols].copy()
    display_df.insert(0, "S.No", range(1, len(display_df) + 1))
    st.dataframe(display_df, use_container_width=True, height=280, hide_index=True)


def render_recommendation_inspect(
    df_m12,
    rec_label,
    caption,
    back_key,
    csv_name,
    xlsx_name,
    sheet_name,
    csv_key,
    xlsx_key,
    table_title,
):
    _back_to_overview_button(back_key)
    rec = df_m12["AI_Recommendation"].astype(str).str.strip()
    filtered = df_m12[rec == rec_label].copy()
    cols = [c for c in EVENT_DETAIL_COLS if c in filtered.columns]
    export_df = filtered[cols] if cols else filtered
    st.markdown(f"**AI Recommended: {rec_label}**")
    st.caption(caption.format(n=len(filtered)))
    rec_color = REC_COLOR_MAP.get(rec_label, "#38bdf8")
    _render_inspect_chart(
        filtered if "P(RETEST_BENEFICIAL)" in filtered.columns else filtered.iloc[0:0],
        f"No {rec_label} events available for visualization.",
        lambda d: px.histogram(
            d,
            x="P(RETEST_BENEFICIAL)",
            nbins=20,
            color_discrete_sequence=[rec_color],
            title=f"Probability Distribution of {rec_label} Recommendations",
        ),
    )
    e1, e2, _ = st.columns([2, 2, 6])
    with e1:
        st.download_button(
            "Export CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=csv_name,
            mime="text/csv",
            key=csv_key,
        )
    with e2:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            export_df.to_excel(writer, index=False, sheet_name=sheet_name)
        st.download_button(
            "Export Excel",
            data=buf.getvalue(),
            file_name=xlsx_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=xlsx_key,
        )
    show_event_table(filtered, table_title)


def render_cost_inspect(df_m12, impact, view, back_key):
    _back_to_overview_button(back_key)
    cost_df = _with_cost_columns(df_m12, impact["cost_per_hour"])
    rec = cost_df["AI_Recommendation"].astype(str).str.strip()
    if view == "ai_retest_cost":
        shown = cost_df[rec == RETEST_LABEL].copy()
        title = "AI predicted retest cost"
        caption = (
            f"{format_money(impact['ai_predicted_retest_cost'], impact['currency'])}  \n"
            f"{int(impact['retest_recommendations_count'])} RETEST events · "
            f"{format_seconds(impact['ai_predicted_retest_time_sec'])} · "
            f"{format_money(impact['cost_per_hour'], impact['currency'])}/h"
        )
        chart_title = "Estimated cost of AI RETEST events by Fail_Test"
        value_col = "AI_Predicted_Retest_Cost"
    else:
        shown = cost_df.copy()
        title = "Actual cost of all devices"
        caption = (
            f"{format_money(impact['all_device_retest_cost'], impact['currency'])}  \n"
            f"If every failure event is retested · {int(impact['total_events'])} events · "
            f"{format_seconds(impact['all_device_retest_time_sec'])} · "
            f"{format_money(impact['cost_per_hour'], impact['currency'])}/h"
        )
        chart_title = "Estimated all-device retest cost by Fail_Test"
        value_col = "Estimated_Retest_Cost"

    st.markdown(f"**{title}**")
    st.caption(caption)
    st.caption(
        "Duration is estimated from historical Retest_Time_sec by Fail_Test (Month 0 + Month 6). "
        "It is not actual Month 12 tester time and not used as a model feature."
    )

    compare_df = pd.DataFrame({
        "Scenario": ["All devices retested", "AI recommended RETEST", "Estimated savings"],
        "Cost": [
            impact["all_device_retest_cost"],
            impact["ai_predicted_retest_cost"],
            impact["estimated_savings"],
        ],
    })
    fig_cmp = px.bar(
        compare_df,
        x="Scenario",
        y="Cost",
        color="Scenario",
        color_discrete_map={
            "All devices retested": "#38bdf8",
            "AI recommended RETEST": "#10b981",
            "Estimated savings": "#a855f7",
        },
        title="Tester-time cost comparison",
    )
    _style_inspect_fig(fig_cmp, height=230, title="Tester-time cost comparison")
    st.plotly_chart(fig_cmp, use_container_width=True)

    if "Fail_Test" in shown.columns and len(shown) > 0:
        by_test = (
            shown.groupby("Fail_Test", dropna=False)[value_col]
            .sum()
            .reset_index()
            .sort_values(value_col, ascending=False)
        )
        by_test.columns = ["Fail_Test", "Cost"]
        fig_test = px.bar(by_test, x="Fail_Test", y="Cost", color_discrete_sequence=["#38bdf8"], title=chart_title)
        _style_inspect_fig(fig_test, height=230, title=chart_title)
        st.plotly_chart(fig_test, use_container_width=True)

    show_event_table(shown, title)


def _month12_test_family(val):
    s = str(val).lower()
    if "scan" in s:
        return "Scan"
    if "mbist" in s:
        return "MBIST"
    if "iddq" in s:
        return "IDDQ"
    if "func" in s:
        return "Func"
    if "atspeed" in s or "at_speed" in s:
        return "AtSpeed"
    return str(val)


UPLOADED_SOURCE_LABEL = "Uploaded pre-retest data"


def _init_analysis_session():
    defaults = {
        "prediction_source": None,
        "uploaded_predictions": None,
        "active_outcomes": None,
        "outcomes_loaded_for_active_dataset": False,
        "uploaded_pre_retest_name": None,
        "pending_pre_retest_name": None,
        "outcome_uploader_nonce": 0,
        "online_learning_flash": None,
        "confirm_reset_online_learning": False,
        "outcomes_learned_for_active_dataset": False,
        "ate_cost_per_hour": float(ATE_COST_PER_HOUR),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_service_outcomes():
    ml_service.month_12_outcomes = None


def _reset_validation_state(bump_outcome_uploader=True):
    st.session_state["active_outcomes"] = None
    st.session_state["outcomes_loaded_for_active_dataset"] = False
    st.session_state["outcomes_learned_for_active_dataset"] = False
    _reset_service_outcomes()
    if bump_outcome_uploader:
        st.session_state["outcome_uploader_nonce"] = int(st.session_state.get("outcome_uploader_nonce") or 0) + 1


def _clear_analysis_state():
    st.session_state["prediction_source"] = None
    st.session_state["uploaded_predictions"] = None
    st.session_state["uploaded_pre_retest_name"] = None
    st.session_state["pending_pre_retest_name"] = None
    st.session_state["overview_view"] = "overview"
    _reset_validation_state(bump_outcome_uploader=True)


def get_active_prediction_frame():
    if st.session_state.get("prediction_source") == "uploaded":
        uploaded = st.session_state.get("uploaded_predictions")
        if uploaded is not None and len(uploaded) > 0:
            return uploaded, UPLOADED_SOURCE_LABEL
    return None, None


def _join_active_validation(preds, outcomes):
    if preds is None or outcomes is None or len(preds) == 0 or len(outcomes) == 0:
        return None
    if TARGET_COL not in outcomes.columns:
        return None
    keys = [c for c in IDENTIFIER_COLS if c in preds.columns and c in outcomes.columns]
    if not keys:
        return None
    extra = [c for c in [TARGET_COL, "Retest_Result", "Final_Result"] if c in outcomes.columns]
    return preds.merge(outcomes[list(dict.fromkeys(keys + extra))], on=keys, how="inner", suffixes=("", "_outcome"))


def _online_learning_flash_message(result):
    if not result:
        return None, None
    if result.get("reset"):
        return "info", "Online learning was reset. Future predictions use the base model until new validated outcomes are learned."
    if result.get("already_learned"):
        return "info", "These validated outcomes have already been used for online learning."
    learned = int(result.get("learned") or 0)
    if learned <= 0 and not result.get("reset"):
        skipped = int(result.get("skipped") or 0)
        if skipped:
            return "warning", f"No valid Ground_Truth rows were available for online learning ({skipped} skipped)."
        return "warning", "No valid Ground_Truth rows were available for online learning."
    if result.get("active"):
        return "success", f"Online learning updated with {learned} validated events. The adaptation layer is active."
    collected = int(result.get("update_count") or learned)
    threshold = int(result.get("activation_threshold") or 20)
    return "success", (
        f"Online learning has collected {collected} validated events. "
        f"It will begin adapting predictions after {threshold} events."
    )


def render_online_learning_panel(m12_val, m12_has_outcomes):
    status = ml_service.get_online_learning_status()
    update_count = int(status.get("update_count") or 0)
    threshold = int(status.get("activation_threshold") or 20)
    active = bool(status.get("active"))
    if active:
        learned_display = str(update_count)
        adaptation_label = "Active"
        behavior = "Future probabilities are adjusted using recent approved post-retest outcomes."
        adapt_color = "#6ee7b7"
    else:
        learned_display = f"{update_count} / {threshold}"
        adaptation_label = "Warming Up"
        behavior = "Predictions currently use the base model until enough validated outcomes are learned."
        adapt_color = "#fbbf24"

    if m12_has_outcomes and m12_val is not None and len(m12_val) > 0:
        if st.button("Learn from These Validated Outcomes", key="learn_validated_outcomes"):
            result = ml_service.update_from_validated_outcomes(m12_val)
            st.session_state["online_learning_flash"] = result
            if result and (int(result.get("learned") or 0) > 0 or result.get("already_learned")):
                st.session_state["outcomes_learned_for_active_dataset"] = True
            st.rerun()

    flash = st.session_state.get("online_learning_flash")
    if flash:
        kind, message = _online_learning_flash_message(flash)
        if kind == "success":
            st.success(message)
        elif kind == "warning":
            st.warning(message)
        elif kind == "info":
            st.info(message)
        st.session_state["online_learning_flash"] = None

    st.markdown(
        f"""
        <div class="dark-card-compact">
            <div class="dark-card-header" style="margin-bottom:8px;">ONLINE LEARNING</div>
            <div class="ol-row"><div class="ol-label">Base Model</div><div class="ol-val">{ml_service.model_name}</div></div>
            <div class="ol-row"><div class="ol-label">Validated Events Learned</div><div class="ol-val">{learned_display}</div></div>
            <div class="ol-row"><div class="ol-label">Adaptation</div><div class="ol-val" style="color:{adapt_color};">{adaptation_label}</div></div>
            <div class="ol-row"><div class="ol-label">Forgetting Factor</div><div class="ol-val">{status.get("forgetting_factor")}</div></div>
            <div class="ol-behavior">{behavior}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.get("confirm_reset_online_learning"):
        st.warning(
            "Resetting online learning returns future predictions to the base model until new "
            "validated outcomes are learned. This does not delete the trained model, uploaded "
            "predictions, outcomes, or validation results."
        )
        confirm_col, cancel_col = st.columns(2)
        with confirm_col:
            if st.button("Confirm Reset", key="confirm_reset_online_learning_yes"):
                ml_service.reset_online_learning()
                st.session_state["confirm_reset_online_learning"] = False
                st.session_state["outcomes_learned_for_active_dataset"] = False
                st.session_state["online_learning_flash"] = {"reset": True}
                st.rerun()
        with cancel_col:
            if st.button("Cancel", key="confirm_reset_online_learning_no"):
                st.session_state["confirm_reset_online_learning"] = False
                st.rerun()
    else:
        if st.button("Reset Online Learning", key="reset_online_learning"):
            st.session_state["confirm_reset_online_learning"] = True
            st.rerun()


def render_month12_analysis(df_m12, m12_has_outcomes, source_label=UPLOADED_SOURCE_LABEL):
    title = "Pre-Retest Analysis"
    st.markdown(f"### {title}")
    st.caption(
        f"Active data: {source_label}. Predictions use only pre-retest features. "
        "These recommendations are not claimed correct unless outcomes are loaded separately."
    )

    k1, k2, k3, k4 = st.columns(4, gap="medium")
    with k1:
        st.markdown('<div class="kpi-gap">' + kpi_card_html("Total Events", f"{len(df_m12)}", source_label) + "</div>", unsafe_allow_html=True)
    with k2:
        st.markdown('<div class="kpi-gap">' + kpi_card_html("Average Probability", f"{df_m12['P(RETEST_BENEFICIAL)'].mean()*100:.2f}%", "Lot mean P", "#38bdf8") + "</div>", unsafe_allow_html=True)
    with k3:
        st.markdown('<div class="kpi-gap">' + kpi_card_html("Highest Probability", f"{df_m12['P(RETEST_BENEFICIAL)'].max()*100:.2f}%", "Max P", "#10b981") + "</div>", unsafe_allow_html=True)
    with k4:
        st.markdown('<div class="kpi-gap">' + kpi_card_html("Lowest Probability", f"{df_m12['P(RETEST_BENEFICIAL)'].min()*100:.2f}%", "Min P", "#ef4444") + "</div>", unsafe_allow_html=True)

    if not m12_has_outcomes:
        st.info("Accuracy / Unnecessary Retests / Missed Opportunities are not shown because actual outcomes have not been loaded.")

    def _rec_distribution_fig(df):
        rec_counts = (
            df["AI_Recommendation"].astype(str).str.strip()
            .value_counts()
            .reindex(["RETEST", "DON'T RETEST"])
            .fillna(0)
            .reset_index()
        )
        rec_counts.columns = ["AI_Recommendation", "Events"]
        return px.bar(
            rec_counts,
            x="AI_Recommendation",
            y="Events",
            color="AI_Recommendation",
            color_discrete_map=REC_COLOR_MAP,
            title="AI Recommendation Distribution",
        )

    _render_inspect_chart(
        df_m12,
        "No events available for visualization.",
        _rec_distribution_fig,
    )

    fig_pdist = px.histogram(
        df_m12,
        x="P(RETEST_BENEFICIAL)",
        nbins=20,
        color="AI_Recommendation",
        color_discrete_map={"RETEST": "#10b981", "DON'T RETEST": "#ef4444"},
    )
    fig_pdist.update_layout(
        height=240, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#1e293b", title="P(RETEST_BENEFICIAL)"),
        yaxis=dict(gridcolor="#1e293b", title="Count"),
        font={"family": "Inter", "color": "#f1f5f9"},
    )
    st.plotly_chart(fig_pdist, use_container_width=True)
    df_g = df_m12.copy()
    df_g["Test_Family"] = df_g["Fail_Test"].map(_month12_test_family)
    df_g = df_g.groupby(["Test_Family", "AI_Recommendation"]).size().reset_index(name="Count")
    fig_g = px.bar(
        df_g, x="Test_Family", y="Count", color="AI_Recommendation", barmode="stack",
        color_discrete_map={"RETEST": "#10b981", "DON'T RETEST": "#ef4444"},
    )
    fig_g.update_layout(
        height=220, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"),
        font={"family": "Inter", "color": "#f1f5f9"},
    )
    st.plotly_chart(fig_g, use_container_width=True)

    f1, f2, f3, f4, f5 = st.columns(5)
    with f1:
        rec_filter = st.multiselect(
            "Recommendation",
            ["RETEST", "DON'T RETEST"],
            default=[],
            key="m12_rec_filter",
        )
    with f2:
        test_filter = st.multiselect("Fail Test", sorted(df_m12["Fail_Test"].unique().tolist()), key="m12_test_filter")
    with f3:
        wafer_filter = st.multiselect("Wafer ID", sorted(df_m12["Wafer_ID"].unique().tolist()), key="m12_wafer_filter")
    with f4:
        site_filter = st.multiselect(
            "ATE Site",
            sorted(df_m12["ATE_Site"].unique().tolist()) if "ATE_Site" in df_m12.columns else [],
            key="m12_site_filter",
        )
    with f5:
        prob_range = st.slider("Probability range", 0.0, 1.0, (0.0, 1.0), 0.05, key="m12_prob_range")

    df_filtered = filter_month12_batch_table(
        df_m12,
        rec_filter=rec_filter,
        test_filter=test_filter,
        wafer_filter=wafer_filter,
        site_filter=site_filter,
        prob_range=prob_range,
    )

    st.caption(f"Showing {len(df_filtered)} of {len(df_m12)} events")

    cost_view = _with_cost_columns(df_filtered, _active_cost_per_hour())
    disp_cols = [c for c in [
        "Device_ID", "Failure_Event", "Fail_Test", "Fail_Bin", "Voltage_V", "Temperature_C",
        ESTIMATED_TIME_COL, "AI_Predicted_Retest_Cost",
        "P(RETEST_BENEFICIAL)", "AI_Recommendation",
    ] if c in cost_view.columns]
    display_df = cost_view[disp_cols].copy()
    display_df.insert(0, "S.No", range(1, len(display_df) + 1))
    st.dataframe(display_df, use_container_width=True, height=360, hide_index=True)

    e1, e2, _ = st.columns([2, 2, 6])
    with e1:
        st.download_button("Export CSV", data=display_df.to_csv(index=False).encode("utf-8"),
                           file_name="Retest_Recommendations.csv", mime="text/csv")
    with e2:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            display_df.to_excel(writer, index=False, sheet_name="Recommendations")
        st.download_button("Export Excel", data=buf.getvalue(),
                           file_name="Retest_Recommendations.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-icon">⚡</div>
        <div>
            <div class="brand-text-title">ATE RETEST</div>
            <div class="brand-text-sub">AI AGENT</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-header">DASHBOARD</div>', unsafe_allow_html=True)
    nav_dashboard = st.button("Overview Dashboard", use_container_width=True)
    st.markdown('<div class="nav-header">ANALYSIS</div>', unsafe_allow_html=True)
    nav_single = st.button("Single Event Analysis", use_container_width=True)
    nav_models = st.button("Historical Temporal Validation", use_container_width=True)
    st.markdown('<div class="nav-header">SYSTEM</div>', unsafe_allow_html=True)
    nav_info = st.button("Model Info & Specs", use_container_width=True)
    nav_settings = st.button("Decision Policy", use_container_width=True)

    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "overview"
    if "overview_view" not in st.session_state:
        st.session_state["overview_view"] = "overview"
    if "show_pre_retest_upload" not in st.session_state:
        st.session_state["show_pre_retest_upload"] = False
    _init_analysis_session()

    if nav_dashboard:
        st.session_state["current_page"] = "overview"
        st.session_state["overview_view"] = "overview"
    elif nav_single:
        st.session_state["current_page"] = "single"
    elif nav_models:
        st.session_state["current_page"] = "models"
    elif nav_info:
        st.session_state["current_page"] = "info"
    elif nav_settings:
        st.session_state["current_page"] = "settings"

    st.markdown("""
    <div class="sidebar-footer-card">
        <div style="font-size: 12px; font-weight: 700; color: #ffffff; margin-bottom: 6px;">
            <span class="status-indicator"></span>System Status
        </div>
        <div style="font-size: 11px; color: #94a3b8;">ATE Retest-Benefit Prediction AI</div>
    </div>
    """, unsafe_allow_html=True)

current_page = st.session_state["current_page"]
if current_page == "batch":
    st.session_state["current_page"] = "overview"
    st.session_state["overview_view"] = "month12"
    current_page = "overview"
if current_page == "reference":
    st.session_state["current_page"] = "overview"
    current_page = "overview"
st.markdown("""
<div class="top-header">
    <div>
        <div class="top-title">ATE Retest-Benefit Prediction AI</div>
    </div>
</div>
""", unsafe_allow_html=True)

df_m12, prediction_source_label = get_active_prediction_frame()
has_active_analysis = df_m12 is not None and len(df_m12) > 0
hist_kpis = ml_service.get_historical_decision_kpis()
hist_table = ml_service.get_historical_validation_table()


# =========================================================
# OVERVIEW
# =========================================================
if current_page == "overview":
    n_retest = n_skip = n_m12 = total_devices = retest_devices = dont_retest_devices = 0
    m12_val = None
    m12_kpis = None
    m12_has_outcomes = False
    benefit_n = persist_n = 0
    benefit_rate = 0.0
    cost_impact = None
    cost_per_hour = _active_cost_per_hour()
    if has_active_analysis:
        counts = overview_recommendation_counts(df_m12)
        n_retest = counts["retest"]
        n_skip = counts["dont_retest"]
        n_m12 = counts["total_events"]
        total_devices = int(df_m12["Device_ID"].nunique())
        rec = df_m12["AI_Recommendation"].astype(str).str.strip()
        retest_device_ids = set(df_m12.loc[rec == "RETEST", "Device_ID"].dropna().unique())
        all_device_ids = set(df_m12["Device_ID"].dropna().unique())
        dont_retest_device_ids = all_device_ids - retest_device_ids
        retest_devices = len(retest_device_ids)
        dont_retest_devices = len(dont_retest_device_ids)
        cost_impact = ml_service.get_cost_impact(df_m12, cost_per_hour=cost_per_hour)
        if st.session_state.get("outcomes_loaded_for_active_dataset"):
            m12_val = _join_active_validation(df_m12, st.session_state.get("active_outcomes"))
            if m12_val is not None and len(m12_val) > 0:
                m12_kpis = validate_recommendations_against_outcomes(
                    m12_val[TARGET_COL], m12_val["AI_Recommendation"], events=m12_val
                )
                m12_has_outcomes = True
                benefit_n = int((m12_val["Ground_Truth"] == "RETEST_BENEFICIAL").sum())
                persist_n = int((m12_val["Ground_Truth"] == "PERSISTENT_FAILURE").sum())
                benefit_rate = benefit_n / len(m12_val) * 100 if len(m12_val) else 0.0

    overview_view = st.session_state.get("overview_view", "overview")
    if not has_active_analysis and overview_view != "overview":
        st.session_state["overview_view"] = "overview"
        overview_view = "overview"

    def _overview_test_family(val):
        s = str(val).lower()
        if "scan" in s:
            return "Scan"
        if "mbist" in s:
            return "MBIST"
        if "iddq" in s:
            return "IDDQ"
        if "func" in s:
            return "Func"
        if "atspeed" in s or "at_speed" in s:
            return "AtSpeed"
        return str(val)

    if overview_view == "month12":
        _back_to_overview_button("ov_back_month12")
        render_month12_analysis(df_m12, m12_has_outcomes, source_label=prediction_source_label)

    elif overview_view == "benefit_rate":
        _back_to_overview_button("ov_back_benefit")
        if m12_has_outcomes and m12_kpis is not None and m12_val is not None:
            rec_col = m12_val["AI_Recommendation"].astype(str).str.strip()
            beneficial_events = m12_val[
                (rec_col == "RETEST") & (m12_val["Ground_Truth"] == "RETEST_BENEFICIAL")
            ]
            total_ai_retest = int((rec_col == "RETEST").sum())
            beneficial_retests = len(beneficial_events)
            st.markdown(
                f"**Retest Benefit Rate**  \n"
                f"{beneficial_retests} / {total_ai_retest} = {benefit_rate:.1f}%  \n"
                f"Definition: share of AI RETEST recommendations whose actual outcome was RETEST_BENEFICIAL."
            )
            ai_retest_outcomes = m12_val[rec_col == "RETEST"]

            def _benefit_outcome_fig(d):
                gt = d["Ground_Truth"].astype(str).str.strip().value_counts()
                chart_df = pd.DataFrame({"Ground_Truth": gt.index.astype(str), "Events": gt.to_numpy()})
                return px.bar(
                    chart_df,
                    x="Ground_Truth",
                    y="Events",
                    color="Ground_Truth",
                    color_discrete_map=OUTCOME_COLOR_MAP,
                    title="Outcome of AI Recommended RETEST Events",
                )

            _render_inspect_chart(
                ai_retest_outcomes if "Ground_Truth" in ai_retest_outcomes.columns else ai_retest_outcomes.iloc[0:0],
                "No validation outcomes available for visualization.",
                _benefit_outcome_fig,
            )
            show_event_table(beneficial_events, "Beneficial events")
        else:
            st.caption("Outcomes are not loaded for the current analysis.")

    elif overview_view == "unnecessary_retests":
        _back_to_overview_button("ov_back_fp")
        if m12_has_outcomes and m12_kpis is not None and m12_val is not None:
            st.markdown(
                f"**Unnecessary Retests**  \n"
                f"{m12_kpis['fp']} / {m12_kpis['total_events']} = {m12_kpis['unnecessary_retests_pct']:.1f}%  \n"
                f"Definition: events where AI recommended RETEST but the actual outcome was PERSISTENT_FAILURE."
            )
            fp_events = m12_kpis["unnecessary_retest_events"]

            def _unnecessary_fig(d):
                if "Ground_Truth" in d.columns:
                    gt = d["Ground_Truth"].astype(str).str.strip().value_counts()
                    chart_df = pd.DataFrame({"Ground_Truth": gt.index.astype(str), "Events": gt.to_numpy()})
                    return px.bar(
                        chart_df,
                        x="Ground_Truth",
                        y="Events",
                        color="Ground_Truth",
                        color_discrete_map=OUTCOME_COLOR_MAP,
                        title="Unnecessary Retests: Recommended RETEST but Outcome Was Persistent Failure",
                    )
                chart_df = pd.DataFrame({"Category": ["Unnecessary Retests"], "Events": [len(d)]})
                return px.bar(
                    chart_df,
                    x="Category",
                    y="Events",
                    color_discrete_sequence=["#f59e0b"],
                    title="Unnecessary Retests: Recommended RETEST but Outcome Was Persistent Failure",
                )

            _render_inspect_chart(
                fp_events,
                "No unnecessary retest events available.",
                _unnecessary_fig,
            )
            show_event_table(fp_events, "Unnecessary Retest events")
        else:
            st.caption("Outcomes are not loaded for the current analysis.")

    elif overview_view == "retest_recommendations":
        render_recommendation_inspect(
            df_m12,
            rec_label="RETEST",
            caption="{n} events recommended for retest",
            back_key="back_retest_recommendations",
            csv_name="ai_recommended_retest_events.csv",
            xlsx_name="ai_recommended_retest_events.xlsx",
            sheet_name="RETEST Events",
            csv_key="export_retest_csv",
            xlsx_key="export_retest_excel",
            table_title="RETEST events",
        )

    elif overview_view == "dont_retest_recommendations":
        render_recommendation_inspect(
            df_m12,
            rec_label="DON'T RETEST",
            caption="{n} events not recommended for retest",
            back_key="back_dont_retest_recommendations",
            csv_name="ai_recommended_dont_retest_events.csv",
            xlsx_name="ai_recommended_dont_retest_events.xlsx",
            sheet_name="DONT RETEST Events",
            csv_key="export_dont_retest_csv",
            xlsx_key="export_dont_retest_excel",
            table_title="DON'T RETEST events",
        )

    elif overview_view == "all_device_cost":
        if cost_impact is None:
            _back_to_overview_button("back_all_device_cost_empty")
            st.caption("Upload a pre-retest workbook and analyze with AI to estimate cost.")
        else:
            render_cost_inspect(df_m12, cost_impact, "all_device_cost", "back_all_device_cost")

    elif overview_view == "ai_retest_cost":
        if cost_impact is None:
            _back_to_overview_button("back_ai_retest_cost_empty")
            st.caption("Upload a pre-retest workbook and analyze with AI to estimate cost.")
        else:
            render_cost_inspect(df_m12, cost_impact, "ai_retest_cost", "back_ai_retest_cost")

    else:
        st.markdown("### Overview")

        if m12_has_outcomes and st.session_state.get("outcomes_learned_for_active_dataset"):
            workflow_stage = "learned"
        elif m12_has_outcomes:
            workflow_stage = "validate"
        elif has_active_analysis:
            workflow_stage = "recommend"
        else:
            workflow_stage = "empty"

        toggle_label = (
            "− Upload Pre-Retest Data"
            if st.session_state.get("show_pre_retest_upload")
            else "+ Upload Pre-Retest Data"
        )
        workflow_col, upload_col = st.columns([5, 2], gap="medium")
        with workflow_col:
            render_analysis_workflow(workflow_stage)
        with upload_col:
            if st.button(toggle_label, key="toggle_pre_retest_upload", use_container_width=True):
                st.session_state["show_pre_retest_upload"] = not st.session_state.get("show_pre_retest_upload", False)
                st.rerun()
            if has_active_analysis:
                if st.button("Clear Analysis", key="clear_analysis", use_container_width=True):
                    _clear_analysis_state()
                    st.rerun()
            if st.session_state.get("show_pre_retest_upload"):
                pre_retest_file = st.file_uploader(
                    "Pre-retest events workbook",
                    type=["xlsx"],
                    key="pre_retest_upload",
                )
                if pre_retest_file is not None:
                    selected_name = getattr(pre_retest_file, "name", "uploaded.xlsx")
                    if st.session_state.get("pending_pre_retest_name") != selected_name:
                        st.session_state["pending_pre_retest_name"] = selected_name
                        st.session_state["outcome_uploader_nonce"] = int(st.session_state.get("outcome_uploader_nonce") or 0) + 1
                        st.rerun()
                analyze_clicked = st.button("Analyze with AI", key="analyze_pre_retest")
                if analyze_clicked:
                    if pre_retest_file is None:
                        st.error("Upload a pre-retest XLSX before analyzing.")
                    else:
                        try:
                            predicted = ml_service.load_and_predict_pre_retest_workbook(pre_retest_file)
                            _reset_validation_state(bump_outcome_uploader=True)
                            st.session_state["uploaded_predictions"] = predicted
                            st.session_state["prediction_source"] = "uploaded"
                            st.session_state["uploaded_pre_retest_name"] = getattr(pre_retest_file, "name", "uploaded.xlsx")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))

        if not has_active_analysis:
            st.caption("Upload a pre-retest workbook and select Analyze with AI to start analysis.")
        else:
            st.markdown('<div class="dark-card-header">AI Recommended Retest</div>', unsafe_allow_html=True)
            t1, t2, t3 = st.columns(3, gap="medium")
            with t1:
                st.markdown('<div class="kpi-gap">' + kpi_events_devices_card_html("Total Events", str(n_m12), str(total_devices)) + "</div>", unsafe_allow_html=True)
                if st.button("Inspect", key="ov_kpi_total", use_container_width=True):
                    _go_overview_view("month12")
            with t2:
                st.markdown('<div class="kpi-gap">' + kpi_events_devices_card_html("RETEST", str(n_retest), str(retest_devices), "#10b981", context_label="AI Recommended") + "</div>", unsafe_allow_html=True)
                if st.button("Inspect", key="ov_kpi_retest", use_container_width=True):
                    _go_overview_view("retest_recommendations")
            with t3:
                st.markdown('<div class="kpi-gap">' + kpi_events_devices_card_html("DON'T RETEST", str(n_skip), str(dont_retest_devices), "#ef4444", context_label="AI Recommended") + "</div>", unsafe_allow_html=True)
                if st.button("Inspect", key="ov_kpi_dont_retest", use_container_width=True):
                    _go_overview_view("dont_retest_recommendations")

            st.markdown('<div class="kpi-row-spacer"></div>', unsafe_allow_html=True)
            st.markdown('<div class="dark-card-header">Retest Cost Estimate</div>', unsafe_allow_html=True)
            st.caption(
                f"Cost = estimated retest time × {format_money(cost_per_hour, ATE_COST_CURRENCY)}/h tester rate "
                "(configurable on Decision Policy). Time is estimated from historical Retest_Time_sec by Fail_Test, "
                "not from actual Month 12 retest duration."
            )
            c1, c2, c3 = st.columns(3, gap="medium")
            with c1:
                st.markdown(
                    '<div class="kpi-gap">'
                    + kpi_cost_card_html(
                        "All-device retest cost",
                        format_money(cost_impact["all_device_retest_cost"], ATE_COST_CURRENCY),
                        f"{n_m12} events · {format_seconds(cost_impact['all_device_retest_time_sec'])}",
                        "#38bdf8",
                        context_label="If every fail is retested",
                    )
                    + "</div>",
                    unsafe_allow_html=True,
                )
                if st.button("Inspect", key="ov_kpi_all_cost", use_container_width=True):
                    _go_overview_view("all_device_cost")
            with c2:
                st.markdown(
                    '<div class="kpi-gap">'
                    + kpi_cost_card_html(
                        "AI predicted retest cost",
                        format_money(cost_impact["ai_predicted_retest_cost"], ATE_COST_CURRENCY),
                        f"{n_retest} RETEST events · {format_seconds(cost_impact['ai_predicted_retest_time_sec'])}",
                        "#10b981",
                        context_label="AI recommended RETEST",
                    )
                    + "</div>",
                    unsafe_allow_html=True,
                )
                if st.button("Inspect", key="ov_kpi_ai_cost", use_container_width=True):
                    _go_overview_view("ai_retest_cost")
            with c3:
                st.markdown(
                    '<div class="kpi-gap">'
                    + kpi_cost_card_html(
                        "Estimated savings",
                        format_money(cost_impact["estimated_savings"], ATE_COST_CURRENCY),
                        f"{n_skip} skipped events · {format_seconds(cost_impact['skipped_retest_time_sec'])}",
                        "#a855f7",
                        context_label="All-device minus AI retest",
                    )
                    + "</div>",
                    unsafe_allow_html=True,
                )

            st.markdown('<div class="kpi-row-spacer"></div>', unsafe_allow_html=True)

            st.markdown('<div class="dark-card-header">Validation of AI Recommended Retest</div>', unsafe_allow_html=True)
            k1, k2, k3 = st.columns(3, gap="medium")
            if m12_has_outcomes and m12_kpis is not None and m12_val is not None:
                rec_v = df_m12["AI_Recommendation"].astype(str).str.strip()
                retest_device_ids = set(df_m12.loc[rec_v == "RETEST", "Device_ID"].dropna().astype(str))
                unnecessary_events = m12_kpis.get("unnecessary_retest_events")
                if (
                    unnecessary_events is not None
                    and len(unnecessary_events) > 0
                    and "Device_ID" in unnecessary_events.columns
                ):
                    unnecessary_device_ids = set(
                        unnecessary_events["Device_ID"].dropna().astype(str)
                    ).intersection(retest_device_ids)
                else:
                    unnecessary_device_ids = set()
                benefit_device_ids = retest_device_ids - unnecessary_device_ids
                benefit_devices = len(benefit_device_ids)
                unnecessary_devices = len(unnecessary_device_ids)
                with k1:
                    st.markdown(
                        '<div class="kpi-gap">'
                        + kpi_pct_events_devices_card_html(
                            "Retest Benefit Rate",
                            f"{benefit_rate:.1f}%",
                            str(benefit_n),
                            str(benefit_devices),
                            "#a855f7",
                        )
                        + "</div>",
                        unsafe_allow_html=True,
                    )
                    if st.button("Inspect", key="ov_kpi_benefit", use_container_width=True):
                        _go_overview_view("benefit_rate")
                with k2:
                    st.markdown(
                        '<div class="kpi-gap">'
                        + kpi_pct_events_devices_card_html(
                            "Unnecessary Retests",
                            f'{m12_kpis["unnecessary_retests_pct"]:.1f}%',
                            str(m12_kpis["fp"]),
                            str(unnecessary_devices),
                            "#f59e0b",
                        )
                        + "</div>",
                        unsafe_allow_html=True,
                    )
                    if st.button("Inspect", key="ov_kpi_fp", use_container_width=True):
                        _go_overview_view("unnecessary_retests")
            else:
                with k1:
                    st.caption("Outcomes are not loaded for the current analysis. Retest Benefit Rate and Unnecessary Retests are not shown.")
            with k3:
                st.markdown('<div class="kpi-gap"><div class="dark-card"><div class="dark-card-header">UPLOAD OUTCOME DATA</div>', unsafe_allow_html=True)
                st.caption("Upload actual post-retest outcomes for validation only. Never used as AI prediction input.")
                outcome_key = f"outcome_upload_{st.session_state.get('outcome_uploader_nonce', 0)}"
                outcome_file = st.file_uploader("Upload outcomes workbook (Ground_Truth)", type=["xlsx"], key=outcome_key)
                load_uploaded = st.button("Load uploaded outcomes", key="load_outcome_upload")
                load_local = st.button("Load local private outcomes file if present", key="load_outcome_local")
                if load_uploaded:
                    if outcome_file is None:
                        st.error("Choose an outcomes XLSX before loading.")
                    else:
                        tmp_path = os.path.join(os.path.dirname(MONTH_12_OUTCOMES_FILE), "_uploaded_m12_outcomes.xlsx")
                        with open(tmp_path, "wb") as f:
                            f.write(outcome_file.getbuffer())
                        try:
                            df_gt = ml_service.load_month_12_outcomes(tmp_path)
                            st.session_state["active_outcomes"] = df_gt.copy()
                            st.session_state["outcomes_loaded_for_active_dataset"] = True
                            _reset_service_outcomes()
                            st.success("Outcomes loaded for validation only.")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
                elif load_local:
                    try:
                        df_gt = ml_service.load_month_12_outcomes()
                        st.session_state["active_outcomes"] = df_gt.copy()
                        st.session_state["outcomes_loaded_for_active_dataset"] = True
                        _reset_service_outcomes()
                        st.success("Outcomes loaded for validation only.")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
                st.markdown("</div></div>", unsafe_allow_html=True)

            if m12_has_outcomes and m12_kpis is not None and m12_val is not None:
                st.caption(
                    "Device counts are unique across these two KPIs and sum to the AI Recommended RETEST devices. "
                    "A device with any unnecessary post-retest event is counted as Unnecessary Retest."
                )
            render_online_learning_panel(m12_val, m12_has_outcomes)

            st.markdown('<div class="dark-card"><div class="dark-card-header">Model Quality</div>', unsafe_allow_html=True)
            st.caption("Historical Temporal Validation — Month 0 train / Month 6 holdout. These metrics are not the current upload's performance.")
            m6_metrics = comparison_results["results"][ml_service.model_name]["calibrated_metrics"]
            st.markdown(kpi_card_html("Active Model", ml_service.model_name, "Selected from Month 6 holdout", "#a855f7"), unsafe_allow_html=True)
            st.markdown(f"""
            - **Precision:** `{m6_metrics['Precision']*100:.1f}%`
            - **Recall:** `{m6_metrics['Recall']*100:.1f}%`
            - **Specificity:** `{m6_metrics['Specificity']*100:.1f}%`
            - **ROC-AUC:** `{m6_metrics['ROC-AUC']:.3f}`
            - **PR-AUC:** `{m6_metrics['PR-AUC']:.3f}`
            - **Brier Score:** `{m6_metrics['Brier Score']:.4f}`
            - **Log Loss:** `{m6_metrics['Log Loss']:.4f}`
            """)
            st.markdown('<div class="dark-card-header" style="margin-top:12px;">Test Type Breakdown</div>', unsafe_allow_html=True)
            st.caption(f"{prediction_source_label}: failure events by test family")
            df_tf = df_m12.copy()
            df_tf["Test_Family"] = df_tf["Fail_Test"].map(_overview_test_family)
            fam_order = ["Scan", "Func", "MBIST", "IDDQ", "AtSpeed"]
            fam_counts = df_tf["Test_Family"].value_counts().reindex(fam_order).fillna(0).reset_index()
            fam_counts.columns = ["Test_Family", "Events"]
            fig_fam = px.bar(
                fam_counts, x="Test_Family", y="Events",
                color="Test_Family",
                color_discrete_sequence=["#8b5cf6", "#38bdf8", "#10b981", "#f59e0b", "#ec4899"],
            )
            fig_fam.update_layout(
                height=220, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"),
                font={"family": "Inter", "color": "#f1f5f9"},
            )
            st.plotly_chart(fig_fam, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# SINGLE EVENT
# =========================================================
elif current_page == "single":
    st.markdown("### Single Event Analysis")
    st.caption("Should this failed event be sent for retest? Probability and recommendation are separate outputs.")

    df_src = datasets["month_12"].copy()
    sel_c1, sel_c2, sel_c3 = st.columns([3, 3, 4])
    with sel_c1:
        dev_list = sorted(df_src["Device_ID"].unique().tolist())
        default_dev = "DEV004" if "DEV004" in dev_list else dev_list[0]
        selected_dev = st.selectbox("Device Identifier", dev_list, index=dev_list.index(default_dev))
    dev_events = df_src[df_src["Device_ID"] == selected_dev]
    with sel_c2:
        event_indices = dev_events.index.tolist()
        event_labels = [f"Event #{row['Failure_Event']} ({row['Fail_Test']})" for _, row in dev_events.iterrows()]
        selected_idx_rel = st.selectbox("Failure Event Index", range(len(event_indices)), format_func=lambda i: event_labels[i])
        selected_row_idx = event_indices[selected_idx_rel]
        event_row = df_src.loc[[selected_row_idx]].copy()
    with sel_c3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        st.markdown("Dataset: **Month 12 (unseen inference)** — outcomes not used.", unsafe_allow_html=True)

    event_dict = event_row[ALL_MODEL_FEATURES].to_dict(orient="records")[0]
    event_dict["Device_ID"] = event_row["Device_ID"].values[0]
    event_dict["Failure_Event"] = int(event_row["Failure_Event"].values[0])
    pred_res = ml_service.predict_single_event(event_dict)
    prob_val = pred_res["probability_retest_beneficial"]
    prob_pct = pred_res["probability_percent"]
    rec = pred_res["recommendation"]
    base_p = float(pred_res.get("probability_base", prob_val))
    adapted_p = float(pred_res.get("probability_adapted", prob_val))
    ol_active = bool(pred_res.get("online_adaptation_active", False))
    explanation = explainer.explain_instance(event_row)

    col_left, col_right = st.columns([5, 5])
    with col_left:
        st.markdown('<div class="dark-card"><div class="dark-card-header">Event Information</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="param-grid">
            <div class="param-item"><div class="param-item-label">Device ID</div><div class="param-item-val" style="color:#38bdf8;">{event_row['Device_ID'].values[0]}</div></div>
            <div class="param-item"><div class="param-item-label">Failure Event</div><div class="param-item-val">{event_row['Failure_Event'].values[0]}</div></div>
            <div class="param-item"><div class="param-item-label">Wafer ID</div><div class="param-item-val">{event_row['Wafer_ID'].values[0]}</div></div>
            <div class="param-item"><div class="param-item-label">ATE Site</div><div class="param-item-val">Site {event_row['ATE_Site'].values[0]}</div></div>
            <div class="param-item"><div class="param-item-label">Fail Test</div><div class="param-item-val" style="color:#a855f7;">{event_row['Fail_Test'].values[0]}</div></div>
            <div class="param-item"><div class="param-item-label">Fail Bin</div><div class="param-item-val">Bin {event_row['Fail_Bin'].values[0]}</div></div>
            <div class="param-item"><div class="param-item-label">Voltage</div><div class="param-item-val">{event_row['Voltage_V'].values[0]:.2f} V</div></div>
            <div class="param-item"><div class="param-item-label">Temperature</div><div class="param-item-val">{event_row['Temperature_C'].values[0]} °C</div></div>
            <div class="param-item"><div class="param-item-label">First Test Time</div><div class="param-item-val">{event_row['First_Test_Time_sec'].values[0]:.1f} s</div></div>
            <div class="param-item"><div class="param-item-label">Est. Retest Time</div><div class="param-item-val">{float(pred_res.get('estimated_retest_time_sec') or 0):.1f} s</div></div>
            <div class="param-item"><div class="param-item-label">Initial Result</div><div class="param-item-val" style="color:#ef4444;">{event_row['First_Result'].values[0]}</div></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="dark-card"><div class="dark-card-header">AI Prediction and Recommendation</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="prob-focal-card">
            <div class="prob-focal-label">RETEST-BENEFIT PROBABILITY</div>
            <div class="prob-focal-value">{prob_pct:.1f}%</div>
            <div style="font-size: 12px; color: #64748b; margin-top: 4px;">
                P(RETEST_BENEFICIAL) = <code style="color:#a855f7;">{prob_val:.4f}</code>
            </div>
            <div style="font-size: 12px; color: #94a3b8; margin-top: 10px; text-align: left; line-height: 1.6;">
                Base Probability: <code style="color:#38bdf8;">{base_p * 100:.1f}%</code><br>
                Adapted Probability: <code style="color:#a855f7;">{adapted_p * 100:.1f}%</code><br>
                Final Recommendation: <b>{rec}</b><br>
                Online adaptation: {"Active" if ol_active else "Not active"}
            </div>
        </div>
        """, unsafe_allow_html=True)
        if rec == "RETEST":
            st.markdown(f"""
            <div class="retest-banner" style="margin-top:14px;">
                <div class="rec-heading" style="color:#10b981;">AI RECOMMENDATION</div>
                <div class="rec-badge-retest">RETEST</div>
                <div class="policy-note">{POLICY_LABEL}<br>If P ≥ {DOCX_REFERENCE_THRESHOLD:.2f} → RETEST. Probability is not modified by this policy.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="skip-banner" style="margin-top:14px;">
                <div class="rec-heading" style="color:#ef4444;">AI RECOMMENDATION</div>
                <div class="rec-badge-skip">DON'T RETEST</div>
                <div class="policy-note">{POLICY_LABEL}<br>If P &lt; {DOCX_REFERENCE_THRESHOLD:.2f} → DON'T RETEST. Probability is not modified by this policy.</div>
            </div>
            """, unsafe_allow_html=True)
        est_sec = float(pred_res.get("estimated_retest_time_sec") or 0.0)
        pred_sec = float(pred_res.get("predicted_retest_time_sec") or 0.0)
        event_cost = seconds_to_cost(pred_sec, _active_cost_per_hour())
        if_run_cost = seconds_to_cost(est_sec, _active_cost_per_hour())
        st.markdown(
            f"""
            <div class="param-item" style="margin-top:14px;">
                <div class="param-item-label">Predicted retest cost</div>
                <div class="param-item-val" style="color:{'#10b981' if rec == 'RETEST' else '#ef4444'};">{format_money(event_cost, ATE_COST_CURRENCY)}</div>
                <div class="policy-note">
                    Estimated duration {est_sec:.1f} s · rate {format_money(_active_cost_per_hour(), ATE_COST_CURRENCY)}/h.
                    {"Charged because AI recommends RETEST." if rec == "RETEST" else f"AI skip → $0. If retested anyway, about {format_money(if_run_cost, ATE_COST_CURRENCY)}."}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="dark-card"><div class="dark-card-header">Which input features contributed to this prediction?</div>', unsafe_allow_html=True)
    sh_col1, sh_col2 = st.columns([5, 5])
    with sh_col1:
        df_shap_plot = pd.DataFrame(explanation["top_features"]).sort_values(by="shap_value", ascending=True)
        fig_shap = px.bar(df_shap_plot, x="shap_value", y="feature", orientation="h", color="shap_value",
                          color_continuous_scale=["#ef4444", "#64748b", "#10b981"],
                          labels={"shap_value": "Contribution to model prediction", "feature": "Feature"})
        fig_shap.update_layout(height=260, margin=dict(l=20, r=20, t=10, b=10),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"),
                               font={"family": "Inter", "color": "#f1f5f9"}, coloraxis_showscale=False)
        st.plotly_chart(fig_shap, use_container_width=True)
    with sh_col2:
        for bullet in explanation["engineering_explanations"][:4]:
            st.markdown(f"• {bullet}")
        st.caption("SHAP shows association with the model prediction, not physical causation.")
    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# MODEL PERFORMANCE — HISTORICAL TEMPORAL VALIDATION
# =========================================================
elif current_page == "models":
    st.markdown("### Historical Temporal Validation")
    st.caption("Train Month 0 → validate Month 6. These metrics are not Month 12 operational performance.")
    st.info(ml_service.selection_reason)
    st.caption(
        f"Classification metrics below use the operational {POLICY_LABEL} "
        f"(threshold={DOCX_REFERENCE_THRESHOLD:.2f}). "
        "ROC-AUC, PR-AUC, Brier, and Log Loss are threshold-free. "
        "A 0.5 evaluation/reporting cutoff may appear in the comparison footnote only and does not override the 30% operational policy."
    )

    m6_metrics = comparison_results["results"][ml_service.model_name]["calibrated_metrics"]
    best_diag = comparison_results["results"][ml_service.model_name]["calibrated_calibration"]

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.markdown(kpi_card_html("Selected Model", ml_service.model_name, "Evidence-based", "#a855f7"), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_card_html("ROC-AUC", f"{m6_metrics['ROC-AUC']:.3f}", "Threshold-free"), unsafe_allow_html=True)
    with k3:
        st.markdown(kpi_card_html("Accuracy", f"{m6_metrics['Accuracy']*100:.1f}%", "At DOCX 30% policy"), unsafe_allow_html=True)
    with k4:
        st.markdown(kpi_card_html("Recall", f"{m6_metrics['Recall']*100:.1f}%", "At DOCX 30% policy"), unsafe_allow_html=True)
    with k5:
        st.markdown(kpi_card_html("Brier Score", f"{best_diag['brier_score']:.4f}", "Lower is better"), unsafe_allow_html=True)
    with k6:
        st.markdown(kpi_card_html("Log Loss", f"{best_diag['log_loss']:.4f}", "Lower is better"), unsafe_allow_html=True)

    st.dataframe(comparison_results["comparison_table"], use_container_width=True)
    st.caption(
        "Accuracy/Precision/Recall/Specificity/F1 in this table use the isolated DOCX-reference 30% policy. "
        "They are not computed at 0.5. ROC-AUC / PR-AUC / Brier / Log Loss do not use a decision threshold."
    )

    reporting = comparison_results["results"][ml_service.model_name].get("calibrated_metrics_reporting_cutoff")
    if reporting:
        with st.expander("Optional evaluation/reporting cutoff (0.5) — not the operational policy"):
            st.write(
                "This 0.5 cutoff is for model-evaluation comparison only and does not produce RETEST / DON'T RETEST."
            )
            st.json({
                "label": reporting.get("classification_threshold_label"),
                "Accuracy": reporting["Accuracy"],
                "Precision": reporting["Precision"],
                "Recall": reporting["Recall"],
                "Specificity": reporting["Specificity"],
            })

    categories = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC-AUC", "PR-AUC"]
    fig_radar = go.Figure()
    colors = {"XGBoost": "#a855f7", "Logistic Regression": "#38bdf8", "Gradient Boosting": "#10b981"}
    for name, color in colors.items():
        m = comparison_results["results"][name]["calibrated_metrics"]
        fig_radar.add_trace(go.Scatterpolar(
            r=[m["Accuracy"], m["Precision"], m["Recall"], m["Specificity"], m["F1"], m["ROC-AUC"], m["PR-AUC"]],
            theta=categories, fill="toself", name=name, line_color=color
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], gridcolor="#1e293b"), bgcolor="rgba(0,0,0,0)"),
        height=320, paper_bgcolor="rgba(0,0,0,0)", font={"family": "Inter", "color": "#f1f5f9"}
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    df_rel = pd.DataFrame(best_diag["bucket_table"])
    fig_rel = go.Figure()
    fig_rel.add_trace(go.Bar(x=df_rel["bucket"], y=df_rel["observed_benefit_rate"] * 100, name="Observed beneficial rate (%)", marker_color="#8b5cf6"))
    fig_rel.add_trace(go.Scatter(x=df_rel["bucket"], y=df_rel["mean_predicted_prob"] * 100, name="Mean predicted probability (%)", mode="lines+markers", line=dict(color="#38bdf8", width=3)))
    fig_rel.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"),
                          font={"family": "Inter", "color": "#f1f5f9"})
    st.plotly_chart(fig_rel, use_container_width=True)

    st.markdown(f"""
    <div class="bottom-info-bar">
        <div class="info-bar-item"><div class="info-bar-icon">1</div><div><div class="info-bar-text-label">Training (selection)</div><div class="info-bar-text-val">Month 0</div></div></div>
        <div class="info-bar-item"><div class="info-bar-icon">2</div><div><div class="info-bar-text-label">Validation holdout</div><div class="info-bar-text-val">Month 6</div></div></div>
        <div class="info-bar-item"><div class="info-bar-icon">3</div><div><div class="info-bar-text-label">Deploy training</div><div class="info-bar-text-val">Month 0 + Month 6</div></div></div>
        <div class="info-bar-item"><div class="info-bar-icon">4</div><div><div class="info-bar-text-label">Inference</div><div class="info-bar-text-val">Month 12</div></div></div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# MODEL INFO
# =========================================================
elif current_page == "info":
    st.markdown("### Model Architecture & Information")
    info = ml_service.get_model_info()
    st.json(info)
    i1, i2 = st.columns(2)
    with i1:
        st.markdown("**Pre-retest feature whitelist**")
        for f in ALL_MODEL_FEATURES:
            st.markdown(f"- `{f}`")
    with i2:
        st.markdown("**Never used as prediction input**")
        st.markdown("""
        - `Ground_Truth`, `Retest_Result`, `Final_Result`, `Retest_Count`
        - `True_Retest_Pass_Probability`, `AI_Retest_Probability`, `AI_Recommendation`
        - `Retest_Time_sec` (actual post-retest duration; never a feature)
        - Device_ID / Failure_Event (tracking only)

        Estimated retest time on the Overview cost cards is a KPI derived from historical
        `Retest_Time_sec` by `Fail_Test`. It is attached after scoring and is not a model input.
        """)


# =========================================================
# DECISION POLICY (NO SLIDER)
# =========================================================
elif current_page == "settings":
    st.markdown("### Decision Policy")
    st.markdown(f"""
    <div class="dark-card">
        <div class="dark-card-header">{POLICY_LABEL}</div>
        <p>The ML model outputs <b>P(RETEST_BENEFICIAL)</b> only. This isolated layer converts that probability into an operational recommendation:</p>
        <p>
            If <code>P(RETEST_BENEFICIAL) ≥ {DOCX_REFERENCE_THRESHOLD:.2f}</code> → <b style="color:#10b981">RETEST</b><br>
            If <code>P(RETEST_BENEFICIAL) &lt; {DOCX_REFERENCE_THRESHOLD:.2f}</code> → <b style="color:#ef4444">DON'T RETEST</b>
        </p>
        <p class="policy-note">
            This 30% rule is referenced from the supplied DOCX analysis. It is <b>not</b> presented as a scientifically
            proven or permanently approved production threshold. It can be replaced in
            <code>retest_ai/decision/decision_policy.py</code> without retraining the model.
            The probability itself is never modified by this policy.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.warning("There is no 50% threshold and no threshold slider in this prototype.")
    st.markdown("### ATE tester cost rate")
    st.caption(
        "Used only for cost KPIs: all-device retest cost vs AI predicted retest cost. "
        "This is a configurable plant input, not a rate from the workbooks, and it does not change the ML probability."
    )
    st.number_input(
        "ATE tester cost per hour (USD)",
        min_value=0.0,
        step=50.0,
        key="ate_cost_per_hour",
        help="Cost = estimated retest seconds × (this rate / 3600).",
    )
