"""Streamlit dashboard comparing the VAN (NPV) of two investment projects."""

from __future__ import annotations

import html
import textwrap
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from finance import (
    Comparison,
    Project,
    ProjectAnalysis,
    compare,
    discounted_cumulative_flows,
    npv,
)

# ---------------------------------------------------------------------------
# Page configuration and visual system
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="VAN Comparator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLOR_A = "#2F6BFF"
COLOR_B = "#F59E0B"
COLOR_POSITIVE = "#0E9F79"
COLOR_NEGATIVE = "#D1435B"
INK = "#12213B"
MUTED = "#64748B"
CHART_FONT = "Avenir Next, Segoe UI, system-ui, sans-serif"

CURRENCIES = {
    "EUR": "€",
    "USD": "$",
    "GBP": "£",
    "CHF": "CHF",
    "MAD": "MAD",
}

DEFAULTS = {
    "A": {
        "name": "Project Alpha",
        "initial": 120_000.0,
        "annual": 35_000.0,
        "duration": 5,
    },
    "B": {
        "name": "Project Beta",
        "initial": 200_000.0,
        "annual": 58_000.0,
        "duration": 6,
    },
}

st.markdown(
    """
    <style>
    :root {
        --ink: #12213B;
        --muted: #64748B;
        --line: #E4EAF3;
        --canvas: #F5F8FC;
        --blue: #2F6BFF;
        --amber: #F59E0B;
        --green: #0E9F79;
        --red: #D1435B;
    }
    html, body, [class*="css"] {
        font-family: "Avenir Next", "Segoe UI", system-ui, -apple-system, sans-serif;
        color: var(--ink);
    }
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(1100px 540px at 8% -10%, rgba(47, 107, 255, 0.085), transparent 62%),
            radial-gradient(900px 460px at 102% 0%, rgba(245, 158, 11, 0.065), transparent 58%),
            #F5F8FC;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="stToolbar"] {
        right: 1rem;
    }
    [data-testid="stAppDeployButton"] {
        display: none !important;
    }
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.96);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.6rem;
    }
    .block-container {
        max-width: 1320px;
        padding-top: 1.7rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3, h4 {
        letter-spacing: -0.02em;
        color: var(--ink);
    }
    p {
        line-height: 1.6;
    }
    div[data-testid="stVerticalBlockBorderWrapper"],
    div[class*="st-key-card_"] {
        background: rgba(255, 255, 255, 0.88) !important;
        border: 1px solid var(--line) !important;
        border-radius: 18px !important;
        box-shadow: 0 10px 30px rgba(17, 43, 84, 0.045) !important;
    }
    div[class*="st-key-card_"] {
        padding: 17px 18px !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        padding: 0.4rem 0.55rem;
    }
    div[data-baseweb="input"] {
        border-radius: 10px;
        border-color: #DCE4EF;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: var(--blue);
        box-shadow: 0 0 0 1px var(--blue);
    }
    div[data-baseweb="select"] > div {
        border-radius: 10px;
        border-color: #DCE4EF;
    }
    div[data-testid="stRadioGroup"] {
        gap: 0.4rem;
    }
    div[data-testid="stRadioGroup"] > div {
        background: #F8FAFD;
        border: 1px solid #E4EAF3;
        border-radius: 999px;
        padding: 1px 9px;
        transition: background 120ms ease, border-color 120ms ease;
    }
    div[data-testid="stRadioGroup"] > div[data-selected="true"] {
        background: #EDF3FF;
        border-color: #B9CDF9;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid #DCE4EF;
        font-weight: 600;
    }
    .stDownloadButton > button {
        color: #214DA8;
        background: #F2F6FF;
        border-color: #CEDCFA;
    }
    .hero {
        position: relative;
        overflow: hidden;
        border-radius: 24px;
        padding: 34px 38px 31px;
        color: white;
        background:
            radial-gradient(420px 260px at 88% -5%, rgba(84, 154, 255, 0.55), transparent 65%),
            radial-gradient(360px 260px at 108% 100%, rgba(245, 158, 11, 0.22), transparent 65%),
            linear-gradient(120deg, #0A1E42 0%, #123676 55%, #1B4FA8 100%);
        box-shadow: 0 20px 55px rgba(10, 35, 79, 0.19);
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    .hero::after {
        content: "";
        position: absolute;
        width: 340px;
        height: 340px;
        right: -90px;
        bottom: -230px;
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 50%;
        box-shadow:
            0 0 0 40px rgba(255, 255, 255, 0.035),
            0 0 0 82px rgba(255, 255, 255, 0.025);
        pointer-events: none;
    }
    .hero-eyebrow {
        position: relative;
        z-index: 1;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #9FC5FF;
        margin-bottom: 12px;
    }
    .hero h1 {
        position: relative;
        z-index: 1;
        max-width: 720px;
        margin: 0;
        color: white;
        font-size: clamp(30px, 3.7vw, 46px);
        line-height: 1.08;
        font-weight: 750;
        letter-spacing: -0.035em;
    }
    .hero p {
        position: relative;
        z-index: 1;
        max-width: 710px;
        margin: 14px 0 0;
        color: #D8E5FA;
        font-size: 14px;
        line-height: 1.65;
    }
    .hero p strong {
        color: white;
        font-weight: 650;
    }
    .hero-chips {
        position: relative;
        z-index: 1;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 22px;
    }
    .hero-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        background: rgba(255, 255, 255, 0.09);
        color: #EAF2FF;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.01em;
    }
    .hero-chip.live::before {
        content: "";
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #52E0B0;
        box-shadow: 0 0 0 3px rgba(82, 224, 176, 0.16);
    }
    .section-head {
        margin: 30px 0 14px;
    }
    .section-kicker {
        font-size: 10px;
        line-height: 1;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: #7A8AA2;
        font-weight: 800;
    }
    .section-title {
        margin-top: 8px;
        font-size: clamp(22px, 2.4vw, 28px);
        line-height: 1.2;
        font-weight: 750;
        letter-spacing: -0.03em;
        color: var(--ink);
    }
    .section-subtitle {
        margin-top: 7px;
        max-width: 760px;
        color: var(--muted);
        font-size: 13px;
        line-height: 1.55;
    }
    .project-kicker {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 10px;
        letter-spacing: 0.16em;
        font-weight: 800;
        text-transform: uppercase;
        margin: 0.2rem 0 0.7rem;
    }
    .project-kicker.a { color: #2457C5; }
    .project-kicker.b { color: #9A5B00; }
    .project-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }
    .project-dot.a { background: var(--blue); box-shadow: 0 0 0 3px rgba(47, 107, 255, 0.12); }
    .project-dot.b { background: var(--amber); box-shadow: 0 0 0 3px rgba(245, 158, 11, 0.15); }
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr)) minmax(0, 1.12fr);
        gap: 15px;
        margin: 24px 0 20px;
    }
    .kpi-card {
        position: relative;
        overflow: hidden;
        min-width: 0;
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 18px 20px 16px;
        box-shadow: 0 10px 30px rgba(17, 43, 84, 0.045);
    }
    .kpi-card::before {
        content: "";
        position: absolute;
        inset: 0 0 auto 0;
        height: 4px;
        background: #CDD8E9;
    }
    .kpi-card.a::before { background: linear-gradient(90deg, #2F6BFF, #7CA4FF); }
    .kpi-card.b::before { background: linear-gradient(90deg, #F59E0B, #F8C76B); }
    .kpi-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        margin-bottom: 8px;
    }
    .kpi-label {
        color: #7A8AA2;
        font-size: 9px;
        font-weight: 800;
        letter-spacing: 0.13em;
        text-transform: uppercase;
    }
    .tag {
        display: inline-flex;
        align-items: center;
        white-space: nowrap;
        padding: 4px 8px;
        border-radius: 999px;
        font-size: 9px;
        font-weight: 800;
        letter-spacing: 0.07em;
        text-transform: uppercase;
    }
    .tag.blue { background: #EAF1FF; color: #2457C5; }
    .tag.amber { background: #FFF3DC; color: #945900; }
    .tag.muted { background: #F1F4F8; color: #738196; }
    .tag.warn { background: #FDECEF; color: #B02A43; }
    .tag.good { background: #E7F8F2; color: #08775A; }
    .tag.dark { background: rgba(255, 255, 255, 0.12); color: #C9F7E7; border: 1px solid rgba(255, 255, 255, 0.14); }
    .tag.danger { background: rgba(255, 151, 170, 0.14); color: #FFD2DA; border: 1px solid rgba(255, 151, 170, 0.22); }
    .kpi-name {
        color: var(--ink);
        font-size: 17px;
        font-weight: 700;
        line-height: 1.3;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .kpi-value {
        margin-top: 5px;
        color: var(--ink);
        font-size: clamp(26px, 2.7vw, 36px);
        font-weight: 750;
        line-height: 1.16;
        letter-spacing: -0.04em;
        font-variant-numeric: tabular-nums;
        overflow-wrap: anywhere;
    }
    .kpi-value.negative { color: var(--red); }
    .kpi-caption {
        margin-top: 3px;
        color: #7A8AA2;
        font-size: 11px;
        line-height: 1.4;
    }
    .kpi-footer {
        display: flex;
        gap: 15px;
        flex-wrap: wrap;
        margin-top: 13px;
        padding-top: 11px;
        border-top: 1px solid #EEF2F7;
        color: #7A8AA2;
        font-size: 11px;
    }
    .kpi-footer b {
        color: #34435B;
        font-weight: 700;
    }
    .verdict-card {
        background:
            radial-gradient(230px 150px at 100% 0%, rgba(70, 137, 255, 0.36), transparent 70%),
            linear-gradient(145deg, #0C2248, #143B7E);
        border-color: rgba(20, 54, 116, 0.2);
        color: white;
    }
    .verdict-card::before { background: linear-gradient(90deg, #53E2B1, #8DF3D3); }
    .verdict-card.no-go {
        background:
            radial-gradient(230px 150px at 100% 0%, rgba(209, 67, 91, 0.26), transparent 70%),
            linear-gradient(145deg, #281620, #4B2030);
    }
    .verdict-card.no-go::before { background: linear-gradient(90deg, #F07A91, #FFB7C5); }
    .verdict-card.tie::before { background: linear-gradient(90deg, #7CB1FF, #B9D4FF); }
    .verdict-card .kpi-label { color: #A8C3EC; }
    .verdict-name {
        margin-top: 6px;
        color: white;
        font-size: 25px;
        line-height: 1.2;
        font-weight: 750;
        letter-spacing: -0.03em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .verdict-copy {
        margin-top: 7px;
        color: #D6E4F9;
        font-size: 12px;
        line-height: 1.55;
        min-height: 37px;
    }
    .verdict-footer {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        margin-top: 13px;
        padding-top: 11px;
        border-top: 1px solid rgba(255, 255, 255, 0.14);
        color: #B8CDEB;
        font-size: 11px;
    }
    .verdict-footer b { color: white; font-weight: 700; }
    .chart-head {
        padding: 6px 4px 0;
        min-height: 62px;
    }
    .chart-kicker {
        color: #8391A5;
        font-size: 9px;
        font-weight: 800;
        letter-spacing: 0.13em;
        text-transform: uppercase;
    }
    .chart-title {
        margin-top: 4px;
        color: var(--ink);
        font-size: 16px;
        line-height: 1.3;
        font-weight: 750;
        letter-spacing: -0.015em;
    }
    .chart-copy {
        margin-top: 2px;
        color: #7A8AA2;
        font-size: 11px;
        line-height: 1.4;
    }
    .table-wrap {
        width: 100%;
        overflow-x: auto;
        margin-top: 4px;
    }
    .detail-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 12.5px;
    }
    .detail-table th {
        text-align: left;
        padding: 10px 12px;
        border-bottom: 1px solid #E4EAF3;
        background: #F8FAFD;
        color: #7A8AA2;
        font-size: 9px;
        font-weight: 800;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        white-space: nowrap;
    }
    .detail-table th.num, .detail-table td.num {
        text-align: right;
        font-variant-numeric: tabular-nums;
    }
    .detail-table th.a { color: #2457C5; }
    .detail-table th.b { color: #945900; }
    .detail-table td {
        padding: 10px 12px;
        border-bottom: 1px solid #EFF3F8;
        color: #3C4B62;
        white-space: nowrap;
    }
    .detail-table tr:last-child td { border-bottom: none; }
    .detail-table td.metric {
        color: #20304A;
        font-weight: 600;
    }
    .detail-table tr:hover td { background: #FAFCFF; }
    .status-pill {
        display: inline-flex;
        border-radius: 999px;
        padding: 3px 7px;
        font-size: 9px;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .status-pill.accept { background: #E7F8F2; color: #08775A; }
    .status-pill.reject { background: #FDECEF; color: #B02A43; }
    .status-pill.even { background: #F1F4F8; color: #66758B; }
    .sidebar-brand {
        margin-bottom: 2px;
        color: #0D2145;
        font-size: 22px;
        font-weight: 800;
        letter-spacing: -0.04em;
        line-height: 1.15;
    }
    .sidebar-brand span { color: var(--blue); }
    .sidebar-sub {
        color: #8290A4;
        font-size: 11px;
        margin-bottom: 20px;
    }
    .sidebar-note {
        margin-top: 10px;
        padding: 13px 14px;
        border-radius: 13px;
        background: #F6F9FE;
        border: 1px solid #E5ECF7;
        color: #586A82;
        font-size: 11.5px;
        line-height: 1.55;
    }
    .sidebar-note b { color: #20304A; }
    .sidebar-note .mini-title {
        display: block;
        margin-bottom: 4px;
        color: #7A8AA2;
        font-size: 9px;
        font-weight: 800;
        letter-spacing: 0.11em;
        text-transform: uppercase;
    }
    .method-note {
        color: var(--muted);
        font-size: 12px;
        line-height: 1.65;
    }
    .footer {
        margin-top: 28px;
        padding-top: 15px;
        border-top: 1px solid #E4EAF3;
        color: #8A97A9;
        font-size: 10.5px;
        line-height: 1.6;
    }
    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.64);
        border: 1px solid var(--line);
        border-radius: 14px;
        margin-top: 16px;
    }
    .js-plotly-plot .plotly .modebar { display: none !important; }
    @media (max-width: 980px) {
        .kpi-grid { grid-template-columns: 1fr 1fr; }
        .verdict-card { grid-column: 1 / -1; }
    }
    @media (max-width: 640px) {
        .block-container { padding-top: 0.9rem; }
        .hero { padding: 26px 22px 24px; border-radius: 19px; }
        .hero p { font-size: 13px; }
        .kpi-grid { grid-template-columns: 1fr; gap: 11px; }
        .verdict-card { grid-column: auto; }
        .kpi-card { padding: 16px 17px; }
        .section-head { margin-top: 24px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _escape(value: object) -> str:
    return html.escape(str(value))


def _money(value: float, symbol: str, *, compact: bool = False) -> str:
    if not np.isfinite(value):
        return "—"
    sign = "-" if value < 0 else ""
    absolute = abs(float(value))
    prefix = symbol if len(symbol) == 1 else f"{symbol} "
    if compact:
        if absolute >= 1_000_000_000:
            body = f"{absolute / 1_000_000_000:,.1f}B"
        elif absolute >= 1_000_000:
            body = f"{absolute / 1_000_000:,.1f}M"
        elif absolute >= 1_000:
            body = f"{absolute / 1_000:,.1f}k"
        else:
            body = f"{absolute:,.0f}"
    else:
        body = f"{absolute:,.0f}"
    return f"{sign}{prefix}{body}"


def _percent(value: float | None, *, digits: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value * 100:,.{digits}f}%"


def _payback(value: float | None) -> str:
    if value is None:
        return "Not recovered"
    return f"{value:,.1f} yrs"


def _wrap_label(value: str, width: int = 23) -> str:
    lines = textwrap.wrap(value, width=width, break_long_words=True)
    return "<br>".join(lines[:2]) if lines else value


def _axis_prefix(symbol: str) -> str:
    return symbol if len(symbol) == 1 else f"{symbol} "


def _chart_layout(
    figure: go.Figure,
    *,
    height: int = 330,
    x_title: str | None = None,
    y_title: str | None = None,
    showlegend: bool = False,
    symbol: str | None = None,
) -> go.Figure:
    figure.update_layout(
        height=height,
        margin=dict(l=6, r=8, t=10, b=6),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=CHART_FONT, size=11, color=MUTED),
        showlegend=showlegend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10.5, color="#526178"),
        ),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="#DFE6F0",
            font=dict(family=CHART_FONT, color=INK, size=11),
        ),
        hovermode="x unified",
        dragmode=False,
    )
    figure.update_xaxes(
        title_text=x_title,
        title_font=dict(size=10, color="#8995A6"),
        showgrid=False,
        showline=False,
        zeroline=False,
        ticks="",
        tickfont=dict(size=10.5, color="#7A8AA2"),
    )
    figure.update_yaxes(
        title_text=y_title,
        title_font=dict(size=10, color="#8995A6"),
        showgrid=True,
        gridcolor="#EDF1F7",
        gridwidth=1,
        showline=False,
        zeroline=False,
        ticks="",
        tickfont=dict(size=10.5, color="#7A8AA2"),
        tickformat=",.0f",
        tickprefix=_axis_prefix(symbol) if symbol else None,
    )
    return figure


def _chart_head(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="chart-head">
            <div class="chart-kicker">{_escape(kicker)}</div>
            <div class="chart-title">{_escape(title)}</div>
            <div class="chart-copy">{_escape(copy)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------


with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">VAN<span>·</span>Studio</div>'
        '<div class="sidebar-sub">Two-project value comparator</div>',
        unsafe_allow_html=True,
    )
    st.markdown("##### Model settings")

    discount_rate_pct = st.number_input(
        "Discount rate (annual)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=0.5,
        format="%.1f",
        help=(
            "The annual rate used to bring future cash flows back to today. "
            "It usually represents the cost of capital or the required return."
        ),
        key="discount_rate",
    )
    currency = st.selectbox(
        "Display currency",
        options=list(CURRENCIES),
        index=0,
        help="Changing the currency only changes the display symbol; the cash flow values stay the same.",
        key="currency",
    )
    symbol = CURRENCIES[currency]

    st.markdown(
        """
        <div class="sidebar-note">
            <span class="mini-title">Decision rule</span>
            Accept a project when <b>VAN &gt; 0</b>. Between mutually exclusive
            projects, choose the one with the <b>highest VAN</b>.
        </div>
        <div class="sidebar-note">
            <span class="mini-title">Timing convention</span>
            Initial outlay at <b>t = 0</b>; all net cash flows at the
            <b>end of each year</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Reset project inputs", use_container_width=True):
        for state_key in list(st.session_state):
            if state_key.startswith(
                ("name_", "initial_", "duration_", "mode_", "annual_", "cashflows_")
            ):
                del st.session_state[state_key]
        st.rerun()

    st.caption("Built with Streamlit · results update as you type")


discount_rate = discount_rate_pct / 100.0


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------


st.markdown(
    f"""
    <section class="hero">
        <div class="hero-eyebrow">Investment decision workspace</div>
        <h1>Compare the value of two projects.<br>See what drives the decision.</h1>
        <p>
            Model each project, compare its <strong>VAN (net present value)</strong>,
            and stress-test the ranking against your discount rate. A clearer
            case for where to invest.
        </p>
        <div class="hero-chips">
            <span class="hero-chip live">Live model</span>
            <span class="hero-chip">Discount rate · {discount_rate_pct:,.1f}%</span>
            <span class="hero-chip">Currency · {currency}</span>
            <span class="hero-chip">End-of-year cash flows</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Project inputs
# ---------------------------------------------------------------------------


def _cash_flow_inputs(
    project_key: Literal["A", "B"],
    duration: int,
    *,
    annual_step: float,
) -> tuple[tuple[float, ...], str | None]:
    default = DEFAULTS[project_key]
    mode = st.radio(
        "Cash-flow pattern",
        options=("Flat", "Year by year"),
        horizontal=True,
        key=f"mode_{project_key}",
        help=(
            "Flat repeats one annual net cash flow. Year by year lets you model "
            "ramp-ups, maintenance costs, or other uneven cash flows."
        ),
    )

    if mode == "Flat":
        annual = st.number_input(
            "Annual net cash flow",
            value=default["annual"],
            step=annual_step,
            format="%.0f",
            help="Net operating cash flow for every year of the project. Enter a negative amount if needed.",
            key=f"annual_{project_key}",
        )
        return tuple(float(annual) for _ in range(duration)), None

    fallback = float(st.session_state.get(f"annual_{project_key}", default["annual"]))
    flow_table = pd.DataFrame(
        {
            "Year": np.arange(1, duration + 1, dtype=int),
            "Net cash flow": [fallback] * duration,
        }
    )
    edited = st.data_editor(
        flow_table,
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        height=min(302, 78 + 34 * duration),
        key=f"cashflows_{project_key}_{duration}",
        column_config={
            "Year": st.column_config.NumberColumn("Year", format="%d", width="small", disabled=True),
            "Net cash flow": st.column_config.NumberColumn(
                "Net cash flow",
                format="%.0f",
                width="medium",
                help="Net amount received at the end of this year.",
            ),
        },
    )
    values = pd.to_numeric(edited["Net cash flow"], errors="coerce")
    if values.isna().any():
        return (), "Every custom cash-flow year must contain a number."
    if len(values) != duration:
        return (), "The cash-flow table must contain one row per year."
    st.caption("Changing the project life regenerates this table from the flat annual amount.")
    return tuple(float(value) for value in values), None


def _project_card(project_key: Literal["A", "B"]) -> tuple[str, float, int, tuple[float, ...], str | None]:
    default = DEFAULTS[project_key]
    accent = "a" if project_key == "A" else "b"
    step = 5_000.0
    annual_step = 1_000.0

    with st.container(border=True, key=f"card_project_{project_key.lower()}"):
        st.markdown(
            f'<div class="project-kicker {accent}"><span class="project-dot {accent}"></span>'
            f"Project {project_key}</div>",
            unsafe_allow_html=True,
        )
        name = st.text_input(
            "Project name",
            value=default["name"],
            key=f"name_{project_key}",
            placeholder=f"Project {project_key}",
        )
        left, right = st.columns([1.15, 1], gap="medium")
        with left:
            initial = st.number_input(
                "Initial investment",
                min_value=0.0,
                value=default["initial"],
                step=step,
                format="%.0f",
                help="Cash outlay at t=0. Enter a positive amount.",
                key=f"initial_{project_key}",
            )
        with right:
            duration = int(
                st.number_input(
                    "Project life (years)",
                    min_value=1,
                    max_value=25,
                    value=default["duration"],
                    step=1,
                    help="Cash flows are modelled for this many years.",
                    key=f"duration_{project_key}",
                )
            )
        cash_flows, error = _cash_flow_inputs(
            project_key,
            duration,
            annual_step=annual_step,
        )
    return name, float(initial), duration, cash_flows, error


st.markdown(
    """
    <div class="section-head">
        <div class="section-kicker">01 · Define the scenarios</div>
        <div class="section-title">Build the two project cases</div>
        <div class="section-subtitle">
            Enter net cash flows — after operating costs and taxes — in today's
            currency. Use “Year by year” for uneven streams.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

input_col_a, input_col_b = st.columns(2, gap="large")
with input_col_a:
    name_a, initial_a, duration_a, flows_a, error_a = _project_card("A")
with input_col_b:
    name_b, initial_b, duration_b, flows_b, error_b = _project_card("B")

input_errors = [message for message in (error_a, error_b) if message]
if input_errors:
    st.error(" ".join(input_errors))
    st.stop()

try:
    project_a = Project(name_a.strip() or "Project A", initial_a, flows_a)
    project_b = Project(name_b.strip() or "Project B", initial_b, flows_b)
    comparison = compare(project_a, project_b, discount_rate)
except ValueError as error:
    st.error(f"Please check the inputs: {error}")
    st.stop()


# ---------------------------------------------------------------------------
# KPI cards and verdict
# ---------------------------------------------------------------------------


def _npv_tag(analysis: ProjectAnalysis, is_higher: bool, project_key: str) -> tuple[str, str]:
    if analysis.npv <= 0:
        return "Below zero", "warn"
    if is_higher:
        return "Higher VAN", "blue" if project_key == "A" else "amber"
    return "Lower VAN", "muted"


def _kpi_card(
    analysis: ProjectAnalysis,
    *,
    project_key: Literal["A", "B"],
    is_higher: bool,
) -> str:
    tag_text, tag_class = _npv_tag(analysis, is_higher, project_key)
    value_class = " negative" if analysis.npv < 0 else ""
    return f"""
    <div class="kpi-card {project_key.lower()}">
        <div class="kpi-top">
            <span class="kpi-label">Project {project_key} · VAN</span>
            <span class="tag {tag_class}">{tag_text}</span>
        </div>
        <div class="kpi-name" title="{_escape(analysis.project.name)}">{_escape(analysis.project.name)}</div>
        <div class="kpi-value{value_class}">{_escape(_money(analysis.npv, symbol))}</div>
        <div class="kpi-caption">at a {discount_rate_pct:,.1f}% annual discount rate</div>
        <div class="kpi-footer">
            <span>IRR <b>{_escape(_percent(analysis.irr))}</b></span>
            <span>Disc. payback <b>{_escape(_payback(analysis.discounted_payback))}</b></span>
        </div>
    </div>
    """


def _verdict_card(comparison: Comparison) -> str:
    difference = abs(comparison.npv_difference)
    tie = difference < 0.5

    if tie and comparison.project_a.npv > 0:
        card_class = "verdict-card tie"
        badge = '<span class="tag dark">Tie</span>'
        headline = "Equal VAN"
        copy = (
            f"Both projects create the same value at {discount_rate_pct:,.1f}%. "
            "Use qualitative factors and risk to break the tie."
        )
        footer = (
            f"<span>VAN each</span><b>{_escape(_money(comparison.project_a.npv, symbol))}</b>"
        )
    elif comparison.recommended is not None:
        winner = comparison.recommended
        other = comparison.lower_npv
        card_class = "verdict-card"
        badge = '<span class="tag dark">Recommended</span>'
        headline = winner.project.name
        copy = (
            f"Creates {_money(difference, symbol)} more VAN than "
            f"{other.project.name} at {discount_rate_pct:,.1f}%."
        )
        cross = _percent(comparison.crossover_rate) if comparison.crossover_rate is not None else "n/a"
        footer = f"<span>Ranking flips at</span><b>{cross}</b>"
    else:
        card_class = "verdict-card no-go"
        badge = '<span class="tag danger">No-go</span>'
        headline = "Neither project"
        copy = (
            f"Both VANs are below zero at {discount_rate_pct:,.1f}%. "
            "Under the NPV rule, both should be rejected."
        )
        closest = comparison.higher_npv
        footer = (
            f"<span>Closest to break-even</span><b>{_escape(closest.project.name)}"
            f" · {_escape(_money(closest.npv, symbol))}</b>"
        )

    if tie and comparison.project_a.npv <= 0:
        headline = "Neither project"
        badge = '<span class="tag danger">No-go</span>'
        copy = (
            f"Both projects have a VAN below zero at {discount_rate_pct:,.1f}%. "
            "The NPV rule says reject both."
        )

    return f"""
    <div class="{card_class}">
        <div class="kpi-top">
            <span class="kpi-label">Decision</span>
            {badge}
        </div>
        <div class="verdict-name" title="{_escape(headline)}">{_escape(headline)}</div>
        <div class="verdict-copy">{_escape(copy)}</div>
        <div class="verdict-footer">{footer}</div>
    </div>
    """


a_is_higher = comparison.project_a.npv >= comparison.project_b.npv
st.markdown(
    f"""
    <div class="kpi-grid">
        {_kpi_card(comparison.project_a, project_key="A", is_higher=a_is_higher)}
        {_kpi_card(comparison.project_b, project_key="B", is_higher=not a_is_higher)}
        {_verdict_card(comparison)}
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def _build_npv_bar(comparison: Comparison) -> go.Figure:
    analyses = [comparison.project_a, comparison.project_b]
    values = [analysis.npv for analysis in analyses]
    labels = [_wrap_label(analysis.project.name) for analysis in analyses]
    colors = [COLOR_A, COLOR_B]
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker=dict(color=colors, cornerradius=7, line=dict(width=0)),
            text=[_money(value, symbol, compact=True) for value in values],
            textposition="outside",
            textfont=dict(size=11.5, color="#34435B"),
            cliponaxis=False,
            customdata=[
                [_money(value, symbol), analysis.project.name]
                for value, analysis in zip(values, analyses)
            ],
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                f"VAN: %{{customdata[0]}}<extra></extra>"
            ),
            width=0.52,
        )
    )
    figure.add_hline(y=0, line_color="#CBD5E1", line_width=1.2)
    low, high = min(values + [0.0]), max(values + [0.0])
    span = max(high - low, 1.0)
    figure.update_yaxes(range=[low - 0.24 * span, high + 0.24 * span])
    figure.update_xaxes(tickfont=dict(size=11, color="#526178"))
    return _chart_layout(figure, height=317, symbol=symbol)


def _build_sensitivity(comparison: Comparison) -> go.Figure:
    upper_rate = min(1.0, max(0.30, discount_rate + 0.15, discount_rate * 1.6))
    rates = np.linspace(0.0, upper_rate, 241)
    figure = go.Figure()

    for analysis, color in ((comparison.project_a, COLOR_A), (comparison.project_b, COLOR_B)):
        values = [npv(analysis.project, float(rate)) for rate in rates]
        figure.add_trace(
            go.Scatter(
                x=rates * 100,
                y=values,
                mode="lines",
                name=analysis.project.name,
                line=dict(color=color, width=2.8),
                hovertemplate=(
                    f"<b>{_escape(analysis.project.name)}</b><br>"
                    "Rate: %{x:.1f}%<br>"
                    "VAN: %{y:,.0f}<extra></extra>"
                ),
            )
        )

    figure.add_vline(
        x=discount_rate * 100,
        line_dash="dot",
        line_color="#8A98AB",
        line_width=1.2,
    )
    figure.add_hline(y=0, line_color="#CBD5E1", line_width=1.2)
    figure.add_annotation(
        x=discount_rate * 100,
        y=1,
        yref="paper",
        text=f"Your rate<br><b>{discount_rate_pct:,.1f}%</b>",
        showarrow=False,
        yanchor="top",
        font=dict(size=9.5, color="#66758B"),
        bgcolor="rgba(255,255,255,0.82)",
        borderpad=3,
        xanchor="left",
        xshift=5,
    )

    cross = comparison.crossover_rate
    if cross is not None and cross <= upper_rate:
        cross_value = npv(comparison.project_a.project, cross)
        figure.add_trace(
            go.Scatter(
                x=[cross * 100],
                y=[cross_value],
                mode="markers",
                name="Crossover",
                marker=dict(
                    symbol="star",
                    size=13,
                    color=COLOR_POSITIVE,
                    line=dict(color="white", width=1.2),
                ),
                hovertemplate=(
                    "Crossover rate: %{x:.1f}%<br>"
                    "VAN: %{y:,.0f}<extra></extra>"
                ),
            )
        )
        figure.add_annotation(
            x=cross * 100,
            y=cross_value,
            text=f"Crossover {cross * 100:,.1f}%",
            showarrow=True,
            arrowhead=0,
            arrowcolor="#83A0AB",
            ax=38,
            ay=-28,
            font=dict(size=9, color="#08775A"),
            bgcolor="rgba(231,248,242,0.92)",
            borderpad=3,
        )

    figure.update_xaxes(
        title_text="Annual discount rate",
        ticksuffix="%",
        tickformat=",.0f",
        range=[0, upper_rate * 100],
    )
    return _chart_layout(
        figure,
        height=317,
        showlegend=True,
        symbol=symbol,
    )


def _build_cumulative(comparison: Comparison) -> go.Figure:
    figure = go.Figure()
    for analysis, color in ((comparison.project_a, COLOR_A), (comparison.project_b, COLOR_B)):
        values = discounted_cumulative_flows(analysis.project, discount_rate)
        years = list(range(len(values)))
        figure.add_trace(
            go.Scatter(
                x=years,
                y=values,
                mode="lines+markers",
                name=analysis.project.name,
                line=dict(color=color, width=2.6),
                marker=dict(size=4.5, color=color),
                hovertemplate=(
                    f"<b>{_escape(analysis.project.name)}</b><br>"
                    "Year: %{x}<br>"
                    "Cumulative VAN: %{y:,.0f}<extra></extra>"
                ),
            )
        )

    figure.add_hline(y=0, line_color="#CBD5E1", line_width=1.2)
    max_duration = max(comparison.project_a.project.duration, comparison.project_b.project.duration)
    figure.update_xaxes(
        title_text="Year",
        dtick=1 if max_duration <= 12 else 2,
        range=[-0.1, max_duration + 0.1],
    )
    return _chart_layout(figure, height=310, showlegend=True, symbol=symbol)


def _build_cash_flow_bars(comparison: Comparison) -> go.Figure:
    max_duration = max(comparison.project_a.project.duration, comparison.project_b.project.duration)
    years = list(range(1, max_duration + 1))
    flows_a = list(comparison.project_a.project.cash_flows) + [None] * (
        max_duration - comparison.project_a.project.duration
    )
    flows_b = list(comparison.project_b.project.cash_flows) + [None] * (
        max_duration - comparison.project_b.project.duration
    )
    figure = go.Figure()
    for name, values, color in (
        (comparison.project_a.project.name, flows_a, COLOR_A),
        (comparison.project_b.project.name, flows_b, COLOR_B),
    ):
        figure.add_trace(
            go.Bar(
                x=years,
                y=values,
                name=name,
                marker=dict(color=color, line=dict(width=0), cornerradius=4),
                hovertemplate=(
                    f"<b>{_escape(name)}</b><br>"
                    "Year: %{x}<br>"
                    "Net cash flow: %{y:,.0f}<extra></extra>"
                ),
            )
        )
    figure.update_layout(barmode="group", bargap=0.24, bargroupgap=0.08)
    figure.add_hline(y=0, line_color="#CBD5E1", line_width=1.2)
    figure.update_xaxes(title_text="Year", dtick=1 if max_duration <= 12 else 2)
    return _chart_layout(figure, height=310, showlegend=True, symbol=symbol)


st.markdown(
    """
    <div class="section-head">
        <div class="section-kicker">02 · Read the signal</div>
        <div class="section-title">Compare the VAN, then test the ranking</div>
        <div class="section-subtitle">
            The bars show today's value. The sensitivity curve shows how the
            preferred project changes with the rate used to discount cash flows.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

chart_left, chart_right = st.columns([0.9, 1.25], gap="medium")
with chart_left:
    with st.container(border=True, key="card_chart_npv"):
        _chart_head(
            "Snapshot",
            f"VAN at {discount_rate_pct:,.1f}%",
            "Value created in today's money. A negative bar destroys value.",
        )
        st.plotly_chart(
            _build_npv_bar(comparison),
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )

with chart_right:
    with st.container(border=True, key="card_chart_sensitivity"):
        _chart_head(
            "Sensitivity",
            "Which project leads at different rates?",
            "The star marks the rate at which both projects have the same VAN.",
        )
        st.plotly_chart(
            _build_sensitivity(comparison),
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )

bottom_left, bottom_right = st.columns(2, gap="medium")
with bottom_left:
    with st.container(border=True, key="card_chart_cumulative"):
        _chart_head(
            "Trajectory",
            "When does the project pay back?",
            "Cumulative discounted cash flow. Crossing zero is the discounted payback point.",
        )
        st.plotly_chart(
            _build_cumulative(comparison),
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )

with bottom_right:
    with st.container(border=True, key="card_chart_cashflow"):
        _chart_head(
            "Cash flows",
            "Year-by-year net cash flow",
            "Initial outlays fund the project before these annual flows begin.",
        )
        st.plotly_chart(
            _build_cash_flow_bars(comparison),
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )


# ---------------------------------------------------------------------------
# Detailed comparison table
# ---------------------------------------------------------------------------


def _decision_pill(analysis: ProjectAnalysis) -> str:
    if analysis.npv > 0:
        return '<span class="status-pill accept">Accept</span>'
    if analysis.npv < 0:
        return '<span class="status-pill reject">Reject</span>'
    return '<span class="status-pill even">Break-even</span>'


def _detail_table(comparison: Comparison) -> str:
    a = comparison.project_a
    b = comparison.project_b
    rows = [
        ("Initial investment", _money(a.project.initial_investment, symbol), _money(b.project.initial_investment, symbol)),
        ("Project life", f"{a.project.duration} years", f"{b.project.duration} years"),
        (f"VAN at {discount_rate_pct:,.1f}%", _money(a.npv, symbol), _money(b.npv, symbol)),
        ("IRR (TRI)", _percent(a.irr), _percent(b.irr)),
        (
            "Profitability index",
            f"{a.profitability_index:,.2f}x" if a.profitability_index is not None else "n/a",
            f"{b.profitability_index:,.2f}x" if b.profitability_index is not None else "n/a",
        ),
        ("Discounted payback", _payback(a.discounted_payback), _payback(b.discounted_payback)),
        ("PV of future cash flows", _money(a.pv_of_cash_flows, symbol), _money(b.pv_of_cash_flows, symbol)),
        (
            "Total net cash flow (undiscounted)",
            _money(a.project.total_net_cash_flow, symbol),
            _money(b.project.total_net_cash_flow, symbol),
        ),
        ("VAN decision", _decision_pill(a), _decision_pill(b)),
    ]
    body = "".join(
        f'<tr><td class="metric">{_escape(metric)}</td>'
        f'<td class="num">{value_a if metric == "VAN decision" else _escape(value_a)}</td>'
        f'<td class="num">{value_b if metric == "VAN decision" else _escape(value_b)}</td></tr>'
        for metric, value_a, value_b in rows
    )
    return f"""
    <div class="table-wrap">
        <table class="detail-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th class="num a">{_escape(a.project.name)}</th>
                    <th class="num b">{_escape(b.project.name)}</th>
                </tr>
            </thead>
            <tbody>{body}</tbody>
        </table>
    </div>
    """


with st.container(border=True, key="card_detail"):
    _chart_head(
        "Project detail",
        "Full comparison",
        "IRR is shown for conventional cash-flow patterns; the VAN remains the primary decision metric.",
    )
    st.markdown(_detail_table(comparison), unsafe_allow_html=True)

    csv_columns = {
        "Currency": [currency, currency],
        "Project": [comparison.project_a.project.name, comparison.project_b.project.name],
        "Initial investment": [
            comparison.project_a.project.initial_investment,
            comparison.project_b.project.initial_investment,
        ],
        "Duration (years)": [
            comparison.project_a.project.duration,
            comparison.project_b.project.duration,
        ],
        "Discount rate": [discount_rate, discount_rate],
        "VAN": [comparison.project_a.npv, comparison.project_b.npv],
        "IRR": [comparison.project_a.irr, comparison.project_b.irr],
        "Profitability index": [
            comparison.project_a.profitability_index,
            comparison.project_b.profitability_index,
        ],
        "Discounted payback (years)": [
            comparison.project_a.discounted_payback,
            comparison.project_b.discounted_payback,
        ],
        "PV of future cash flows": [
            comparison.project_a.pv_of_cash_flows,
            comparison.project_b.pv_of_cash_flows,
        ],
        "Total net cash flow (undiscounted)": [
            comparison.project_a.project.total_net_cash_flow,
            comparison.project_b.project.total_net_cash_flow,
        ],
    }
    csv_data = pd.DataFrame(csv_columns).to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download comparison as CSV",
        data=csv_data,
        file_name="van_comparison.csv",
        mime="text/csv",
        use_container_width=False,
    )


with st.expander("Methodology, assumptions & limitations"):
    formula_col, note_col = st.columns([0.9, 1.45], gap="large")
    with formula_col:
        st.markdown("**Discounted value**")
        st.latex(
            r"VAN = -I_0 + \sum_{t=1}^{n}\frac{CF_t}{(1+r)^t}"
        )
        st.markdown("**Internal rate of return**")
        st.latex(r"VAN(r_{TRI}) = 0")
    with note_col:
        st.markdown(
            """
            <div class="method-note">
                <b>What the numbers assume.</b> The initial investment occurs at
                <i>t = 0</i>, cash flows occur at year-end, and the same discount
                rate applies to both projects. Cash flows are net of operating
                costs and taxes. The projects are treated as mutually exclusive,
                so the decision rule is to select the highest positive VAN.<br><br>
                <b>IRR.</b> The IRR is useful for conventional cash-flow profiles
                (one initial outflow followed by inflows), but can be misleading
                with non-conventional flows. The VAN is the primary metric here.<br><br>
                <b>Crossover rate.</b> The positive rate at which both projects have
                the same VAN. It is searched from 0% to 200%; if no crossover is
                found, one project dominates over that range.<br><br>
                <b>Different lives and scale.</b> This comparison does not replace
                projects with a common-life assumption or adjust for different
                risk. Use project-specific discount rates if risk differs.
                This tool is for decision support, not financial advice.
            </div>
            """,
            unsafe_allow_html=True,
        )


st.markdown(
    f"""
    <div class="footer">
        VAN·Studio · Net present value comparison · All figures in {_escape(currency)}.
        The analysis updates automatically as inputs change.
    </div>
    """,
    unsafe_allow_html=True,
)
