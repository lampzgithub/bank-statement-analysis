from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analyzer.categorize import (
    beneficiary_summary,
    enrich_transactions,
    monthly_summary,
    summarize,
    top_merchants,
)
from analyzer.parser import parse_statement_pdf


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Bank Statement Analyzer",
    page_icon="\U0001f4ca",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS \u2014 dark glass-morphism theme
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Global */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0a0f 0%, #0d0d1a 50%, #0a0a0f 100%); }
header[data-testid="stHeader"] { background: rgba(10,10,15,0.8); backdrop-filter: blur(10px); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #12121f 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] .stMarkdown p { color: #9ca3af; }
section[data-testid="stSidebar"] h2 { color: #e2e8f0 !important; }

/* Hide default metric styling, we use custom HTML cards */
[data-testid="stMetricValue"] { display: none; }
[data-testid="stMetricLabel"] { display: none; }
div[data-testid="metric-container"] { display: none; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(18, 18, 30, 0.6);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid rgba(255,255,255,0.06);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #9ca3af;
    font-weight: 500;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background: rgba(59, 130, 246, 0.15) !important;
    color: #60a5fa !important;
    border-bottom: none !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    overflow: hidden;
}

/* Buttons */
.stButton > button, .stFormSubmitButton > button {
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    box-shadow: 0 4px 20px rgba(59,130,246,0.3) !important;
    transform: translateY(-1px) !important;
}

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(135deg, #1e293b, #0f172a) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}

/* Expander */
.streamlit-expanderHeader { background: rgba(18,18,30,0.6); border-radius: 8px; }

/* File uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(59,130,246,0.3);
    border-radius: 12px;
    padding: 8px;
}

/* Success/info/warning boxes */
.stAlert { border-radius: 10px !important; }

/* Custom scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #475569; }

/* Plotly charts in dark */
.js-plotly-plot .plotly .modebar { background: transparent !important; }

/* Radio buttons */
.stRadio > div { gap: 0.5rem; }
.stRadio [data-baseweb="radio"] label { color: #d1d5db; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _indian_format(n: float) -> str:
    is_negative = n < 0
    n = abs(n)
    integer_part = int(n)
    decimal_part = f"{n - integer_part:.2f}"[1:]
    s = str(integer_part)
    if len(s) <= 3:
        result = s
    else:
        result = s[-3:]
        s = s[:-3]
        while s:
            result = s[-2:] + "," + result
            s = s[:-2]
    formatted = result + decimal_part
    return f"-{formatted}" if is_negative else formatted


def money(value: float) -> str:
    return f"\u20b9{_indian_format(value)}"


def _file_hash(buf: bytes) -> str:
    return hashlib.md5(buf).hexdigest()


def _kpi_card(label: str, value: str, color: str = "#3b82f6", icon: str = "") -> str:
    return f"""
    <div style="
        background: linear-gradient(135deg, rgba(18,18,30,0.9), rgba(15,15,25,0.95));
        border: 1px solid rgba(255,255,255,0.06);
        border-left: 3px solid {color};
        border-radius: 12px;
        padding: 16px 20px;
        backdrop-filter: blur(10px);
    ">
        <div style="color: #6b7280; font-size: 0.75rem; font-weight: 500;
                    text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">
            {icon} {label}
        </div>
        <div style="color: #f1f5f9; font-size: 1.25rem; font-weight: 700;">
            {value}
        </div>
    </div>
    """


def _kpi_card_with_delta(label: str, value: str, delta: str,
                         color: str = "#3b82f6", delta_color: str = "#22c55e") -> str:
    return f"""
    <div style="
        background: linear-gradient(135deg, rgba(18,18,30,0.9), rgba(15,15,25,0.95));
        border: 1px solid rgba(255,255,255,0.06);
        border-left: 3px solid {color};
        border-radius: 12px;
        padding: 16px 20px;
        backdrop-filter: blur(10px);
    ">
        <div style="color: #6b7280; font-size: 0.75rem; font-weight: 500;
                    text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">
            {label}
        </div>
        <div style="color: #f1f5f9; font-size: 1.25rem; font-weight: 700;">
            {value}
        </div>
        <div style="color: {delta_color}; font-size: 0.8rem; font-weight: 500; margin-top: 4px;">
            {delta}
        </div>
    </div>
    """


def _section_header(title: str, subtitle: str = "") -> str:
    sub = f'<span style="color:#6b7280; font-weight:400; font-size:0.9rem;"> &mdash; {subtitle}</span>' if subtitle else ""
    return f'<h3 style="color:#e2e8f0; margin:0 0 16px 0; font-size:1.15rem; font-weight:600;">{title}{sub}</h3>'


# Plotly dark theme
_PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#9ca3af", size=12),
    title_font=dict(color="#e2e8f0", size=15, family="Inter"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#9ca3af")),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)"),
    margin=dict(l=20, r=20, t=50, b=20),
    hoverlabel=dict(bgcolor="#1e293b", font_color="#f1f5f9", bordercolor="#334155"),
)


_CAT_COLORS: dict[str, str] = {
    "salary":             "#22c55e",
    "interest":           "#86efac",
    "incoming_refund":    "#6ee7b7",
    "incoming_cash":      "#34d399",
    "incoming_upi":       "#0ea5e9",
    "incoming_transfer":  "#38bdf8",
    "incoming_other":     "#7dd3fc",
    "atm":                "#f97316",
    "emi":                "#fb923c",
    "bill_payment":       "#fbbf24",
    "charges":            "#f59e0b",
    "fuel":               "#fb7185",
    "investment":         "#a78bfa",
    "outgoing_upi":       "#ef4444",
    "outgoing_transfer":  "#dc2626",
    "outgoing_other":     "#fca5a5",
    "neutral":            "#6b7280",
}

_CAT_LABELS: dict[str, str] = {
    "salary": "Salary", "interest": "Interest", "incoming_refund": "Refunds",
    "incoming_cash": "Cash Deposit", "incoming_upi": "UPI Received",
    "incoming_transfer": "Transfer In", "incoming_other": "Other Inflow",
    "atm": "ATM", "emi": "EMI / Loan", "bill_payment": "Bill Payment",
    "charges": "Charges", "fuel": "Fuel", "investment": "Investment",
    "outgoing_upi": "UPI Paid", "outgoing_transfer": "Transfer Out",
    "outgoing_other": "Other Outflow", "neutral": "Neutral",
}


# ---------------------------------------------------------------------------
# Header + upload
# ---------------------------------------------------------------------------

st.markdown("""
<div style="text-align:center; padding: 20px 0 10px 0;">
    <h1 style="color:#f1f5f9; font-size:2rem; font-weight:700; margin:0;">
        Bank Statement Analyzer
    </h1>
    <p style="color:#6b7280; font-size:0.95rem; margin-top:6px;">
        Upload any Indian bank PDF statement for a complete cash-flow dashboard
    </p>
</div>
""", unsafe_allow_html=True)

with st.form("upload_form"):
    col_up, col_path = st.columns([1, 2])
    with col_up:
        uploaded = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
    with col_path:
        manual_path = st.text_input(
            "Or enter absolute file path",
            placeholder=r"C:\Users\you\Downloads\AccountStatement.pdf",
        )
    submitted = st.form_submit_button("Analyze Statement", type="primary", width="stretch")


# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------

if submitted:
    try:
        if uploaded is not None:
            buf = bytes(uploaded.getbuffer())
            fhash = _file_hash(buf)
            if st.session_state.get("file_hash") != fhash:
                suffix = Path(uploaded.name).suffix or ".pdf"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(buf)
                    pdf_path = Path(tmp.name)
                with st.spinner("Parsing PDF..."):
                    parsed = parse_statement_pdf(str(pdf_path))
                    enriched = enrich_transactions(parsed.transactions)
                st.session_state.update(file_hash=fhash, parsed=parsed, df_full=enriched)

        elif manual_path.strip():
            pdf_path = Path(manual_path.strip())
            if not pdf_path.exists():
                st.error(f"File not found: {pdf_path}")
                st.stop()
            buf = pdf_path.read_bytes()
            fhash = _file_hash(buf)
            if st.session_state.get("file_hash") != fhash:
                with st.spinner("Parsing PDF..."):
                    parsed = parse_statement_pdf(str(pdf_path))
                    enriched = enrich_transactions(parsed.transactions)
                st.session_state.update(file_hash=fhash, parsed=parsed, df_full=enriched)
        else:
            st.warning("Please upload a PDF or enter a file path.")
            st.stop()
    except Exception as exc:
        st.exception(exc)
        st.stop()


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

if "df_full" not in st.session_state:
    st.markdown("""
    <div style="text-align:center; padding:60px 20px; color:#6b7280;">
        <div style="font-size:3rem; margin-bottom:16px;">\U0001f4c4</div>
        <p style="font-size:1.1rem;">Upload a statement and click <strong style="color:#60a5fa;">Analyze Statement</strong></p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

parsed = st.session_state.parsed
df_full: pd.DataFrame = st.session_state.df_full

if df_full.empty:
    st.error("No transactions could be extracted. Ensure it is a native (non-scanned) bank statement.")
    st.stop()

# Success banner
st.markdown(f"""
<div style="
    background: linear-gradient(135deg, rgba(34,197,94,0.1), rgba(34,197,94,0.05));
    border: 1px solid rgba(34,197,94,0.2);
    border-radius: 12px;
    padding: 14px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
">
    <span style="font-size:1.3rem;">&#x2705;</span>
    <div>
        <span style="color:#86efac; font-weight:600;">{len(df_full)} transactions</span>
        <span style="color:#9ca3af;"> parsed from </span>
        <span style="color:#86efac; font-weight:600;">{parsed.page_count} pages</span>
        <span style="color:#9ca3af;"> &mdash; </span>
        <span style="color:#60a5fa; font-weight:600;">{parsed.bank_name}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<h2 style="color:#e2e8f0; font-size:1.2rem; margin-bottom:16px;">Filters</h2>',
                unsafe_allow_html=True)

    min_d = df_full["date"].min().date()
    max_d = df_full["date"].max().date()
    date_range = st.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)

    direction_choice = st.radio("Direction", ["All", "Inflow only", "Outflow only"], horizontal=True)

    all_cats = sorted(df_full["category"].unique().tolist())
    selected_cats = st.multiselect("Categories", all_cats, default=all_cats)

    max_amt = float(df_full["amount"].max()) or 1.0
    amount_range = st.slider("Amount", 0.0, max_amt, (0.0, max_amt), step=100.0, format="\u20b9%.0f")

    search = st.text_input("Search particulars / merchant", placeholder="e.g. Zomato, NEFT...")

    st.markdown("---")
    st.markdown(f"""
    <div style="padding:8px 0; color:#6b7280; font-size:0.8rem;">
        <div><strong style="color:#9ca3af;">Bank:</strong> {parsed.bank_name}</div>
        <div><strong style="color:#9ca3af;">Pages:</strong> {parsed.page_count}</div>
        <div><strong style="color:#9ca3af;">Period:</strong> {min_d} to {max_d}</div>
    </div>
    """, unsafe_allow_html=True)


# Apply filters
df = df_full.copy()
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    df = df[df["date"].dt.date.between(date_range[0], date_range[1])]
if direction_choice == "Inflow only":
    df = df[df["direction"] == "inflow"]
elif direction_choice == "Outflow only":
    df = df[df["direction"] == "outflow"]
if selected_cats:
    df = df[df["category"].isin(selected_cats)]
df = df[df["amount"].between(amount_range[0], amount_range[1])]
if search.strip():
    mask = (
        df["particulars"].str.contains(search, case=False, na=False)
        | df["merchant"].str.contains(search, case=False, na=False)
    )
    df = df[mask]

summary = summarize(df)


# ---------------------------------------------------------------------------
# KPI Cards
# ---------------------------------------------------------------------------

# Row 1: Main metrics
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(_kpi_card("Total Inflow", money(summary["total_inflow"]), "#22c55e"), unsafe_allow_html=True)
with c2:
    st.markdown(_kpi_card("Total Outflow", money(summary["total_outflow"]), "#ef4444"), unsafe_allow_html=True)
with c3:
    net = summary["net"]
    delta_sign = "+" if net >= 0 else ""
    delta_col = "#22c55e" if net >= 0 else "#ef4444"
    st.markdown(_kpi_card_with_delta("Net Change", money(net),
                f"{delta_sign}{_indian_format(net)}", "#3b82f6", delta_col), unsafe_allow_html=True)
with c4:
    st.markdown(_kpi_card("Transactions", str(len(df)), "#8b5cf6"), unsafe_allow_html=True)

st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

# Row 2: Inflow breakdown
c5, c6, c7, c8, c9 = st.columns(5)
with c5:
    st.markdown(_kpi_card("Salary / Pay", money(summary.get("salary", 0)), "#22c55e"), unsafe_allow_html=True)
with c6:
    st.markdown(_kpi_card("Transfer In", money(summary.get("incoming_transfer", 0)), "#38bdf8"), unsafe_allow_html=True)
with c7:
    st.markdown(_kpi_card("UPI Received", money(summary.get("incoming_upi", 0)), "#0ea5e9"), unsafe_allow_html=True)
with c8:
    st.markdown(_kpi_card("Refunds", money(summary.get("incoming_refund", 0)), "#6ee7b7"), unsafe_allow_html=True)
with c9:
    st.markdown(_kpi_card("Interest", money(summary.get("interest", 0)), "#86efac"), unsafe_allow_html=True)

st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

# Row 3: Outflow breakdown
ca, cb, cc, cd, ce = st.columns(5)
with ca:
    st.markdown(_kpi_card("Transfer Out", money(summary.get("outgoing_transfer", 0)), "#dc2626"), unsafe_allow_html=True)
with cb:
    st.markdown(_kpi_card("UPI Paid", money(summary.get("outgoing_upi", 0)), "#ef4444"), unsafe_allow_html=True)
with cc:
    st.markdown(_kpi_card("ATM Withdrawals", money(summary.get("atm", 0)), "#f97316"), unsafe_allow_html=True)
with cd:
    st.markdown(_kpi_card("EMI / Loan", money(summary.get("emi", 0)), "#fb923c"), unsafe_allow_html=True)
with ce:
    st.markdown(_kpi_card("Charges", money(summary.get("charges", 0)), "#f59e0b"), unsafe_allow_html=True)

st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_inflow, tab_outflow, tab_benef, tab_monthly, tab_txns = st.tabs([
    "Overview",
    "Where Money Came From",
    "Where Money Went",
    "Beneficiary Transfers",
    "Monthly Trends",
    "Transactions",
])


# \u2500\u2500 Overview \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

with tab_overview:
    col_a, col_b = st.columns(2)

    with col_a:
        fig = go.Figure(data=[go.Pie(
            labels=["Inflow", "Outflow"],
            values=[summary["total_inflow"], summary["total_outflow"]],
            hole=0.6,
            marker=dict(colors=["#22c55e", "#ef4444"]),
            textinfo="percent+label",
            textfont=dict(color="#e2e8f0", size=13),
            hovertemplate="<b>%{label}</b><br>\u20b9%{value:,.2f}<br>%{percent}<extra></extra>",
        )])
        fig.update_layout(**_PLOTLY_LAYOUT, title="Inflow vs Outflow",
                          annotations=[dict(text=f"<b>Net</b><br>{money(summary['net'])}",
                                            x=0.5, y=0.5, font_size=14, font_color="#e2e8f0",
                                            showarrow=False)])
        st.plotly_chart(fig, width="stretch")

    with col_b:
        cat_df = (
            df[df["direction"].isin(["inflow", "outflow"])]
            .groupby("category", as_index=False)["amount"].sum()
        )
        cat_df["label"] = cat_df["category"].map(_CAT_LABELS).fillna(cat_df["category"])
        colors = [_CAT_COLORS.get(c, "#6b7280") for c in cat_df["category"]]
        fig = go.Figure(data=[go.Pie(
            labels=cat_df["label"], values=cat_df["amount"],
            hole=0.6,
            marker=dict(colors=colors),
            textinfo="percent+label",
            textfont=dict(color="#e2e8f0", size=11),
            hovertemplate="<b>%{label}</b><br>\u20b9%{value:,.2f}<br>%{percent}<extra></extra>",
        )])
        fig.update_layout(**_PLOTLY_LAYOUT, title="By Category")
        st.plotly_chart(fig, width="stretch")

    if not df.empty:
        daily = (
            df.assign(day=df["date"].dt.date)
            .groupby("day", as_index=False)
            .agg(credit=("credit", "sum"), debit=("debit", "sum"))
        )
        daily["net"] = daily["credit"] - daily["debit"]
        colors = ["#22c55e" if v >= 0 else "#ef4444" for v in daily["net"]]
        fig = go.Figure(data=[go.Bar(
            x=daily["day"], y=daily["net"],
            marker_color=colors,
            hovertemplate="<b>%{x}</b><br>Net: \u20b9%{y:,.2f}<extra></extra>",
        )])
        fig.update_layout(**_PLOTLY_LAYOUT, title="Daily Net Cash Flow",
                          xaxis_title="", yaxis_title="Net (\u20b9)")
        fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.1)")
        st.plotly_chart(fig, width="stretch")


# \u2500\u2500 Where Money Came From \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

with tab_inflow:
    inflow_df = df[df["direction"] == "inflow"]

    if inflow_df.empty:
        st.info("No inflow transactions match the current filters.")
    else:
        col_l, col_r = st.columns([3, 2])

        with col_l:
            inc_cats = (
                inflow_df.groupby("category", as_index=False)["amount"]
                .sum().sort_values("amount")
            )
            inc_cats["label"] = inc_cats["category"].map(_CAT_LABELS).fillna(inc_cats["category"])
            colors = [_CAT_COLORS.get(c, "#6b7280") for c in inc_cats["category"]]
            fig = go.Figure(data=[go.Bar(
                y=inc_cats["label"], x=inc_cats["amount"],
                orientation="h",
                marker_color=colors,
                hovertemplate="<b>%{y}</b><br>\u20b9%{x:,.2f}<extra></extra>",
            )])
            fig.update_layout(**_PLOTLY_LAYOUT, title="Inflow by Category",
                              xaxis_title="Amount (\u20b9)", yaxis_title="")
            st.plotly_chart(fig, width="stretch")

        with col_r:
            st.markdown(_section_header("Top Senders / Sources"), unsafe_allow_html=True)
            senders = top_merchants(inflow_df, "inflow", 10)
            senders = senders.rename(columns={
                "merchant": "Sender", "total_amount": "Amount (\u20b9)", "transactions": "Txns"
            })
            senders["Amount (\u20b9)"] = senders["Amount (\u20b9)"].map(lambda v: money(v))
            st.dataframe(senders, width="stretch", hide_index=True)

        inflow_time = (
            inflow_df.assign(day=inflow_df["date"].dt.date)
            .groupby(["day", "category"], as_index=False)["amount"].sum()
        )
        inflow_time["label"] = inflow_time["category"].map(_CAT_LABELS).fillna(inflow_time["category"])
        fig = px.area(
            inflow_time, x="day", y="amount", color="label",
            title="Inflow Over Time",
            labels={"day": "", "amount": "Amount (\u20b9)", "label": "Category"},
            color_discrete_map={_CAT_LABELS.get(k, k): v for k, v in _CAT_COLORS.items()},
        )
        fig.update_layout(**_PLOTLY_LAYOUT)
        st.plotly_chart(fig, width="stretch")


# \u2500\u2500 Where Money Went \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

with tab_outflow:
    outflow_df = df[df["direction"] == "outflow"]

    if outflow_df.empty:
        st.info("No outflow transactions match the current filters.")
    else:
        # Build category breakdown
        out_cats = (
            outflow_df.groupby("category", as_index=False)["amount"]
            .sum().sort_values("amount")
        )
        out_cats["label"] = out_cats["category"].map(_CAT_LABELS).fillna(out_cats["category"])

        # Category selector pills
        cat_options = out_cats.sort_values("amount", ascending=False)["category"].tolist()
        cat_labels_map = dict(zip(out_cats["category"], out_cats["label"]))

        selected_outflow_cat = st.pills(
            "Click a category to view its transactions",
            options=cat_options,
            format_func=lambda c: cat_labels_map.get(c, c),
            default=None,
            key="outflow_cat_pills",
        )

        col_l, col_r = st.columns([3, 2])

        with col_l:
            # Highlight selected bar, dim others
            if selected_outflow_cat:
                bar_colors = []
                for c in out_cats["category"]:
                    hex_color = _CAT_COLORS.get(c, "#6b7280")
                    if c == selected_outflow_cat:
                        bar_colors.append(hex_color)
                    else:
                        r = int(hex_color[1:3], 16)
                        g = int(hex_color[3:5], 16)
                        b = int(hex_color[5:7], 16)
                        bar_colors.append(f"rgba({r},{g},{b},0.25)")
            else:
                bar_colors = [_CAT_COLORS.get(c, "#6b7280") for c in out_cats["category"]]

            fig = go.Figure(data=[go.Bar(
                y=out_cats["label"], x=out_cats["amount"],
                orientation="h",
                marker_color=bar_colors,
                hovertemplate="<b>%{y}</b><br>\u20b9%{x:,.2f}<extra></extra>",
            )])
            fig.update_layout(**_PLOTLY_LAYOUT, title="Outflow by Category",
                              xaxis_title="Amount (\u20b9)", yaxis_title="")
            st.plotly_chart(fig, width="stretch")

        with col_r:
            if selected_outflow_cat:
                # Show transactions for the selected category
                cat_label = _CAT_LABELS.get(selected_outflow_cat, selected_outflow_cat)
                cat_color = _CAT_COLORS.get(selected_outflow_cat, "#6b7280")
                cat_txns = outflow_df[outflow_df["category"] == selected_outflow_cat].copy()
                cat_total = cat_txns["amount"].sum()

                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                    <span style="
                        background: {cat_color}18; color: {cat_color};
                        padding: 4px 12px; border-radius: 6px;
                        font-size: 0.9rem; font-weight: 600;
                    ">{cat_label}</span>
                    <span style="color: #6b7280; font-size: 0.8rem;">
                        {len(cat_txns)} transactions &bull; Total: {money(cat_total)}
                    </span>
                </div>
                """, unsafe_allow_html=True)

                cat_view = cat_txns[["date", "particulars", "merchant", "debit", "balance"]].copy()
                cat_view["date"] = cat_view["date"].dt.strftime("%d-%m-%Y")
                cat_view["debit"] = cat_view["debit"].map(lambda v: money(v) if v else "")
                cat_view["balance"] = cat_view["balance"].map(lambda v: money(v) if v else "")
                cat_view = cat_view.rename(columns={
                    "date": "Date", "particulars": "Particulars",
                    "merchant": "Merchant", "debit": "Amount (\u20b9)", "balance": "Balance (\u20b9)",
                })
                st.dataframe(cat_view, width="stretch", hide_index=True, height=400)
            else:
                st.markdown(_section_header("Top Merchants / Payees"), unsafe_allow_html=True)
                payees = top_merchants(outflow_df, "outflow", 10)
                payees = payees.rename(columns={
                    "merchant": "Payee", "total_amount": "Amount (\u20b9)", "transactions": "Txns"
                })
                payees["Amount (\u20b9)"] = payees["Amount (\u20b9)"].map(lambda v: money(v))
                st.dataframe(payees, width="stretch", hide_index=True)

        # Sunburst
        sun_df = outflow_df.copy()
        sun_df["cat_label"] = sun_df["category"].map(_CAT_LABELS).fillna(sun_df["category"])
        fig = px.sunburst(
            sun_df, path=["direction", "cat_label"], values="amount",
            title="Spending Hierarchy",
            color="cat_label",
            color_discrete_map={_CAT_LABELS.get(k, k): v for k, v in _CAT_COLORS.items()},
        )
        fig.update_traces(textinfo="label+percent parent")
        fig.update_layout(**_PLOTLY_LAYOUT)
        st.plotly_chart(fig, width="stretch")


# \u2500\u2500 Beneficiary Transfers \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

with tab_benef:
    benef_dir = st.radio(
        "Transfer direction", ["Outgoing", "Incoming"], horizontal=True, key="benef_dir"
    )
    benef_direction = "outflow" if benef_dir == "Outgoing" else "inflow"
    benef_df = beneficiary_summary(df, direction=benef_direction)

    if benef_df.empty:
        st.info(f"No {benef_dir.lower()} transfers with identified beneficiaries.")
    else:
        col_chart, col_table = st.columns([3, 2])

        with col_chart:
            top_n = min(15, len(benef_df))
            chart_df = benef_df.head(top_n).sort_values("total_amount")
            bar_color = "#dc2626" if benef_direction == "outflow" else "#22c55e"
            fig = go.Figure(data=[go.Bar(
                y=chart_df["beneficiary"], x=chart_df["total_amount"],
                orientation="h",
                marker_color=bar_color,
                hovertemplate="<b>%{y}</b><br>\u20b9%{x:,.2f}<extra></extra>",
            )])
            fig.update_layout(**_PLOTLY_LAYOUT,
                              title=f"Top {top_n} {benef_dir} Beneficiaries",
                              xaxis_title="Amount (\u20b9)", yaxis_title="")
            st.plotly_chart(fig, width="stretch")

        with col_table:
            st.markdown(_section_header(f"{benef_dir} Transfer Breakdown"), unsafe_allow_html=True)
            display_benef = benef_df.copy()
            display_benef = display_benef.rename(columns={
                "beneficiary": "Beneficiary", "total_amount": "Amount (\u20b9)", "transactions": "Txns"
            })
            display_benef["Amount (\u20b9)"] = display_benef["Amount (\u20b9)"].map(lambda v: money(v))
            st.dataframe(display_benef, width="stretch", hide_index=True)

        st.markdown(_section_header("Transaction Details by Beneficiary"), unsafe_allow_html=True)
        transfer_cat = "outgoing_transfer" if benef_direction == "outflow" else "incoming_transfer"
        transfer_txns = df[
            (df["category"] == transfer_cat) & (df["beneficiary"] != "")
        ]
        if not transfer_txns.empty:
            selected_benef = st.selectbox(
                "Select beneficiary",
                benef_df["beneficiary"].tolist(),
                key="benef_select",
            )
            detail = transfer_txns[transfer_txns["beneficiary"] == selected_benef][
                ["date", "particulars", "amount", "channel"]
            ].copy()
            detail["date"] = detail["date"].dt.strftime("%d-%m-%Y")
            detail["amount"] = detail["amount"].map(lambda v: money(v))
            detail = detail.rename(columns={
                "date": "Date", "particulars": "Particulars",
                "amount": "Amount (\u20b9)", "channel": "Channel",
            })
            st.dataframe(detail, width="stretch", hide_index=True)


# \u2500\u2500 Monthly Trends \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

with tab_monthly:
    monthly = monthly_summary(df)

    if monthly.empty:
        st.info("Not enough data for monthly trends.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly["month"], y=monthly["inflow"], name="Inflow",
            marker_color="#22c55e",
            hovertemplate="Inflow: \u20b9%{y:,.2f}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            x=monthly["month"], y=monthly["outflow"], name="Outflow",
            marker_color="#ef4444",
            hovertemplate="Outflow: \u20b9%{y:,.2f}<extra></extra>",
        ))
        fig.update_layout(**_PLOTLY_LAYOUT, title="Monthly Inflow vs Outflow",
                          barmode="group", xaxis_title="", yaxis_title="Amount (\u20b9)")
        st.plotly_chart(fig, width="stretch")

        net_colors = ["#22c55e" if v >= 0 else "#ef4444" for v in monthly["net"]]
        fig = go.Figure(data=[go.Scatter(
            x=monthly["month"], y=monthly["net"],
            mode="lines+markers",
            line=dict(color="#38bdf8", width=2),
            marker=dict(color=net_colors, size=10),
            hovertemplate="Net: \u20b9%{y:,.2f}<extra></extra>",
        )])
        fig.update_layout(**_PLOTLY_LAYOUT, title="Monthly Net Flow",
                          xaxis_title="", yaxis_title="Net (\u20b9)")
        fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.15)")
        st.plotly_chart(fig, width="stretch")

        # Heatmap
        df_h = df.copy()
        df_h["month"] = df_h["date"].dt.to_period("M").astype(str)
        df_h["cat_label"] = df_h["category"].map(_CAT_LABELS).fillna(df_h["category"])
        pivot = df_h.pivot_table(
            index="cat_label", columns="month",
            values="amount", aggfunc="sum", fill_value=0,
        )
        if not pivot.empty:
            fig = px.imshow(
                pivot, aspect="auto",
                title="Category Spend Heatmap",
                labels={"x": "Month", "y": "Category", "color": "Amount (\u20b9)"},
                color_continuous_scale=["#0f172a", "#1e40af", "#3b82f6", "#22c55e", "#fbbf24", "#ef4444"],
            )
            fig.update_layout(**_PLOTLY_LAYOUT)
            st.plotly_chart(fig, width="stretch")


# \u2500\u2500 Transactions \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

with tab_txns:
    st.markdown(f"""
    <div style="color:#6b7280; font-size:0.85rem; margin-bottom:12px;">
        Showing <strong style="color:#e2e8f0;">{len(df)}</strong> of
        <strong style="color:#e2e8f0;">{len(df_full)}</strong> transactions
    </div>
    """, unsafe_allow_html=True)

    view = df[[
        "date", "particulars", "merchant", "beneficiary", "reference_no",
        "debit", "credit", "balance", "channel", "category",
    ]].copy()
    view["date"] = view["date"].dt.strftime("%d-%m-%Y")
    for col in ("debit", "credit", "balance"):
        view[col] = view[col].map(lambda v: money(v) if v else "")

    st.dataframe(
        view,
        width="stretch",
        hide_index=True,
        column_config={
            "date":         st.column_config.TextColumn("Date"),
            "particulars":  st.column_config.TextColumn("Particulars", width="large"),
            "merchant":     st.column_config.TextColumn("Merchant / Sender"),
            "beneficiary":  st.column_config.TextColumn("Beneficiary"),
            "reference_no": st.column_config.TextColumn("Ref No"),
            "debit":        st.column_config.TextColumn("Debit (\u20b9)"),
            "credit":       st.column_config.TextColumn("Credit (\u20b9)"),
            "balance":      st.column_config.TextColumn("Balance (\u20b9)"),
            "channel":      st.column_config.TextColumn("Channel"),
            "category":     st.column_config.TextColumn("Category"),
        },
    )

    csv = view.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered transactions as CSV",
        csv, "transactions.csv", "text/csv",
        width="stretch",
    )
