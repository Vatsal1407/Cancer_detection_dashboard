import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BreaKHis Cancer Classification Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap');

/* ── Global Reset ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main, .stApp { background: #02060f !important; }
section[data-testid="stSidebar"] > div { background: #030912; }

/* ── Hero Block ── */
.hero {
    position: relative;
    background: radial-gradient(ellipse at 30% 50%, #0a1628 0%, #02060f 60%),
                radial-gradient(ellipse at 80% 20%, #0d1a2e 0%, transparent 50%);
    border: 1px solid rgba(34, 211, 238, 0.18);
    border-radius: 20px;
    padding: 44px 52px;
    margin-bottom: 30px;
    overflow: hidden;
}
.hero-scan {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg,
        transparent 0%, transparent 20%,
        #22d3ee 40%, #a78bfa 55%, #22d3ee 70%,
        transparent 80%, transparent 100%);
    background-size: 200% 100%;
    animation: scanline 4s linear infinite;
}
@keyframes scanline {
    0%   { background-position: 200% center; }
    100% { background-position: -200% center; }
}
.hero-grid {
    position: absolute; inset: 0; pointer-events: none;
    background-image:
        linear-gradient(rgba(34,211,238,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(34,211,238,0.025) 1px, transparent 1px);
    background-size: 48px 48px;
}
.hero-dot {
    position: absolute; right: 52px; top: 50%; transform: translateY(-50%);
    width: 120px; height: 120px;
    border: 1px solid rgba(34,211,238,0.15);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 3rem;
    box-shadow: 0 0 40px rgba(34,211,238,0.08), inset 0 0 30px rgba(34,211,238,0.04);
}
.hero-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.7rem; font-weight: 700;
    color: #e2f0ff; margin: 0;
    text-transform: uppercase; letter-spacing: 1.5px;
    text-shadow: 0 0 40px rgba(34,211,238,0.2);
}
.hero-title em { font-style: normal; color: #22d3ee; }
.hero-sub {
    font-size: 0.92rem; color: #3d5470;
    margin-top: 10px; letter-spacing: 0.5px;
}
.hero-tags { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 20px; }
.htag {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem; font-weight: 700;
    letter-spacing: 1px; padding: 4px 14px;
    border-radius: 3px; text-transform: uppercase;
}
.htag-c { background: rgba(34,211,238,0.08); color: #22d3ee;   border: 1px solid rgba(34,211,238,0.25); }
.htag-p { background: rgba(167,139,250,0.08); color: #a78bfa;  border: 1px solid rgba(167,139,250,0.25); }
.htag-g { background: rgba(52,211,153,0.08);  color: #34d399;  border: 1px solid rgba(52,211,153,0.25); }
.htag-o { background: rgba(251,191,36,0.08);  color: #fbbf24;  border: 1px solid rgba(251,191,36,0.25); }

/* ── KPI Cards ── */
.kpi-grid { display: grid; gap: 16px; grid-template-columns: repeat(4, 1fr); margin-bottom: 4px; }
.kpi {
    background: #030c1a;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 16px;
    padding: 26px 22px 20px;
    text-align: center;
    position: relative; overflow: hidden;
    transition: border-color .25s, box-shadow .25s;
}
.kpi:hover { border-color: rgba(34,211,238,0.2); box-shadow: 0 8px 32px rgba(34,211,238,0.06); }
.kpi-bar {
    position: absolute; bottom: 0; left: 20%; right: 20%; height: 2px;
    border-radius: 2px 2px 0 0;
}
.kpi-icon { font-size: 1.5rem; margin-bottom: 10px; line-height: 1; }
.kpi-lbl {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem; letter-spacing: 2px;
    text-transform: uppercase; color: #2d4562; margin-bottom: 10px;
}
.kpi-val {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.5rem; font-weight: 700; line-height: 1;
}
.kpi-note { font-family: 'Space Mono', monospace; font-size: 0.65rem; color: #2d4562; margin-top: 8px; }

.kpi.amber .kpi-val { color: #fbbf24; }
.kpi.amber .kpi-bar  { background: linear-gradient(90deg, transparent, #fbbf24, transparent); }
.kpi.green .kpi-val  { color: #34d399; }
.kpi.green .kpi-bar  { background: linear-gradient(90deg, transparent, #34d399, transparent); }
.kpi.cyan  .kpi-val  { color: #22d3ee; }
.kpi.cyan  .kpi-bar  { background: linear-gradient(90deg, transparent, #22d3ee, transparent); }
.kpi.purple .kpi-val { color: #a78bfa; }
.kpi.purple .kpi-bar { background: linear-gradient(90deg, transparent, #a78bfa, transparent); }

/* ── Section Header ── */
.sh {
    display: flex; align-items: center; gap: 12px;
    margin: 34px 0 14px 0;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(34,211,238,0.07);
}
.sh-bar { width: 3px; height: 16px; background: #22d3ee; border-radius: 2px; flex-shrink: 0; }
.sh-text {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.72rem; font-weight: 600;
    color: #22d3ee; text-transform: uppercase; letter-spacing: 3px;
}

/* ── Insight Cards ── */
.ic {
    background: #030c1a;
    border: 1px solid rgba(255,255,255,0.04);
    border-left: 3px solid;
    border-radius: 0 12px 12px 0;
    padding: 14px 20px;
    margin-bottom: 10px;
    font-size: 0.87rem; color: #5d7a99; line-height: 1.75;
}
.ic b { color: #ccd9e8; }
.ic.success { border-left-color: #34d399; }
.ic.info    { border-left-color: #22d3ee; }
.ic.warning { border-left-color: #fb923c; }
.ic.purple  { border-left-color: #a78bfa; }

/* ── Rank Cards ── */
.rc {
    display: flex; align-items: center; gap: 16px;
    background: #030c1a;
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 14px 18px; margin-bottom: 10px;
    border-left: 3px solid;
}
.rc-medal { font-size: 1.4rem; }
.rc-model-name {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem; font-weight: 600; color: #8ca5bf;
}
.rc-score {
    font-family: 'Space Mono', monospace;
    font-size: 1.15rem; font-weight: 700; margin-top: 3px;
}

/* ── Model Pills ── */
.pill {
    display: inline-block; padding: 2px 10px; border-radius: 3px;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.5px;
}
.pill-vit      { background: rgba(34,211,238,0.1);  color: #22d3ee;  border: 1px solid rgba(34,211,238,0.2); }
.pill-resnet   { background: rgba(251,146,60,0.1);  color: #fb923c;  border: 1px solid rgba(251,146,60,0.2); }
.pill-densenet { background: rgba(52,211,153,0.1);  color: #34d399;  border: 1px solid rgba(52,211,153,0.2); }
.pill-effnet   { background: rgba(167,139,250,0.1); color: #a78bfa;  border: 1px solid rgba(167,139,250,0.2); }

/* ── Config Cards ── */
.cc {
    background: #030c1a;
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 10px;
    padding: 14px 18px; margin-bottom: 10px;
}
.cc-top { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
.cc-icon { font-size: 1rem; }
.cc-key {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem; color: #2d4562;
    text-transform: uppercase; letter-spacing: 1.5px;
}
.cc-val {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.05rem; font-weight: 600; color: #ccd9e8;
    margin-top: 6px;
}

/* ── Sidebar ── */
div[data-testid="stSidebar"] {
    background: #030912 !important;
    border-right: 1px solid rgba(34,211,238,0.06);
}
div[data-testid="stSidebar"] h3 {
    font-family: 'Rajdhani', sans-serif !important;
    color: #22d3ee !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    font-size: 0.8rem !important;
}

/* ── Footer ── */
.footer {
    text-align: center;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem; color: #1a2d42;
    letter-spacing: 0.8px; margin-top: 52px;
    padding: 18px 0;
    border-top: 1px solid rgba(34,211,238,0.05);
}

/* Streamlit overrides */
div[data-testid="stDataFrame"] { border: 1px solid rgba(34,211,238,0.1) !important; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─── Data ────────────────────────────────────────────────────────────────────────
DATA = {
    "composite_scores": {
        "ViT": 0.9874607416733706,
        "ResNet-50": 0.9168531025173644,
        "DenseNet-121": 0.9790437458837857,
        "EfficientNet": 0.9521605139268855
    },
    "models": {
        "ViT": {
            "40X":  {"accuracy": 0.9767, "auc_roc": 0.9992, "f1_score": 0.9827, "recall": 1.0000, "precision": 0.9660},
            "100X": {"accuracy": 0.9936, "auc_roc": 0.9998, "f1_score": 0.9955, "recall": 0.9955, "precision": 0.9955},
            "200X": {"accuracy": 0.9868, "auc_roc": 0.9966, "f1_score": 0.9902, "recall": 0.9951, "precision": 0.9854},
            "400X": {"accuracy": 0.9707, "auc_roc": 0.9963, "f1_score": 0.9771, "recall": 0.9828, "precision": 0.9716},
        },
        "ResNet-50": {
            "40X":  {"accuracy": 0.8167, "auc_roc": 0.9688, "f1_score": 0.8433, "recall": 0.7437, "precision": 0.9737},
            "100X": {"accuracy": 0.8786, "auc_roc": 0.9791, "f1_score": 0.9087, "recall": 0.8514, "precision": 0.9742},
            "200X": {"accuracy": 0.8907, "auc_roc": 0.9727, "f1_score": 0.9138, "recall": 0.8578, "precision": 0.9777},
            "400X": {"accuracy": 0.8242, "auc_roc": 0.9544, "f1_score": 0.8442, "recall": 0.7471, "precision": 0.9701},
        },
        "DenseNet-121": {
            "40X":  {"accuracy": 0.9633, "auc_roc": 0.9922, "f1_score": 0.9719, "recall": 0.9548, "precision": 0.9896},
            "100X": {"accuracy": 0.9840, "auc_roc": 0.9990, "f1_score": 0.9887, "recall": 0.9820, "precision": 0.9954},
            "200X": {"accuracy": 0.9603, "auc_roc": 0.9945, "f1_score": 0.9700, "recall": 0.9510, "precision": 0.9898},
            "400X": {"accuracy": 0.9451, "auc_roc": 0.9887, "f1_score": 0.9563, "recall": 0.9425, "precision": 0.9704},
        },
        "EfficientNet": {
            "40X":  {"accuracy": 0.9400, "auc_roc": 0.9920, "f1_score": 0.9550, "recall": 0.9598, "precision": 0.9502},
            "100X": {"accuracy": 0.9393, "auc_roc": 0.9819, "f1_score": 0.9565, "recall": 0.9414, "precision": 0.9721},
            "200X": {"accuracy": 0.9073, "auc_roc": 0.9825, "f1_score": 0.9296, "recall": 0.9069, "precision": 0.9536},
            "400X": {"accuracy": 0.9158, "auc_roc": 0.9681, "f1_score": 0.9326, "recall": 0.9138, "precision": 0.9521},
        },
    }
}

MODELS  = list(DATA["models"].keys())
MAGS    = ["40X", "100X", "200X", "400X"]
METRICS = ["accuracy", "auc_roc", "f1_score", "recall", "precision"]
METRIC_LABELS = {
    "accuracy": "Accuracy", "auc_roc": "AUC-ROC",
    "f1_score": "F1-Score", "recall": "Recall", "precision": "Precision"
}
MODEL_COLORS = {
    "ViT": "#22d3ee",
    "ResNet-50": "#fb923c",
    "DenseNet-121": "#34d399",
    "EfficientNet": "#a78bfa",
}
PILL_CLASS = {
    "ViT": "pill-vit", "ResNet-50": "pill-resnet",
    "DenseNet-121": "pill-densenet", "EfficientNet": "pill-effnet",
}

# ─── Color Helper ────────────────────────────────────────────────────────────────
# FIX: Plotly does NOT accept 8-char hex (#rrggbbAA). Use rgba() strings instead.
def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a 6-char hex color to an rgba() string Plotly can accept."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ─── Plotly Helpers ──────────────────────────────────────────────────────────────
_MONO = "'Space Mono', monospace"
_AX   = dict(
    gridcolor="rgba(34,211,238,0.05)",
    linecolor="rgba(34,211,238,0.1)",
    tickfont=dict(color="#2d4562", family=_MONO, size=9),
    zerolinecolor="rgba(34,211,238,0.08)",
)

def make_layout(**kw) -> dict:
    """Return a complete Plotly layout dict merged with project base styles."""
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(3,12,26,0.97)",
        font=dict(family=_MONO, color="#3d5470", size=11),
        margin=dict(l=44, r=24, t=48, b=40),
    )
    base.update(kw)
    return base

def xax(**kw) -> dict:
    return {**_AX, **kw}

def yax(**kw) -> dict:
    return {**_AX, **kw}

def pct(v: float) -> str:
    return f"{v * 100:.2f}%"

# Reusable legend presets
_LEG_H = dict(
    bgcolor="rgba(0,0,0,0)",
    bordercolor="rgba(34,211,238,0.12)", borderwidth=1,
    orientation="h", yanchor="bottom", y=1.02,
    font=dict(size=10, family=_MONO),
)
_LEG_V = dict(
    bgcolor="rgba(0,0,0,0)",
    bordercolor="rgba(34,211,238,0.12)", borderwidth=1,
    font=dict(size=10, family=_MONO),
)

# ─── HTML helpers ────────────────────────────────────────────────────────────────
def hero(title: str, sub: str, tags: list[tuple[str, str]] | None = None) -> str:
    tag_html = ""
    if tags:
        tag_html = '<div class="hero-tags">' + \
            "".join(f'<span class="htag htag-{cls}">{t}</span>' for t, cls in tags) + \
            '</div>'
    return f"""
    <div class="hero">
        <div class="hero-scan"></div>
        <div class="hero-grid"></div>
        <div class="hero-dot">🔬</div>
        <div class="hero-title">{title}</div>
        <div class="hero-sub">{sub}</div>
        {tag_html}
    </div>"""

def sh(label: str) -> str:
    return f'<div class="sh"><div class="sh-bar"></div><div class="sh-text">{label}</div></div>'

def kpi_card(icon, label, value, note, color_class) -> str:
    return f"""
    <div class="kpi {color_class}">
        <div class="kpi-bar"></div>
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-lbl">{label}</div>
        <div class="kpi-val">{value}</div>
        <div class="kpi-note">{note}</div>
    </div>"""

# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 Navigation")
    page = st.radio(
        "Navigation",
        ["🏠 Overview", "📊 Model Comparison", "🔍 Deep Dive",
         "🎯 Best Model: ViT", "📈 Metric Explorer"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("### ⚙️ Filters")
    sel_models = st.multiselect("Models", MODELS, default=MODELS)
    sel_mags   = st.multiselect("Magnifications", MAGS, default=MAGS)
    sel_metric = st.selectbox(
        "Primary Metric", METRICS,
        format_func=lambda x: METRIC_LABELS[x]
    )
    st.markdown("---")
    st.markdown(f"""
    <div style='font-family:"Space Mono",monospace; font-size:0.65rem;
                color:#1e3048; line-height:2.2; letter-spacing:0.5px;'>
    DATASET &nbsp;&nbsp; BreaKHis v1<br>
    TASK &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Benign vs Malignant<br>
    MAGS &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 40X · 100X · 200X · 400X<br>
    MODELS &nbsp;&nbsp;&nbsp; ViT · ResNet · Dense · Effnet<br>
    BEST &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color:#22d3ee;">ViT @ 100X</span>
    </div>
    """, unsafe_allow_html=True)

# ─── Guard helpers ───────────────────────────────────────────────────────────────
def require_models() -> list:
    """Return selected models, or stop with a warning if none selected."""
    active = [m for m in sel_models if m in MODELS]
    if not active:
        st.warning("⚠️ Please select at least one model from the sidebar.", icon="⚠️")
        st.stop()
    return active

def active_mags() -> list:
    """Return selected magnifications, fall back to all if none selected."""
    return sel_mags if sel_mags else MAGS

# ════════════════════════════════════════════════════════════════════════════════
# PAGE — Overview
# ════════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown(hero(
        "Brea<em>K</em>His Cancer <em>Classification</em>",
        "Multi-Model · Multi-Magnification · Breast Histopathology Binary Classification",
        [("ViT · ResNet-50 · DenseNet-121 · EfficientNet", "c"),
         ("40X · 100X · 200X · 400X", "p"),
         ("BreaKHis v1", "g")]
    ), unsafe_allow_html=True)

    # KPI Row
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("c1", "🏆", "Best Model",   "ViT",    "Vision Transformer",   "amber"),
        ("c2", "✅", "Peak Accuracy","99.36%", "ViT @ 100X",           "green"),
        ("c3", "📡", "Best AUC-ROC", "99.98%", "ViT @ 100X",           "cyan"),
        ("c4", "🎯", "Best F1-Score","99.55%", "ViT @ 100X",           "purple"),
    ]
    for col, (_, icon, lbl, val, note, cls) in zip([c1, c2, c3, c4], kpis):
        with col:
            st.markdown(kpi_card(icon, lbl, val, note, cls), unsafe_allow_html=True)

    # Composite Score — FIX: use active models & mags from sidebar filters
    st.markdown(sh("Composite Score Ranking"), unsafe_allow_html=True)
    scores = DATA["composite_scores"]
    # Only show models selected in sidebar
    active_for_overview = [m for m in sel_models if m in MODELS] or MODELS
    filtered_scores = {m: scores[m] for m in active_for_overview}
    sorted_models = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)

    col_l, col_r = st.columns([3, 2])
    with col_l:
        fig = go.Figure()
        for m, s in sorted_models:
            fig.add_trace(go.Bar(
                x=[s * 100], y=[m], orientation='h',
                marker=dict(color=MODEL_COLORS[m], opacity=0.9,
                            line=dict(color="rgba(255,255,255,0.08)", width=1)),
                text=f"  {s * 100:.3f}%",
                textposition='inside', insidetextanchor='end',
                textfont=dict(color="white", size=11, family=_MONO),
                name=m, showlegend=False
            ))
        fig.update_layout(make_layout(
            title=dict(text="Overall Composite Score", font=dict(color="#5d7a99", size=13)),
            xaxis=xax(title="Score (%)", range=[85, 101]),
            yaxis=yax(categoryorder='total ascending'),
            height=270,
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("<br>", unsafe_allow_html=True)
        medals = ["🥇", "🥈", "🥉", "4️⃣"]
        for rank, (m, s) in enumerate(sorted_models):
            c = MODEL_COLORS[m]
            st.markdown(f"""
            <div class="rc" style="border-left-color:{c};">
                <span class="rc-medal">{medals[rank]}</span>
                <div>
                    <span class="pill {PILL_CLASS[m]}">{m}</span>
                    <div class="rc-score" style="color:{c};">{s * 100:.3f}%</div>
                </div>
            </div>""", unsafe_allow_html=True)

    # Heatmap — FIX: respect sidebar model & mag filters
    st.markdown(sh(f"{METRIC_LABELS[sel_metric]} Heatmap — Models × Magnifications"), unsafe_allow_html=True)
    hmap_models = active_for_overview
    hmap_mags   = active_mags()
    if not hmap_mags:
        hmap_mags = MAGS

    heat_z = [[DATA["models"][m][mag][sel_metric] * 100
               for mag in hmap_mags] for m in hmap_models]
    fig_h = go.Figure(go.Heatmap(
        z=heat_z, x=hmap_mags, y=hmap_models,
        colorscale=[[0, "#02060f"], [0.35, "#0a3a4f"], [0.65, "#0e7490"], [1.0, "#22d3ee"]],
        text=[[f"{v:.2f}%" for v in row] for row in heat_z],
        texttemplate="%{text}",
        textfont=dict(size=12, color="white", family=_MONO),
        showscale=True,
        colorbar=dict(
            tickformat=".0f", ticksuffix="%",
            outlinecolor="rgba(34,211,238,0.15)", outlinewidth=1,
            tickfont=dict(color="#2d4562", family=_MONO, size=9)
        )
    ))
    fig_h.update_layout(make_layout(
        xaxis=xax(title="Magnification"),
        yaxis=yax(),
        height=270,
    ))
    st.plotly_chart(fig_h, use_container_width=True)

    # Insights
    st.markdown(sh("Key Insights"), unsafe_allow_html=True)
    i1, i2 = st.columns(2)
    with i1:
        st.markdown('<div class="ic success"><b>🏆 ViT Dominates Across All Magnifications</b><br>Vision Transformer achieves the highest accuracy at every magnification, peaking at <b>99.36% accuracy</b> and <b>99.98% AUC-ROC</b> at 100X — near-perfect clinical performance.</div>', unsafe_allow_html=True)
        st.markdown('<div class="ic info"><b>📡 100X is the Sweet Spot</b><br>Across all 4 models, 100X magnification consistently yields the best or near-best performance — optimal tissue detail for binary classification.</div>', unsafe_allow_html=True)
    with i2:
        st.markdown('<div class="ic warning"><b>⚠️ ResNet-50 Recall Problem</b><br>ResNet-50 recall drops to <b>74–86%</b>, missing actual cancer cases — a critical flaw in medical diagnostics where false negatives are life-threatening.</div>', unsafe_allow_html=True)
        st.markdown('<div class="ic purple"><b>🔍 DenseNet-121 Strong Runner-Up</b><br>97.9% composite score and 98.4% accuracy at 100X makes DenseNet-121 the best lightweight alternative when transformer compute is unavailable.</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE — Model Comparison
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.markdown(hero(
        "Model <em>Comparison</em>",
        "Head-to-head performance across all metrics and magnifications"
    ), unsafe_allow_html=True)

    active = require_models()
    cur_mags = active_mags()   # FIX: use filtered magnifications

    # Tabbed bar charts — FIX: only show tabs for selected magnifications
    st.markdown(sh("Metric Comparison by Magnification"), unsafe_allow_html=True)
    tabs = st.tabs(cur_mags)
    for tab, mag in zip(tabs, cur_mags):
        with tab:
            fig = go.Figure()
            for m in active:
                vals = [DATA["models"][m][mag][met] * 100 for met in METRICS]
                fig.add_trace(go.Bar(
                    name=m, x=[METRIC_LABELS[k] for k in METRICS], y=vals,
                    marker_color=MODEL_COLORS[m], marker_opacity=0.88,
                    text=[f"{v:.1f}%" for v in vals], textposition="outside",
                    textfont=dict(size=9, family=_MONO)
                ))
            fig.update_layout(make_layout(
                barmode="group",
                title=dict(text=f"All Metrics @ {mag}", font=dict(color="#5d7a99", size=13)),
                xaxis=xax(),
                yaxis=yax(title="Score (%)", range=[60, 107]),
                height=420,
                legend=_LEG_H,
            ))
            st.plotly_chart(fig, use_container_width=True)

    # Radar — FIX: use cur_mags for averaging + fix fillcolor (rgba, not 8-char hex)
    st.markdown(sh("Radar — Average Performance Profile (Selected Mags)"), unsafe_allow_html=True)
    fig_r = go.Figure()
    r_metrics = METRICS + [METRICS[0]]
    r_labels  = [METRIC_LABELS[m] for m in r_metrics]
    for m in active:
        # Average only over the currently selected magnifications
        avgs = [np.mean([DATA["models"][m][mg][met] for mg in cur_mags]) * 100
                for met in METRICS]
        avgs += [avgs[0]]
        fig_r.add_trace(go.Scatterpolar(
            r=avgs, theta=r_labels, fill='toself', name=m,
            line_color=MODEL_COLORS[m],
            # FIX: rgba() instead of 8-char hex — Plotly does not accept #rrggbbAA
            fillcolor=hex_to_rgba(MODEL_COLORS[m], 0.094),
            marker=dict(size=6, color=MODEL_COLORS[m])
        ))
    fig_r.update_layout(make_layout(
        polar=dict(
            bgcolor="rgba(3,12,26,0.97)",
            radialaxis=dict(
                visible=True, range=[85, 101],
                gridcolor="rgba(34,211,238,0.07)",
                tickfont=dict(color="#2d4562", size=8, family=_MONO),
            ),
            angularaxis=dict(
                gridcolor="rgba(34,211,238,0.07)",
                tickfont=dict(color="#5d7a99", size=11, family="'Rajdhani',sans-serif"),
            )
        ),
        title=dict(text="Model Capability Radar", font=dict(color="#5d7a99", size=13)),
        height=460,
        legend={**_LEG_H, "y": -0.18},
    ))
    st.plotly_chart(fig_r, use_container_width=True)

    # Summary table — FIX: respect both model and mag filters
    st.markdown(sh("Summary Statistics Table"), unsafe_allow_html=True)
    rows = []
    for m in active:
        for mag in cur_mags:
            d = DATA["models"][m][mag]
            rows.append({
                "Model": m, "Magnification": mag,
                **{METRIC_LABELS[k]: pct(d[k]) for k in METRICS}
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE — Deep Dive
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Deep Dive":
    st.markdown(hero(
        "Deep <em>Dive</em> Analysis",
        "Magnification trends, precision–recall trade-offs and performance degradation"
    ), unsafe_allow_html=True)

    active   = require_models()
    cur_mags = active_mags()   # FIX: filtered magnifications

    # Trend line — FIX: plot only selected mags
    st.markdown(sh(f"Metric Trend — {METRIC_LABELS[sel_metric]} vs Magnification"), unsafe_allow_html=True)
    fig_l = go.Figure()
    for m in active:
        vals = [DATA["models"][m][mag][sel_metric] * 100 for mag in cur_mags]
        fig_l.add_trace(go.Scatter(
            x=cur_mags, y=vals, name=m, mode="lines+markers",
            line=dict(color=MODEL_COLORS[m], width=2.5),
            marker=dict(size=9, color=MODEL_COLORS[m],
                        line=dict(color="#02060f", width=2)),
            hovertemplate=f"<b>{m}</b><br>%{{x}}: %{{y:.3f}}%<extra></extra>"
        ))
    fig_l.update_layout(make_layout(
        title=dict(text=f"{METRIC_LABELS[sel_metric]} vs Magnification",
                   font=dict(color="#5d7a99", size=13)),
        xaxis=xax(title="Magnification"),
        yaxis=yax(title=f"{METRIC_LABELS[sel_metric]} (%)", range=[70, 101]),
        height=420, legend=_LEG_H,
    ))
    st.plotly_chart(fig_l, use_container_width=True)

    # Precision vs Recall — FIX: use filtered mags
    st.markdown(sh("Precision vs Recall Trade-off"), unsafe_allow_html=True)
    fig_pr = go.Figure()
    for m in active:
        prec = [DATA["models"][m][mag]["precision"] * 100 for mag in cur_mags]
        rec  = [DATA["models"][m][mag]["recall"] * 100 for mag in cur_mags]
        fig_pr.add_trace(go.Scatter(
            x=rec, y=prec, mode="markers+text", name=m,
            marker=dict(size=16, color=MODEL_COLORS[m], opacity=0.88,
                        line=dict(color="white", width=1.5)),
            text=cur_mags, textposition="top center",
            textfont=dict(size=8, color="#5d7a99", family=_MONO),
            hovertemplate=f"<b>{m}</b><br>Recall: %{{x:.2f}}%<br>Precision: %{{y:.2f}}%<extra></extra>"
        ))
    fig_pr.add_shape(
        type="line", x0=70, y0=70, x1=102, y1=102,
        line=dict(color="rgba(34,211,238,0.12)", dash="dot", width=1.5)
    )
    fig_pr.update_layout(make_layout(
        title=dict(text="Precision vs Recall (each dot = one magnification)",
                   font=dict(color="#5d7a99", size=13)),
        xaxis=xax(title="Recall (%)", range=[68, 102]),
        yaxis=yax(title="Precision (%)", range=[93, 101]),
        height=450, legend=_LEG_H,
    ))
    st.plotly_chart(fig_pr, use_container_width=True)

    # Degradation — FIX: only show when both 40X and 400X are selected
    st.markdown(sh("Performance Degradation: 40X → 400X"), unsafe_allow_html=True)
    if "40X" not in cur_mags or "400X" not in cur_mags:
        st.info("ℹ️ Select both **40X** and **400X** magnifications in the sidebar to see the degradation chart.")
    else:
        drop_rows = []
        for m in active:
            base  = DATA["models"][m]["40X"][sel_metric] * 100
            end   = DATA["models"][m]["400X"][sel_metric] * 100
            delta = base - end
            drop_rows.append({
                "Model": m,
                "40X":  f"{base:.2f}%",
                "400X": f"{end:.2f}%",
                "Δ Drop (pp)": f"{delta:+.2f}",
                "_delta": delta,
            })
        drop_df = pd.DataFrame(drop_rows)

        fig_d = go.Figure()
        for _, row in drop_df.iterrows():
            clr = "#fb923c" if row["_delta"] > 1.0 else "#34d399"
            fig_d.add_trace(go.Bar(
                x=[row["Model"]], y=[row["_delta"]],
                marker_color=clr,
                text=[f"{row['_delta']:+.2f} pp"],
                textposition="outside",
                textfont=dict(size=10, family=_MONO),
                showlegend=False,
            ))
        fig_d.update_layout(make_layout(
            title=dict(text=f"{METRIC_LABELS[sel_metric]} Drop: 40X → 400X",
                       font=dict(color="#5d7a99", size=13)),
            xaxis=xax(),
            yaxis=yax(title="Δ Score (pp)", zeroline=True,
                      zerolinecolor="rgba(34,211,238,0.15)"),
            height=340,
        ))
        st.plotly_chart(fig_d, use_container_width=True)
        st.dataframe(drop_df.drop(columns=["_delta"]), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE — Best Model: ViT
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Best Model: ViT":
    st.markdown(hero(
        "Vision <em>Transformer</em> (ViT)",
        "Best performing model — Detailed performance profile & training configuration",
        [("vit_small_patch16_224", "c"), ("AdamW · LR=1e-4", "g"),
         ("15 Epochs · Early Stop=5", "p"), ("Tesla T4 · 15.64 GB", "o")]
    ), unsafe_allow_html=True)

    vit      = DATA["models"]["ViT"]
    cur_mags = active_mags()   # FIX: respect sidebar filter

    # Per-mag KPIs — FIX: only show KPIs for selected mags
    kpi_cols = st.columns(len(cur_mags)) if cur_mags else st.columns(4)
    vit_cls  = ["cyan", "amber", "cyan", "cyan"]
    for col, mag in zip(kpi_cols, cur_mags):
        acc  = vit[mag]["accuracy"] * 100
        star = " ⭐" if mag == "100X" else ""
        idx  = MAGS.index(mag) if mag in MAGS else 0
        with col:
            st.markdown(kpi_card("🔭", f"ViT @ {mag}{star}", f"{acc:.2f}%", "Accuracy",
                                 vit_cls[idx % len(vit_cls)]),
                        unsafe_allow_html=True)

    # Multi-metric line — FIX: use filtered mags
    st.markdown(sh("ViT — All Metrics Across Magnifications"), unsafe_allow_html=True)
    met_colors = {
        "accuracy": "#22d3ee", "auc_roc": "#34d399",
        "f1_score": "#fbbf24", "recall": "#fb923c", "precision": "#a78bfa"
    }
    fig_v = go.Figure()
    for met in METRICS:
        vals = [vit[mag][met] * 100 for mag in cur_mags]
        fig_v.add_trace(go.Scatter(
            x=cur_mags, y=vals, name=METRIC_LABELS[met], mode="lines+markers",
            line=dict(color=met_colors[met], width=2.5),
            marker=dict(size=9, color=met_colors[met],
                        line=dict(color="#02060f", width=2)),
            hovertemplate=f"<b>{METRIC_LABELS[met]}</b>: %{{y:.3f}}%<extra></extra>"
        ))
    fig_v.update_layout(make_layout(
        title=dict(text="ViT Performance Profile", font=dict(color="#5d7a99", size=13)),
        xaxis=xax(title="Magnification"),
        yaxis=yax(title="Score (%)", range=[96, 101]),
        height=400, legend=_LEG_H,
    ))
    st.plotly_chart(fig_v, use_container_width=True)

    # Subplots — FIX: only for selected mags
    st.markdown(sh("ViT Metric Breakdown — Per Magnification"), unsafe_allow_html=True)
    n_cols  = len(cur_mags) if cur_mags else 1
    fig_sub = make_subplots(
        rows=1, cols=n_cols,
        subplot_titles=cur_mags if cur_mags else ["No magnification selected"],
        shared_yaxes=True
    )
    for i, mag in enumerate(cur_mags, 1):
        met_vals = [vit[mag][m] * 100 for m in METRICS]
        fig_sub.add_trace(go.Bar(
            x=[METRIC_LABELS[m] for m in METRICS],
            y=met_vals,
            marker_color=[met_colors[m] for m in METRICS],
            text=[f"{v:.1f}" for v in met_vals],
            textposition="outside",
            textfont=dict(size=8, family=_MONO),
            showlegend=False,
        ), row=1, col=i)

    fig_sub.update_layout(make_layout(height=380))
    _sub_ax = dict(
        tickangle=35,
        tickfont=dict(size=7, color="#2d4562", family=_MONO),
        gridcolor="rgba(34,211,238,0.05)",
        linecolor="rgba(34,211,238,0.08)",
    )
    _sub_yax = dict(
        range=[95, 104],
        gridcolor="rgba(34,211,238,0.05)",
        tickfont=dict(size=8, color="#2d4562", family=_MONO),
        linecolor="rgba(34,211,238,0.08)",
    )
    for i in range(1, n_cols + 1):
        xk = "xaxis" if i == 1 else f"xaxis{i}"
        yk = "yaxis" if i == 1 else f"yaxis{i}"
        fig_sub.update_layout(**{xk: _sub_ax, yk: _sub_yax})
    st.plotly_chart(fig_sub, use_container_width=True)

    # Config cards (static — not filter-dependent)
    st.markdown(sh("Training Configuration"), unsafe_allow_html=True)
    configs = [
        ("🧠", "Architecture",    "vit_small_patch16_224"),
        ("📦", "Batch Size",       "32 (Effective: 64)"),
        ("📉", "Learning Rate",    "1e-4"),
        ("🔄", "Epochs",           "15"),
        ("🛑", "Early Stopping",   "Patience = 5"),
        ("⚡", "GPU",              "Tesla T4 (15.64 GB)"),
        ("⚙️", "Optimizer",        "AdamW (wd=0.01)"),
        ("🎯", "Loss",             "Cross-Entropy + AMP"),
        ("🖼️", "Augmentation",     "RandomFlip + ColorJitter"),
    ]
    cols3 = st.columns(3)
    for idx, (icon, key, val) in enumerate(configs):
        with cols3[idx % 3]:
            st.markdown(f"""
            <div class="cc">
                <div class="cc-top">
                    <span class="cc-icon">{icon}</span>
                    <span class="cc-key">{key}</span>
                </div>
                <div class="cc-val">{val}</div>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE — Metric Explorer
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📈 Metric Explorer":
    st.markdown(hero(
        "Metric <em>Explorer</em>",
        "Interactive exploration of any metric × model × magnification combination"
    ), unsafe_allow_html=True)

    active   = require_models()
    cur_mags = active_mags()   # FIX: filtered magnifications

    # Heatmap — FIX: respect model + mag filters
    st.markdown(sh(f"Heatmap — {METRIC_LABELS[sel_metric]}"), unsafe_allow_html=True)
    hz = [[DATA["models"][m][mag][sel_metric] * 100
           for mag in cur_mags] for m in active]
    cs_map = {
        "recall":    [[0, "#1f0505"], [0.3, "#7b1f1f"], [0.6, "#c53030"], [1.0, "#fc8181"]],
        "precision": [[0, "#02060f"], [0.3, "#0a3a4f"], [0.6, "#0e7490"], [1.0, "#22d3ee"]],
        "auc_roc":   [[0, "#02060f"], [0.3, "#1a2d3d"], [0.6, "#2563eb"], [1.0, "#93c5fd"]],
        "f1_score":  [[0, "#0d0a02"], [0.3, "#3d2b00"], [0.6, "#ca8a04"], [1.0, "#fbbf24"]],
        "accuracy":  [[0, "#060202"], [0.3, "#3d1515"], [0.6, "#7c3aed"], [1.0, "#a78bfa"]],
    }
    fig_hm = go.Figure(go.Heatmap(
        z=hz, x=cur_mags, y=active,
        colorscale=cs_map.get(sel_metric, cs_map["accuracy"]),
        text=[[f"{v:.2f}%" for v in row] for row in hz],
        texttemplate="%{text}",
        textfont=dict(size=12, color="white", family=_MONO),
        showscale=True,
        colorbar=dict(
            tickformat=".0f", ticksuffix="%",
            outlinecolor="rgba(34,211,238,0.15)", outlinewidth=1,
            tickfont=dict(color="#2d4562", family=_MONO, size=9)
        )
    ))
    fig_hm.update_layout(make_layout(
        title=dict(text=f"{METRIC_LABELS[sel_metric]} — Selected Models × Magnifications",
                   font=dict(color="#5d7a99", size=13)),
        xaxis=xax(title="Magnification"),
        yaxis=yax(),
        height=295,
    ))
    st.plotly_chart(fig_hm, use_container_width=True)

    # Box plot — FIX: rgba fillcolor instead of 8-char hex; use filtered mags
    st.markdown(sh(f"Score Distribution — {METRIC_LABELS[sel_metric]}"), unsafe_allow_html=True)
    fig_b = go.Figure()
    for m in active:
        vals = [DATA["models"][m][mag][sel_metric] * 100 for mag in cur_mags]
        fig_b.add_trace(go.Box(
            y=vals, name=m,
            marker_color=MODEL_COLORS[m],
            line_color=MODEL_COLORS[m],
            # FIX: rgba() instead of 8-char hex — Plotly does not accept #rrggbbAA
            fillcolor=hex_to_rgba(MODEL_COLORS[m], 0.13),
            boxmean='sd',
            boxpoints="all", jitter=0.35, pointpos=-1.5,
            marker=dict(size=9, line=dict(color="white", width=1)),
        ))
    fig_b.update_layout(make_layout(
        title=dict(text=f"{METRIC_LABELS[sel_metric]} Distribution (selected magnifications)",
                   font=dict(color="#5d7a99", size=13)),
        xaxis=xax(),
        yaxis=yax(title=f"{METRIC_LABELS[sel_metric]} (%)"),
        height=420, legend=_LEG_V,
    ))
    st.plotly_chart(fig_b, use_container_width=True)

    # Full results table — FIX: respect both filters
    st.markdown(sh("Complete Results Table"), unsafe_allow_html=True)
    all_rows = []
    for m in active:
        for mag in cur_mags:
            d = DATA["models"][m][mag]
            all_rows.append({
                "Model": m, "Mag": mag,
                **{METRIC_LABELS[k]: pct(d[k]) for k in METRICS}
            })
    st.dataframe(pd.DataFrame(all_rows), use_container_width=True, hide_index=True)

# ─── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🔬 &nbsp; BreaKHis Breast Cancer Histopathology &nbsp;·&nbsp;
    Multi-Model Classification Dashboard &nbsp;·&nbsp;
    Built with Streamlit + Plotly &nbsp;·&nbsp; Dataset: BreaKHis v1
</div>
""", unsafe_allow_html=True)
