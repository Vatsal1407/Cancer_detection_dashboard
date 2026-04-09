import streamlit as st
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BreakHis Cancer Classification Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0f1117; }

    .hero-header {
        background: linear-gradient(135deg, #1a1f35 0%, #0d1b2a 50%, #1a2744 100%);
        border: 1px solid #2d3748;
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 28px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(99,179,237,0.05) 0%, transparent 60%);
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #718096;
        margin-top: 8px;
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, #2b6cb0, #3182ce);
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-top: 12px;
        letter-spacing: 0.5px;
    }

    .kpi-card {
        background: linear-gradient(135deg, #1a1f35 0%, #161b2e 100%);
        border: 1px solid #2d3748;
        border-radius: 14px;
        padding: 22px 24px;
        text-align: center;
        transition: transform 0.2s, border-color 0.2s;
        height: 100%;
    }
    .kpi-card:hover { transform: translateY(-2px); border-color: #4a5568; }
    .kpi-label { font-size: 0.75rem; font-weight: 600; color: #718096; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
    .kpi-value { font-size: 2.2rem; font-weight: 700; color: #e2e8f0; line-height: 1; }
    .kpi-sub { font-size: 0.78rem; color: #4a9eed; margin-top: 6px; }

    .kpi-card.gold { border-color: #d4a017; background: linear-gradient(135deg, #1f1a0d 0%, #161108 100%); }
    .kpi-card.gold .kpi-value { color: #f6c843; }
    .kpi-card.green { border-color: #276749; background: linear-gradient(135deg, #0d1f15 0%, #081510 100%); }
    .kpi-card.green .kpi-value { color: #68d391; }
    .kpi-card.blue { border-color: #2b6cb0; background: linear-gradient(135deg, #0d1728 0%, #081020 100%); }
    .kpi-card.blue .kpi-value { color: #63b3ed; }
    .kpi-card.purple { border-color: #553c9a; background: linear-gradient(135deg, #160d2a 0%, #0e0820 100%); }
    .kpi-card.purple .kpi-value { color: #b794f4; }

    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 28px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #2d3748;
    }

    .insight-card {
        background: #1a1f35;
        border: 1px solid #2d3748;
        border-left: 4px solid #3182ce;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 10px;
        font-size: 0.88rem;
        color: #cbd5e0;
        line-height: 1.6;
    }
    .insight-card.warning { border-left-color: #ed8936; }
    .insight-card.success { border-left-color: #48bb78; }
    .insight-card.info    { border-left-color: #805ad5; }

    .model-pill {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
    }
    .pill-vit      { background: rgba(99,179,237,0.15); color: #63b3ed; border: 1px solid #2b6cb0; }
    .pill-resnet   { background: rgba(252,129,74,0.15); color: #fc814a; border: 1px solid #c05621; }
    .pill-densenet { background: rgba(104,211,145,0.15); color: #68d391; border: 1px solid #276749; }
    .pill-effnet   { background: rgba(183,148,244,0.15); color: #b794f4; border: 1px solid #553c9a; }

    .stSelectbox label, .stMultiSelect label, .stSlider label { color: #a0aec0 !important; font-size: 0.85rem !important; }
    div[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #1f2937; }

    .footer {
        text-align: center;
        color: #4a5568;
        font-size: 0.78rem;
        margin-top: 40px;
        padding: 16px;
        border-top: 1px solid #1f2937;
    }
</style>
""", unsafe_allow_html=True)

# ─── Data ───────────────────────────────────────────────────────────────────────
DATA = {
    "best_model": "ViT",
    "best_mag": "100X",
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

MODELS = list(DATA["models"].keys())
MAGS   = ["40X", "100X", "200X", "400X"]
METRICS = ["accuracy", "auc_roc", "f1_score", "recall", "precision"]
METRIC_LABELS = {
    "accuracy": "Accuracy", "auc_roc": "AUC-ROC",
    "f1_score": "F1-Score", "recall": "Recall", "precision": "Precision"
}

MODEL_COLORS = {
    "ViT": "#63b3ed",
    "ResNet-50": "#fc814a",
    "DenseNet-121": "#68d391",
    "EfficientNet": "#b794f4"
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,17,27,0.8)",
    font=dict(family="Inter", color="#a0aec0", size=12),
    xaxis=dict(gridcolor="#1f2937", linecolor="#2d3748", tickfont=dict(color="#718096")),
    yaxis=dict(gridcolor="#1f2937", linecolor="#2d3748", tickfont=dict(color="#718096")),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2d3748", borderwidth=1),
    margin=dict(l=40, r=20, t=40, b=40)
)

def fmt(val): return f"{val*100:.2f}%"

# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 Navigation")
    page = st.radio("", ["🏠 Overview", "📊 Model Comparison", "🔍 Deep Dive", "🎯 Best Model: ViT", "📈 Metric Explorer"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### ⚙️ Filters")
    sel_models = st.multiselect("Models", MODELS, default=MODELS)
    sel_mags   = st.multiselect("Magnifications", MAGS, default=MAGS)
    sel_metric = st.selectbox("Primary Metric", METRICS, format_func=lambda x: METRIC_LABELS[x])

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; color:#4a5568; line-height:1.6'>
    <b style='color:#718096'>Dataset:</b> BreaKHis v1<br>
    <b style='color:#718096'>Task:</b> Benign vs Malignant<br>
    <b style='color:#718096'>Magnifications:</b> 40X · 100X · 200X · 400X<br>
    <b style='color:#718096'>Models:</b> ViT · ResNet-50 · DenseNet-121 · EfficientNet
    </div>
    """, unsafe_allow_html=True)

# ─── Page: Overview ─────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">🔬 BreakHis Breast Cancer Classification</div>
        <div class="hero-subtitle">Multi-Model · Multi-Magnification · Histopathology Analysis</div>
        <div class="hero-badge">Review-2 Dashboard · 2025</div>
    </div>
    """, unsafe_allow_html=True)

    # KPI Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class="kpi-card gold">
            <div class="kpi-label">🏆 Best Model</div>
            <div class="kpi-value">ViT</div>
            <div class="kpi-sub">Vision Transformer</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card green">
            <div class="kpi-label">✅ Peak Accuracy</div>
            <div class="kpi-value">99.36%</div>
            <div class="kpi-sub">ViT @ 100X</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card blue">
            <div class="kpi-label">📡 Best AUC-ROC</div>
            <div class="kpi-value">99.98%</div>
            <div class="kpi-sub">ViT @ 100X</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-card purple">
            <div class="kpi-label">🎯 Best F1-Score</div>
            <div class="kpi-value">99.55%</div>
            <div class="kpi-sub">ViT @ 100X</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Composite Score Ranking</div>', unsafe_allow_html=True)

    scores = DATA["composite_scores"]
    sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    col_left, col_right = st.columns([3, 2])
    with col_left:
        fig = go.Figure()
        for rank, (m, s) in enumerate(sorted_models):
            color = MODEL_COLORS[m]
            fig.add_trace(go.Bar(
                x=[s * 100],
                y=[m],
                orientation='h',
                marker=dict(
                    color=color,
                    line=dict(color="rgba(255,255,255,0.1)", width=1),
                    opacity=0.9
                ),
                text=f"{s*100:.2f}%",
                textposition='inside',
                insidetextanchor='end',
                textfont=dict(color="white", size=13, family="Inter"),
                name=m,
                showlegend=False
            ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Overall Composite Score", font=dict(color="#e2e8f0", size=14)),
            xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title="Score (%)", range=[85, 101]),
            yaxis=dict(**PLOTLY_LAYOUT["yaxis"], categoryorder='total ascending'),
            height=280
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("<br>", unsafe_allow_html=True)
        for rank, (m, s) in enumerate(sorted_models):
            medal = ["🥇", "🥈", "🥉", "4️⃣"][rank]
            pill_class = {"ViT": "pill-vit", "ResNet-50": "pill-resnet",
                          "DenseNet-121": "pill-densenet", "EfficientNet": "pill-effnet"}[m]
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px; 
                        background:#1a1f35; border:1px solid #2d3748; border-radius:10px; padding:12px 16px;">
                <span style="font-size:1.4rem">{medal}</span>
                <div>
                    <span class="model-pill {pill_class}">{m}</span>
                    <div style="font-size:1.3rem; font-weight:700; color:#e2e8f0; margin-top:2px">
                        {s*100:.3f}%
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    # Heatmap of all results
    st.markdown('<div class="section-header">Performance Heatmap — Accuracy across Models × Magnifications</div>', unsafe_allow_html=True)

    heat_data = [[DATA["models"][m][mag]["accuracy"] * 100 for mag in MAGS] for m in MODELS]
    fig_heat = go.Figure(go.Heatmap(
        z=heat_data,
        x=MAGS,
        y=MODELS,
        colorscale=[[0, "#1a1f35"], [0.4, "#2b6cb0"], [0.7, "#3182ce"], [1.0, "#63b3ed"]],
        text=[[f"{v:.2f}%" for v in row] for row in heat_data],
        texttemplate="%{text}",
        textfont=dict(size=13, color="white"),
        showscale=True,
        colorbar=dict(
            tickformat=".0f",
            ticksuffix="%",
            outlinecolor="#2d3748",
            outlinewidth=1,
            tickfont=dict(color="#a0aec0")
        )
    ))
    fig_heat.update_layout(
        **PLOTLY_LAYOUT,
        height=280,
        xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title="Magnification"),
        yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title=""),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Insights
    st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)
    i1, i2 = st.columns(2)
    with i1:
        st.markdown("""<div class="insight-card success">
            <b>🏆 ViT Dominates Across All Magnifications</b><br>
            Vision Transformer achieves the highest accuracy at every magnification level, 
            peaking at <b>99.36%</b> accuracy and <b>99.98% AUC-ROC</b> at 100X — near-perfect clinical performance.
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="insight-card">
            <b>📡 100X is the Sweet Spot</b><br>
            Across all 4 models, 100X magnification consistently yields the best or second-best 
            performance, suggesting it provides optimal tissue detail for classification.
        </div>""", unsafe_allow_html=True)
    with i2:
        st.markdown("""<div class="insight-card warning">
            <b>⚠️ ResNet-50 Recall Problem</b><br>
            ResNet-50 shows significantly lower recall (74–86%), meaning it misses actual cancer 
            cases — a critical flaw in a medical diagnostic setting where false negatives are dangerous.
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="insight-card info">
            <b>🔍 DenseNet-121 Competitive Runner-Up</b><br>
            DenseNet-121 scores <b>97.9%</b> composite and reaches <b>98.4% accuracy</b> at 100X, 
            making it a solid choice when transformer compute is unavailable.
        </div>""", unsafe_allow_html=True)

# ─── Page: Model Comparison ──────────────────────────────────────────────────────
elif page == "📊 Model Comparison":
    st.markdown('<div class="hero-header"><div class="hero-title">📊 Model Comparison</div><div class="hero-subtitle">Head-to-head performance across all metrics</div></div>', unsafe_allow_html=True)

    active_models = [m for m in sel_models if m in MODELS]
    if not active_models:
        st.warning("Please select at least one model from the sidebar.")
        st.stop()

    # Grouped bar chart
    st.markdown('<div class="section-header">Metric Comparison by Magnification</div>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["40X", "100X", "200X", "400X"])
    for tab, mag in zip([tab1, tab2, tab3, tab4], MAGS):
        with tab:
            fig = go.Figure()
            for m in active_models:
                vals = [DATA["models"][m][mag][met] * 100 for met in METRICS]
                fig.add_trace(go.Bar(
                    name=m,
                    x=[METRIC_LABELS[k] for k in METRICS],
                    y=vals,
                    marker_color=MODEL_COLORS[m],
                    marker_opacity=0.85,
                    text=[f"{v:.1f}%" for v in vals],
                    textposition="outside",
                    textfont=dict(size=10)
                ))
            fig.update_layout(
                **PLOTLY_LAYOUT,
                barmode="group",
                title=dict(text=f"All Metrics @ {mag}", font=dict(color="#e2e8f0", size=14)),
                yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title="Score (%)", range=[60, 105]),
                height=420,
                legend=dict(**PLOTLY_LAYOUT["legend"], orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)

    # Radar chart
    st.markdown('<div class="section-header">Radar Chart — Average Performance Profile</div>', unsafe_allow_html=True)
    fig_radar = go.Figure()
    radar_metrics = METRICS + [METRICS[0]]
    radar_labels  = [METRIC_LABELS[m] for m in radar_metrics]
    for m in active_models:
        avg_vals = [np.mean([DATA["models"][m][mag][met] for mag in MAGS]) * 100 for met in METRICS]
        avg_vals += [avg_vals[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=avg_vals,
            theta=radar_labels,
            fill='toself',
            name=m,
            line_color=MODEL_COLORS[m],
            fillcolor=MODEL_COLORS[m].replace(")", ",0.15)").replace("rgb", "rgba") if "rgb" in MODEL_COLORS[m] else MODEL_COLORS[m] + "26",
            marker=dict(size=6)
        ))
    fig_radar.update_layout(
        **PLOTLY_LAYOUT,
        polar=dict(
            bgcolor="rgba(13,17,27,0.9)",
            radialaxis=dict(visible=True, range=[85, 101], gridcolor="#2d3748", tickfont=dict(color="#718096")),
            angularaxis=dict(gridcolor="#2d3748", tickfont=dict(color="#a0aec0"))
        ),
        title=dict(text="Model Capability Radar (Avg across Magnifications)", font=dict(color="#e2e8f0", size=14)),
        height=480,
        legend=dict(**PLOTLY_LAYOUT["legend"], orientation="h", yanchor="bottom", y=-0.15)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Summary table
    st.markdown('<div class="section-header">Summary Statistics Table</div>', unsafe_allow_html=True)
    rows = []
    for m in active_models:
        for mag in sel_mags:
            d = DATA["models"][m][mag]
            rows.append({
                "Model": m, "Magnification": mag,
                "Accuracy": fmt(d["accuracy"]),
                "AUC-ROC": fmt(d["auc_roc"]),
                "F1-Score": fmt(d["f1_score"]),
                "Recall": fmt(d["recall"]),
                "Precision": fmt(d["precision"])
            })
    df = pd.DataFrame(rows)
    st.dataframe(df.style.set_properties(**{"background-color": "#1a1f35", "color": "#e2e8f0",
                                            "border-color": "#2d3748"}),
                 use_container_width=True, hide_index=True)

# ─── Page: Deep Dive ─────────────────────────────────────────────────────────────
elif page == "🔍 Deep Dive":
    st.markdown('<div class="hero-header"><div class="hero-title">🔍 Deep Dive Analysis</div><div class="hero-subtitle">Magnification trends and metric correlations</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Metric Trend Across Magnifications</div>', unsafe_allow_html=True)
    fig_line = go.Figure()
    for m in sel_models:
        vals = [DATA["models"][m][mag][sel_metric] * 100 for mag in MAGS]
        fig_line.add_trace(go.Scatter(
            x=MAGS, y=vals, name=m,
            line=dict(color=MODEL_COLORS[m], width=2.5),
            marker=dict(size=9, color=MODEL_COLORS[m], line=dict(color="white", width=1.5)),
            mode="lines+markers",
            text=[f"{v:.2f}%" for v in vals],
            hovertemplate=f"<b>{m}</b><br>%{{x}}: %{{y:.2f}}%<extra></extra>"
        ))
    fig_line.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"{METRIC_LABELS[sel_metric]} vs Magnification", font=dict(color="#e2e8f0", size=14)),
        yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title=f"{METRIC_LABELS[sel_metric]} (%)", range=[70, 101]),
        xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title="Magnification"),
        height=420,
        legend=dict(**PLOTLY_LAYOUT["legend"], orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # Precision vs Recall scatter
    st.markdown('<div class="section-header">Precision vs Recall Trade-off</div>', unsafe_allow_html=True)
    fig_pr = go.Figure()
    for m in sel_models:
        prec = [DATA["models"][m][mag]["precision"] * 100 for mag in MAGS]
        rec  = [DATA["models"][m][mag]["recall"] * 100 for mag in MAGS]
        fig_pr.add_trace(go.Scatter(
            x=rec, y=prec,
            mode="markers+text",
            name=m,
            marker=dict(size=16, color=MODEL_COLORS[m], opacity=0.85,
                        line=dict(color="white", width=1.5)),
            text=MAGS,
            textposition="top center",
            textfont=dict(size=10, color="#a0aec0"),
            hovertemplate=f"<b>{m}</b><br>Recall: %{{x:.2f}}%<br>Precision: %{{y:.2f}}%<extra></extra>"
        ))
    # Perfect classifier line
    fig_pr.add_shape(type="line", x0=95, y0=95, x1=101, y1=101,
                     line=dict(color="#4a5568", dash="dash", width=1))
    fig_pr.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Precision vs Recall by Model & Magnification", font=dict(color="#e2e8f0", size=14)),
        xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title="Recall (%)", range=[70, 102]),
        yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title="Precision (%)", range=[93, 101]),
        height=440,
        legend=dict(**PLOTLY_LAYOUT["legend"], orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig_pr, use_container_width=True)

    # Performance drop analysis
    st.markdown('<div class="section-header">Performance Degradation: 40X → 400X</div>', unsafe_allow_html=True)
    drop_rows = []
    for m in sel_models:
        base = DATA["models"][m]["40X"][sel_metric] * 100
        end  = DATA["models"][m]["400X"][sel_metric] * 100
        drop = base - end
        drop_rows.append({"Model": m, "40X": f"{base:.2f}%", "400X": f"{end:.2f}%",
                           "Drop": f"{drop:+.2f}%", "Drop_val": drop})
    drop_df = pd.DataFrame(drop_rows)

    fig_drop = go.Figure()
    for _, row in drop_df.iterrows():
        color = "#fc814a" if row["Drop_val"] > 1 else "#68d391"
        fig_drop.add_trace(go.Bar(
            x=[row["Model"]], y=[row["Drop_val"]],
            marker_color=color, name=row["Model"],
            text=[f"{row['Drop_val']:+.2f}%"], textposition="outside",
            showlegend=False
        ))
    fig_drop.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"{METRIC_LABELS[sel_metric]} Drop from 40X to 400X", font=dict(color="#e2e8f0", size=14)),
        yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title="Δ Score (pp)", zeroline=True, zerolinecolor="#4a5568"),
        height=340
    )
    st.plotly_chart(fig_drop, use_container_width=True)

    st.dataframe(drop_df.drop(columns=["Drop_val"]).style.set_properties(**{
        "background-color": "#1a1f35", "color": "#e2e8f0", "border-color": "#2d3748"
    }), use_container_width=True, hide_index=True)

# ─── Page: Best Model ViT ────────────────────────────────────────────────────────
elif page == "🎯 Best Model: ViT":
    st.markdown('<div class="hero-header"><div class="hero-title">🎯 Vision Transformer (ViT)</div><div class="hero-subtitle">Best performing model — Detailed analysis</div></div>', unsafe_allow_html=True)

    vit = DATA["models"]["ViT"]

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    for col, mag in zip([c1, c2, c3, c4], MAGS):
        acc = vit[mag]["accuracy"] * 100
        with col:
            st.markdown(f"""<div class="kpi-card {'gold' if mag=='100X' else 'blue'}">
                <div class="kpi-label">{mag}</div>
                <div class="kpi-value">{acc:.2f}%</div>
                <div class="kpi-sub">Accuracy</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">ViT — All Metrics Across Magnifications</div>', unsafe_allow_html=True)

    fig_vit = go.Figure()
    colors_met = {"accuracy": "#63b3ed", "auc_roc": "#68d391",
                  "f1_score": "#f6c843", "recall": "#fc814a", "precision": "#b794f4"}
    for met in METRICS:
        vals = [vit[mag][met] * 100 for mag in MAGS]
        fig_vit.add_trace(go.Scatter(
            x=MAGS, y=vals, name=METRIC_LABELS[met],
            mode="lines+markers",
            line=dict(color=colors_met[met], width=2.5),
            marker=dict(size=9, color=colors_met[met], line=dict(color="white", width=1.5)),
            text=[f"{v:.2f}%" for v in vals],
            hovertemplate=f"<b>{METRIC_LABELS[met]}</b>: %{{y:.2f}}%<extra></extra>"
        ))
    fig_vit.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="ViT Performance Profile", font=dict(color="#e2e8f0", size=14)),
        yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title="Score (%)", range=[96, 101]),
        xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title="Magnification"),
        height=400,
        legend=dict(**PLOTLY_LAYOUT["legend"], orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig_vit, use_container_width=True)

    # Metric breakdown at each magnification
    st.markdown('<div class="section-header">ViT Metric Breakdown — Per Magnification</div>', unsafe_allow_html=True)

    fig_sub = make_subplots(rows=1, cols=4, subplot_titles=MAGS,
                             shared_yaxes=True)
    for i, mag in enumerate(MAGS, 1):
        met_vals = [vit[mag][m] * 100 for m in METRICS]
        fig_sub.add_trace(go.Bar(
            x=[METRIC_LABELS[m] for m in METRICS],
            y=met_vals,
            marker_color=[colors_met[m] for m in METRICS],
            text=[f"{v:.1f}" for v in met_vals],
            textposition="outside",
            textfont=dict(size=9),
            showlegend=False
        ), row=1, col=i)

    fig_sub.update_layout(
        **PLOTLY_LAYOUT,
        height=380,
        yaxis=dict(range=[95, 102], gridcolor="#1f2937", tickfont=dict(color="#718096")),
    )
    for ax in ["xaxis", "xaxis2", "xaxis3", "xaxis4"]:
        fig_sub.update_layout(**{ax: dict(tickangle=30, tickfont=dict(size=9, color="#718096"), gridcolor="#1f2937")})
    for ax in ["yaxis", "yaxis2", "yaxis3", "yaxis4"]:
        fig_sub.update_layout(**{ax: dict(gridcolor="#1f2937", tickfont=dict(color="#718096"))})
    st.plotly_chart(fig_sub, use_container_width=True)

    # Config recap
    st.markdown('<div class="section-header">Training Configuration</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    configs = [
        ("Architecture", "vit_small_patch16_224", "🧠"),
        ("Batch Size", "32 (Effective: 64)", "📦"),
        ("Learning Rate", "1e-4", "📉"),
        ("Epochs", "15", "🔄"),
        ("Early Stopping", "Patience = 5", "🛑"),
        ("GPU", "Tesla T4 (15.64 GB)", "⚡"),
        ("Optimizer", "AdamW (wd=0.01)", "⚙️"),
        ("Loss", "Cross-Entropy + AMP", "🎯"),
        ("Augmentation", "RandomFlip + ColorJitter", "🖼️")
    ]
    for idx, (k, v, icon) in enumerate(configs):
        col = [col1, col2, col3][idx % 3]
        with col:
            st.markdown(f"""
            <div style="background:#1a1f35; border:1px solid #2d3748; border-radius:10px;
                        padding:14px 18px; margin-bottom:10px;">
                <span style="font-size:1.2rem">{icon}</span>
                <span style="font-size:0.75rem; color:#718096; text-transform:uppercase; 
                             letter-spacing:1px; margin-left:8px;">{k}</span>
                <div style="font-size:1rem; color:#e2e8f0; font-weight:600; margin-top:4px">{v}</div>
            </div>""", unsafe_allow_html=True)

# ─── Page: Metric Explorer ───────────────────────────────────────────────────────
elif page == "📈 Metric Explorer":
    st.markdown('<div class="hero-header"><div class="hero-title">📈 Metric Explorer</div><div class="hero-subtitle">Interactive exploration of any metric combination</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Heatmap — Custom Metric</div>', unsafe_allow_html=True)
    heat_data = [[DATA["models"][m][mag][sel_metric] * 100 for mag in MAGS] for m in MODELS]
    ann = [[f"{v:.2f}%" for v in row] for row in heat_data]

    cs = [[0, "#0d1728"], [0.3, "#1a3a6b"], [0.6, "#2b6cb0"], [0.85, "#4299e1"], [1.0, "#63b3ed"]]
    if sel_metric == "recall":
        cs = [[0, "#2d0a0a"], [0.3, "#7b1f1f"], [0.6, "#c53030"], [0.85, "#e53e3e"], [1.0, "#fc8181"]]

    fig_heat2 = go.Figure(go.Heatmap(
        z=heat_data, x=MAGS, y=MODELS, colorscale=cs,
        text=ann, texttemplate="%{text}",
        textfont=dict(size=13, color="white"), showscale=True,
        colorbar=dict(tickformat=".0f", ticksuffix="%",
                      outlinecolor="#2d3748", outlinewidth=1,
                      tickfont=dict(color="#a0aec0"))
    ))
    fig_heat2.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"{METRIC_LABELS[sel_metric]} — All Models × Magnifications",
                   font=dict(color="#e2e8f0", size=14)),
        height=320
    )
    st.plotly_chart(fig_heat2, use_container_width=True)

    # Box plot / distribution
    st.markdown('<div class="section-header">Score Distribution Across Magnifications</div>', unsafe_allow_html=True)
    fig_box = go.Figure()
    for m in sel_models:
        vals = [DATA["models"][m][mag][sel_metric] * 100 for mag in MAGS]
        fig_box.add_trace(go.Box(
            y=vals, name=m,
            marker_color=MODEL_COLORS[m],
            line_color=MODEL_COLORS[m],
            fillcolor=MODEL_COLORS[m] + "33",
            boxmean='sd',
            points="all",
            jitter=0.3,
            pointpos=-1.5,
            marker=dict(size=10)
        ))
    fig_box.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"{METRIC_LABELS[sel_metric]} Distribution (all magnifications)",
                   font=dict(color="#e2e8f0", size=14)),
        yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title=f"{METRIC_LABELS[sel_metric]} (%)"),
        height=400
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # Full numeric table
    st.markdown('<div class="section-header">Complete Results Table</div>', unsafe_allow_html=True)
    all_rows = []
    for m in MODELS:
        for mag in MAGS:
            d = DATA["models"][m][mag]
            all_rows.append({
                "Model": m, "Mag": mag,
                **{METRIC_LABELS[k]: fmt(d[k]) for k in METRICS}
            })
    full_df = pd.DataFrame(all_rows)
    st.dataframe(full_df.style.set_properties(**{
        "background-color": "#1a1f35", "color": "#e2e8f0", "border-color": "#2d3748"
    }), use_container_width=True, hide_index=True)

# ─── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🔬 BreaKHis Breast Cancer Histopathology · Multi-Model Classification Dashboard · 
    Built with Streamlit + Plotly · Dataset: BreaKHis v1
</div>
""", unsafe_allow_html=True)
