# ============================================
# 🎵 InstruNet Dashboard
# CNN-Based Music Instrument Recognition System
# Premium Dark-Themed Analytics Dashboard
# ============================================

import os
import io
import json
import pickle
import warnings

# Suppress torch warnings before importing
warnings.filterwarnings("ignore", category=UserWarning)

import sys

import torch
import torch.nn as nn
from torchvision import models

import numpy as np
import pandas as pd
import librosa
import streamlit as st
import plotly.graph_objects as go

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="InstruNet Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================
# CUSTOM CSS
# ============================================
DARK_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp { background-color: #0d1117 !important; color: #e6edf3 !important; font-family: 'Inter', sans-serif !important; }
    section[data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #30363d !important; }
    section[data-testid="stSidebar"] .stMarkdown p, section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2, section[data-testid="stSidebar"] .stMarkdown h3 { color: #e6edf3 !important; }
    h1, h2, h3, h4, h5, h6 { color: #e6edf3 !important; font-family: 'Inter', sans-serif !important; }
    .section-header { font-size: 0.75rem; font-weight: 700; letter-spacing: 0.15em; text-transform: uppercase; color: #8b949e; margin-top: 2rem; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid #21262d; }
    .metric-card { background: linear-gradient(135deg, #161b22 0%, #1c2333 100%); border: 1px solid #30363d; border-radius: 12px; padding: 1.2rem 1.5rem; transition: all 0.3s ease; position: relative; overflow: hidden; }
    .metric-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #ff6b35, #ff8c42); border-radius: 12px 12px 0 0; }
    .metric-card:hover { border-color: #ff6b35; transform: translateY(-2px); box-shadow: 0 8px 25px rgba(255, 107, 53, 0.15); }
    .metric-label { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #8b949e; margin-bottom: 0.3rem; }
    .metric-value { font-size: 1.8rem; font-weight: 800; color: #e6edf3; line-height: 1.1; }
    .metric-sub { font-size: 0.75rem; color: #3fb950; margin-top: 0.3rem; font-weight: 500; }
    .metric-sub-red { font-size: 0.75rem; color: #f85149; margin-top: 0.3rem; font-weight: 500; }
    .stPlotlyChart { border-radius: 12px; overflow: hidden; }
    .stSelectbox label, .stMultiSelect label, .stSlider label { color: #e6edf3 !important; font-weight: 600 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] { background: #21262d; border-radius: 8px; color: #8b949e; border: 1px solid #30363d; }
    .stTabs [aria-selected="true"] { background: #ff6b35 !important; color: white !important; border-color: #ff6b35 !important; }
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 4px; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    section[data-testid="stFileUploader"] { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 1rem; }
</style>
"""

LIGHT_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp { background-color: #f6f8fa !important; color: #1f2328 !important; font-family: 'Inter', sans-serif !important; }
    section[data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #d0d7de !important; }
    .section-header { font-size: 0.75rem; font-weight: 700; letter-spacing: 0.15em; text-transform: uppercase; color: #656d76; margin-top: 2rem; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid #d0d7de; }
    .metric-card { background: #ffffff; border: 1px solid #d0d7de; border-radius: 12px; padding: 1.2rem 1.5rem; transition: all 0.3s ease; position: relative; overflow: hidden; }
    .metric-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #ff6b35, #ff8c42); border-radius: 12px 12px 0 0; }
    .metric-card:hover { border-color: #ff6b35; transform: translateY(-2px); box-shadow: 0 8px 25px rgba(255, 107, 53, 0.12); }
    .metric-label { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #656d76; margin-bottom: 0.3rem; }
    .metric-value { font-size: 1.8rem; font-weight: 800; color: #1f2328; line-height: 1.1; }
    .metric-sub { font-size: 0.75rem; color: #1a7f37; margin-top: 0.3rem; font-weight: 500; }
    .metric-sub-red { font-size: 0.75rem; color: #cf222e; margin-top: 0.3rem; font-weight: 500; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
"""

# ============================================
# COLOR PALETTE
# ============================================
INSTRUMENT_COLORS = [
    "#ff6b35", "#3fb950", "#58a6ff", "#d2a8ff",
    "#f0883e", "#79c0ff", "#7ee787", "#ffa657",
    "#ff7b72", "#a5d6ff", "#f778ba",
]

INSTRUMENT_NAMES = {
    "cel": "Cello", "cla": "Clarinet", "flu": "Flute",
    "gac": "Acoustic Guitar", "gel": "Electric Guitar",
    "org": "Organ", "pia": "Piano", "sax": "Saxophone",
    "tru": "Trumpet", "vio": "Violin", "voi": "Voice",
}


# ============================================
# THEME HELPER - applies Plotly theme cleanly
# ============================================
def apply_theme(fig, is_dark, height=400, title="", x_title="", y_title="", show_legend=True, x_range=None):
    """Apply theme to a Plotly figure without keyword conflicts."""
    if is_dark:
        paper_bg, plot_bg = "#161b22", "#0d1117"
        font_color, grid_color = "#e6edf3", "#21262d"
        hover_bg, hover_border = "#161b22", "#30363d"
    else:
        paper_bg, plot_bg = "#ffffff", "#f6f8fa"
        font_color, grid_color = "#1f2328", "#e6e9ec"
        hover_bg, hover_border = "#ffffff", "#d0d7de"

    layout_kwargs = dict(
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family="Inter", color=font_color, size=12),
        margin=dict(l=50, r=20, t=60, b=50),
        hoverlabel=dict(bgcolor=hover_bg, font_color=font_color, bordercolor=hover_border),
        height=height,
        showlegend=show_legend,
    )
    if title:
        layout_kwargs["title"] = dict(text=f"<b>{title}</b>", font=dict(size=14, color=font_color))

    fig.update_layout(**layout_kwargs)

    xaxis_kwargs = dict(gridcolor=grid_color, zerolinecolor=grid_color)
    yaxis_kwargs = dict(gridcolor=grid_color, zerolinecolor=grid_color)
    if x_title:
        xaxis_kwargs["title"] = x_title
    if y_title:
        yaxis_kwargs["title"] = y_title
    if x_range:
        xaxis_kwargs["range"] = x_range

    fig.update_xaxes(**xaxis_kwargs)
    fig.update_yaxes(**yaxis_kwargs)
    return fig


# ============================================
# LOAD MODEL BUNDLE
# ============================================
@st.cache_resource
def load_model_bundle():
    pkl_path = os.path.join(os.path.dirname(__file__), "models", "instrument_classifier_full.pkl")
    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)

    num_classes = bundle["architecture_config"]["num_classes"]
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    return model, bundle


def preprocess_audio(audio_bytes, config):
    sr = config["sample_rate"]
    duration = config["duration_seconds"]
    n_mels = config["n_mels"]
    hop = config["hop_length"]

    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr, mono=True)

    segment_len = sr * duration
    segments = [
        audio[i : i + segment_len]
        for i in range(0, len(audio), segment_len)
        if len(audio[i : i + segment_len]) >= sr
    ]
    if not segments:
        segments = [audio]

    all_tensors = []
    for seg in segments:
        target_len = sr * duration
        if len(seg) < target_len:
            seg = np.pad(seg, (0, target_len - len(seg)))
        else:
            seg = seg[:target_len]

        mel_spec = librosa.feature.melspectrogram(y=seg, sr=sr, n_mels=n_mels, hop_length=hop)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-8)

        target_w = config["target_shape"][1]
        if mel_db.shape[1] < target_w:
            mel_db = np.pad(mel_db, ((0, 0), (0, target_w - mel_db.shape[1])), mode="constant")
        else:
            mel_db = mel_db[:, :target_w]

        spec = np.stack([mel_db, mel_db, mel_db], axis=0)
        tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        all_tensors.append(tensor)

    return all_tensors, audio, sr


@st.cache_data
def generate_mock_telemetry(_class_names, _instrument_names):
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=365, freq="D")
    records = []
    for date in dates:
        n_preds = np.random.randint(5, 25)
        for _ in range(n_preds):
            cls = np.random.choice(_class_names)
            conf = np.clip(np.random.beta(5, 2) * 100, 10, 99.9)
            records.append({
                "date": date,
                "instrument_code": cls,
                "instrument": _instrument_names.get(cls, cls),
                "confidence": round(conf, 2),
                "correct": np.random.random() < 0.84,
            })
    return pd.DataFrame(records)


# ============================================
# MAIN APP
# ============================================
def main():
    # --- Load model ---
    try:
        model, bundle = load_model_bundle()
        model_loaded = True
    except Exception as e:
        model_loaded = False
        bundle = {
            "class_names": list(INSTRUMENT_NAMES.keys()),
            "instrument_names": INSTRUMENT_NAMES,
            "preprocessing_config": {},
            "evaluation_results": {},
        }

    class_names = bundle["class_names"]
    instr_names = bundle.get("instrument_names", INSTRUMENT_NAMES)
    eval_results = bundle.get("evaluation_results", {})
    preproc_config = bundle.get("preprocessing_config", {})
    report = eval_results.get("classification_report", {})

    # ================================
    # SIDEBAR
    # ================================
    with st.sidebar:
        st.markdown("## 🎵 InstruNet Dashboard")
        st.markdown("---")

        st.markdown("#### 🎨 Theme")
        theme = st.radio("Select theme", ["🌙 Dark", "☀️ Light"], horizontal=True, label_visibility="collapsed")
        is_dark = "Dark" in theme
        if is_dark:
            st.markdown("<p style='color:#3fb950; font-size:0.8rem;'>✓ Dark mode active</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#ff6b35; font-size:0.8rem;'>✓ Light mode active</p>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Instruments")
        readable_names = [instr_names.get(c, c) for c in class_names]
        selected_instruments = st.multiselect("Filter instruments", options=readable_names, default=readable_names, label_visibility="collapsed")

        st.markdown("---")
        st.markdown("#### Confidence Threshold")
        threshold = st.slider("Min confidence (%)", min_value=10, max_value=95, value=30, step=5, label_visibility="collapsed")

        st.markdown("---")
        st.markdown("#### Date Range")
        date_range = st.date_input("Select range", value=(pd.to_datetime("2024-01-01"), pd.to_datetime("2024-12-31")), label_visibility="collapsed")

        st.markdown("---")
        st.markdown(
            f"<p style='color:#8b949e; font-size:0.7rem; text-align:center;'>"
            f"Model: {'✅ Loaded' if model_loaded else '❌ Not loaded'}<br>"
            f"Framework: PyTorch<br>Backbone: EfficientNet-B0</p>",
            unsafe_allow_html=True,
        )

    # ================================
    # APPLY THEME CSS
    # ================================
    st.markdown(DARK_CSS if is_dark else LIGHT_CSS, unsafe_allow_html=True)
    text_color = "#e6edf3" if is_dark else "#1f2328"

    # ================================
    # LOAD & FILTER TELEMETRY DATA
    # ================================
    telemetry_df = generate_mock_telemetry(class_names, instr_names)
    name_to_code = {v: k for k, v in instr_names.items()}
    selected_codes = [name_to_code.get(n, n) for n in selected_instruments]
    filtered_df = telemetry_df[telemetry_df["instrument_code"].isin(selected_codes)].copy()

    if isinstance(date_range, tuple) and len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df["date"] >= pd.to_datetime(date_range[0]))
            & (filtered_df["date"] <= pd.to_datetime(date_range[1]))
        ]

    # ================================
    # TOP KPI CARDS
    # ================================
    accuracy = eval_results.get("accuracy", 0.839) * 100
    precision_val = eval_results.get("precision", 0.840) * 100
    recall_val = eval_results.get("recall", 0.839) * 100
    f1_val = eval_results.get("f1_score", 0.838) * 100
    total_predictions = len(filtered_df)
    avg_confidence = filtered_df["confidence"].mean() if len(filtered_df) > 0 else 0
    high_conf = len(filtered_df[filtered_df["confidence"] >= threshold])

    best_class, best_f1 = "", 0
    for cls in class_names:
        cf = report.get(cls, {}).get("f1-score", 0)
        if cf > best_f1:
            best_f1 = cf
            best_class = instr_names.get(cls, cls)

    def metric_card(label, value, sub, sub_class="metric-sub"):
        return f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="{sub_class}">{sub}</div>
        </div>"""

    row1 = st.columns(4)
    row1[0].markdown(metric_card("Overall Accuracy", f"{accuracy:.1f}%", "Weighted average"), unsafe_allow_html=True)
    row1[1].markdown(metric_card("Avg Confidence", f"{avg_confidence:.1f}%", f"Across {total_predictions:,} predictions"), unsafe_allow_html=True)
    row1[2].markdown(metric_card("Precision", f"{precision_val:.1f}%", "Weighted average"), unsafe_allow_html=True)
    row1[3].markdown(metric_card("F1 Score", f"{f1_val:.1f}%", "Harmonic mean"), unsafe_allow_html=True)

    row2 = st.columns(4)
    row2[0].markdown(metric_card("Total Predictions", f"{total_predictions:,}", "Filtered period"), unsafe_allow_html=True)
    row2[1].markdown(metric_card("High Confidence", f"{high_conf:,}", f"≥ {threshold}% confidence"), unsafe_allow_html=True)
    row2[2].markdown(metric_card("Recall", f"{recall_val:.1f}%", "Weighted average"), unsafe_allow_html=True)
    row2[3].markdown(metric_card("Best Performer", best_class, f"F1: {best_f1*100:.1f}%"), unsafe_allow_html=True)

    # ================================
    # SECTION: PREDICTION COMPOSITION
    # ================================
    st.markdown('<div class="section-header">Prediction Composition</div>', unsafe_allow_html=True)
    chart_row1 = st.columns(2)

    # Donut Chart
    with chart_row1[0]:
        if len(filtered_df) > 0:
            dist_df = filtered_df.groupby("instrument").size().reset_index(name="count").sort_values("count", ascending=False)
            fig = go.Figure(go.Pie(
                labels=dist_df["instrument"], values=dist_df["count"], hole=0.55,
                marker=dict(colors=INSTRUMENT_COLORS[:len(dist_df)]),
                textinfo="label+percent", textfont=dict(size=11, color=text_color),
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
            ))
            fig = apply_theme(fig, is_dark, height=420, title="Prediction Distribution")
            fig.update_layout(legend=dict(font=dict(size=10, color=text_color), bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig, use_container_width=True)

    # Per-Class F1 Bar
    with chart_row1[1]:
        if report:
            f1_data = [{"Instrument": instr_names.get(c, c), "F1 Score": report.get(c, {}).get("f1-score", 0) * 100} for c in class_names]
            f1_df = pd.DataFrame(f1_data).sort_values("F1 Score", ascending=True)
            fig = go.Figure(go.Bar(
                y=f1_df["Instrument"], x=f1_df["F1 Score"], orientation="h",
                marker=dict(color=f1_df["F1 Score"], colorscale=[[0, "#f85149"], [0.5, "#ff6b35"], [1, "#3fb950"]], line=dict(width=0)),
                text=f1_df["F1 Score"].round(1).astype(str) + "%", textposition="outside", textfont=dict(size=11, color=text_color),
                hovertemplate="<b>%{y}</b><br>F1: %{x:.1f}%<extra></extra>",
            ))
            fig = apply_theme(fig, is_dark, height=420, title="Per-Class F1 Score", x_title="F1 Score (%)", x_range=[0, 105])
            fig.update_yaxes(title="")
            st.plotly_chart(fig, use_container_width=True)

    # ================================
    # SECTION: CONFIDENCE ANALYSIS
    # ================================
    st.markdown('<div class="section-header">Confidence Analysis</div>', unsafe_allow_html=True)
    chart_row2 = st.columns(2)

    # Histogram
    with chart_row2[0]:
        if len(filtered_df) > 0:
            fig = go.Figure(go.Histogram(
                x=filtered_df["confidence"], nbinsx=40,
                marker=dict(color="#ff6b35", line=dict(color="#ff8c42", width=1)),
                opacity=0.85, hovertemplate="Confidence: %{x:.1f}%<br>Count: %{y}<extra></extra>",
            ))
            fig.add_vline(x=threshold, line=dict(color="#f85149", width=2, dash="dash"),
                          annotation_text=f"Threshold ({threshold}%)", annotation_font=dict(color="#f85149", size=11))
            fig = apply_theme(fig, is_dark, height=400, title="Confidence Score Distribution", x_title="Confidence (%)", y_title="Count", show_legend=False)
            st.plotly_chart(fig, use_container_width=True)

    # Box Plot
    with chart_row2[1]:
        if len(filtered_df) > 0:
            fig = go.Figure()
            for i, instr in enumerate(sorted(filtered_df["instrument"].unique())):
                subset = filtered_df[filtered_df["instrument"] == instr]
                fig.add_trace(go.Box(y=subset["confidence"], name=instr,
                    marker=dict(color=INSTRUMENT_COLORS[i % len(INSTRUMENT_COLORS)]), boxmean=True))
            fig = apply_theme(fig, is_dark, height=400, title="Confidence by Instrument", y_title="Confidence (%)", show_legend=False)
            st.plotly_chart(fig, use_container_width=True)

    # ================================
    # SECTION: USAGE & DEMAND OVER TIME
    # ================================
    st.markdown('<div class="section-header">Usage & Demand Over Time</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        primary_metric = st.selectbox("Primary Metric", ["Prediction Count", "Avg Confidence", "Accuracy Rate"])
    with col2:
        group_by = st.radio("Group by", ["instrument", "none"], horizontal=True)

    if len(filtered_df) > 0:
        monthly = filtered_df.copy()
        monthly["month"] = monthly["date"].dt.to_period("M").dt.to_timestamp()

        if primary_metric == "Prediction Count":
            if group_by == "instrument":
                agg = monthly.groupby(["month", "instrument"]).size().reset_index(name="value")
            else:
                agg = monthly.groupby("month").size().reset_index(name="value")
            y_title = "Prediction Count"
        elif primary_metric == "Avg Confidence":
            if group_by == "instrument":
                agg = monthly.groupby(["month", "instrument"])["confidence"].mean().reset_index(name="value")
            else:
                agg = monthly.groupby("month")["confidence"].mean().reset_index(name="value")
            y_title = "Avg Confidence (%)"
        else:
            if group_by == "instrument":
                agg = monthly.groupby(["month", "instrument"])["correct"].mean().reset_index(name="value")
            else:
                agg = monthly.groupby("month")["correct"].mean().reset_index(name="value")
            agg["value"] = agg["value"] * 100
            y_title = "Accuracy Rate (%)"

        fig = go.Figure()
        if group_by == "instrument" and "instrument" in agg.columns:
            for i, instr in enumerate(sorted(agg["instrument"].unique())):
                subset = agg[agg["instrument"] == instr]
                fig.add_trace(go.Scatter(x=subset["month"], y=subset["value"], mode="lines+markers",
                    name=instr, line=dict(width=2.5, color=INSTRUMENT_COLORS[i % len(INSTRUMENT_COLORS)]), marker=dict(size=5)))
        else:
            fig.add_trace(go.Scatter(x=agg["month"], y=agg["value"], mode="lines+markers",
                name=primary_metric, line=dict(width=3, color="#ff6b35"), marker=dict(size=6),
                fill="tozeroy", fillcolor="rgba(255, 107, 53, 0.1)"))

        fig = apply_theme(fig, is_dark, height=420,
            title=f"Monthly {primary_metric} by {'Instrument' if group_by=='instrument' else 'Total'}",
            x_title="Month", y_title=y_title)
        fig.update_layout(legend=dict(font=dict(size=10, color=text_color)))
        st.plotly_chart(fig, use_container_width=True)

    # ================================
    # SECTION: ROLLING STATISTICS
    # ================================
    st.markdown('<div class="section-header">Rolling Statistics (30-Day)</div>', unsafe_allow_html=True)
    roll_row = st.columns([1.2, 0.8])

    with roll_row[0]:
        if len(filtered_df) > 0:
            daily = filtered_df.groupby("date")["confidence"].mean().reset_index().sort_values("date")
            daily["rolling_mean"] = daily["confidence"].rolling(30, min_periods=1).mean()
            daily["rolling_std"] = daily["confidence"].rolling(30, min_periods=1).std().fillna(0)
            daily["upper"] = daily["rolling_mean"] + daily["rolling_std"]
            daily["lower"] = daily["rolling_mean"] - daily["rolling_std"]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pd.concat([daily["date"], daily["date"][::-1]]),
                y=pd.concat([daily["upper"], daily["lower"][::-1]]),
                fill="toself", fillcolor="rgba(88, 166, 255, 0.1)", line=dict(width=0), name="Confidence Band"))
            fig.add_trace(go.Scatter(x=daily["date"], y=daily["confidence"], mode="lines",
                name="Actual Confidence", line=dict(width=1, color="rgba(127, 127, 127, 0.4)")))
            fig.add_trace(go.Scatter(x=daily["date"], y=daily["rolling_mean"], mode="lines",
                name="30-Day Rolling Mean", line=dict(width=2.5, color="#58a6ff")))

            fig = apply_theme(fig, is_dark, height=420, title="Confidence: Actual vs 30-Day Rolling Mean (±1σ)",
                x_title="Date", y_title="Confidence (%)")
            fig.update_layout(legend=dict(font=dict(size=10, color=text_color)))
            st.plotly_chart(fig, use_container_width=True)

    # Weekly Seasonality
    with roll_row[1]:
        if len(filtered_df) > 0:
            fdf = filtered_df.copy()
            fdf["day_name"] = fdf["date"].dt.day_name()
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            day_agg = fdf.groupby("day_name")["confidence"].mean().reindex(day_order).reset_index()
            day_agg.columns = ["Day", "Avg Confidence"]
            day_colors = ["#f85149", "#ff6b35", "#ffa657", "#58a6ff", "#a5d6ff", "#d2a8ff", "#3fb950"]

            fig = go.Figure(go.Bar(
                x=day_agg["Day"].str[:3], y=day_agg["Avg Confidence"],
                marker=dict(color=day_colors),
                text=day_agg["Avg Confidence"].round(1), textposition="outside",
                textfont=dict(size=10, color=text_color),
                hovertemplate="<b>%{x}</b><br>Avg Confidence: %{y:.1f}%<extra></extra>",
            ))
            baseline = day_agg["Avg Confidence"].mean()
            fig.add_hline(y=baseline, line=dict(color="#f85149", width=1.5, dash="dash"),
                          annotation_text="Baseline", annotation_font=dict(color="#f85149", size=10))
            fig = apply_theme(fig, is_dark, height=420, title="Weekly Seasonality", y_title="Avg Confidence (%)", show_legend=False)
            st.plotly_chart(fig, use_container_width=True)

    # ================================
    # SECTION: DAILY GROWTH & ACTIVITY
    # ================================
    st.markdown('<div class="section-header">Daily Growth & Activity</div>', unsafe_allow_html=True)
    growth_row = st.columns(2)

    with growth_row[0]:
        if len(filtered_df) > 0:
            daily_counts = filtered_df.groupby("date").size().reset_index(name="count").sort_values("date")
            daily_counts["growth"] = daily_counts["count"].pct_change().fillna(0) * 100
            daily_counts["month"] = daily_counts["date"].dt.to_period("M").dt.to_timestamp()
            monthly_growth = daily_counts.groupby("month")["growth"].mean().reset_index()

            fig = go.Figure(go.Bar(x=monthly_growth["month"], y=monthly_growth["growth"],
                marker=dict(color="#58a6ff", line=dict(width=0)),
                hovertemplate="<b>%{x|%b %Y}</b><br>Avg Growth: %{y:.2f}%<extra></extra>"))
            fig = apply_theme(fig, is_dark, height=380, title="Avg Daily Growth Rate (%)", y_title="Growth (%)", show_legend=False)
            st.plotly_chart(fig, use_container_width=True)

    with growth_row[1]:
        if len(filtered_df) > 0:
            heat_data = filtered_df.copy()
            heat_data["month"] = heat_data["date"].dt.strftime("%b")
            month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            heat_agg = heat_data.groupby(["instrument", "month"]).size().reset_index(name="count")
            heat_pivot = heat_agg.pivot(index="instrument", columns="month", values="count").fillna(0)
            existing_months = [m for m in month_order if m in heat_pivot.columns]
            heat_pivot = heat_pivot[existing_months]

            fig = go.Figure(go.Heatmap(
                z=heat_pivot.values, x=heat_pivot.columns, y=heat_pivot.index,
                colorscale=[[0, "#0d1117"], [0.5, "#ff6b35"], [1, "#ff8c42"]],
                hovertemplate="<b>%{y}</b><br>Month: %{x}<br>Count: %{z}<extra></extra>",
            ))
            fig = apply_theme(fig, is_dark, height=380, title="Prediction Heatmap (Monthly)", show_legend=False)
            st.plotly_chart(fig, use_container_width=True)

    # ================================
    # SECTION: LIVE PREDICTION
    # ================================
    st.markdown('<div class="section-header">🎤 Live Instrument Prediction</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload an audio file (WAV, MP3, FLAC)", type=["wav", "mp3", "flac"],
        help="Upload an audio file to get real-time instrument predictions.")

    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        uploaded_file.seek(0)

        # Load raw audio for visualizations
        audio_signal, audio_sr = librosa.load(io.BytesIO(audio_bytes), sr=preproc_config.get("sample_rate", 22050), mono=True)
        uploaded_file.seek(0)
        audio_duration = round(len(audio_signal) / audio_sr, 2)

        # ---- Uploaded Audio Info Card ----
        st.markdown(f"""
            <div class="metric-card" style="margin-bottom:1.5rem; border-radius:16px;">
                <h3 style="margin-top:0;">🎧 Uploaded Audio</h3>
                <div style="display:flex; gap:1.5rem; flex-wrap:wrap; margin-top:1rem;">
                    <div style="flex:1; min-width:180px; background:rgba(255,255,255,0.03); border:1px solid #21262d; border-radius:10px; padding:0.8rem 1rem;">
                        <div style="font-size:0.65rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:#ff6b35;">📁 File</div>
                        <div style="font-size:0.95rem; font-weight:600; color:#e6edf3; margin-top:0.3rem; word-break:break-all;">{uploaded_file.name}</div>
                    </div>
                    <div style="flex:1; min-width:120px; background:rgba(255,255,255,0.03); border:1px solid #21262d; border-radius:10px; padding:0.8rem 1rem;">
                        <div style="font-size:0.65rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:#ff6b35;">⏱ Duration</div>
                        <div style="font-size:1.3rem; font-weight:800; color:#e6edf3; margin-top:0.3rem;">{audio_duration} sec</div>
                    </div>
                    <div style="flex:1; min-width:120px; background:rgba(255,255,255,0.03); border:1px solid #21262d; border-radius:10px; padding:0.8rem 1rem;">
                        <div style="font-size:0.65rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:#ff6b35;">🎚 Sample Rate</div>
                        <div style="font-size:1.3rem; font-weight:800; color:#e6edf3; margin-top:0.3rem;">{audio_sr} Hz</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.audio(uploaded_file, format="audio/wav")
        st.markdown("")

        # ---- Audio Analysis Dashboard: Waveform + Spectrogram ----
        st.markdown('<div class="section-header">📊 Audio Analysis Dashboard</div>', unsafe_allow_html=True)
        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            # Waveform
            t_axis = np.linspace(0, len(audio_signal) / audio_sr, len(audio_signal))
            step = max(1, len(audio_signal) // 6000)
            t_ds, audio_ds = t_axis[::step], audio_signal[::step]

            fig_wave = go.Figure()
            fig_wave.add_trace(go.Scatter(
                x=t_ds, y=audio_ds, mode="lines",
                line=dict(width=1, color=INSTRUMENT_COLORS[0]),
                fill="tozeroy", fillcolor="rgba(88, 166, 255, 0.08)",
                hovertemplate="<b>Time:</b> %{x:.3f}s<br><b>Amplitude:</b> %{y:.4f}<extra></extra>",
            ))
            fig_wave = apply_theme(fig_wave, is_dark, height=320, title="📈 Waveform",
                                   x_title="Time (s)", y_title="Amplitude", show_legend=False)
            st.plotly_chart(fig_wave, use_container_width=True)

        with viz_col2:
            # Mel Spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio_signal, sr=audio_sr, n_mels=128, hop_length=512)
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            times = librosa.times_like(mel_db, sr=audio_sr, hop_length=512)
            freqs = librosa.mel_frequencies(n_mels=128, fmin=0, fmax=audio_sr / 2)

            fig_spec = go.Figure(go.Heatmap(
                z=mel_db, x=times, y=freqs,
                colorscale="Magma",
                colorbar=dict(title=dict(text="dB", font=dict(color="#8b949e")), tickfont=dict(color="#8b949e")),
                hovertemplate="<b>Time:</b> %{x:.2f}s<br><b>Freq:</b> %{y:.0f} Hz<br><b>Power:</b> %{z:.1f} dB<extra></extra>",
            ))
            fig_spec = apply_theme(fig_spec, is_dark, height=320, title="🌈 Mel Spectrogram",
                                   x_title="Time (s)", y_title="Frequency (Hz)", show_legend=False)
            st.plotly_chart(fig_spec, use_container_width=True)

        st.markdown("")

        # ---- Prediction + Timeline ----
        if model_loaded:
            with st.spinner("🔍 Analyzing audio..."):
                uploaded_file.seek(0)
                audio_bytes_pred = uploaded_file.read()
                uploaded_file.seek(0)
                try:
                    tensors, _, _ = preprocess_audio(audio_bytes_pred, preproc_config)
                    all_probs = []
                    device = torch.device("cpu")
                    with torch.no_grad():
                        for t in tensors:
                            output = model(t.to(device))
                            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                            all_probs.append(probs)

                    timeline_data = np.array(all_probs)
                    avg_probs = np.mean(timeline_data, axis=0)

                    predictions = []
                    for i, cls in enumerate(class_names):
                        predictions.append({
                            "Instrument": instr_names.get(cls, cls),
                            "Confidence": round(float(avg_probs[i]) * 100, 2),
                        })
                    pred_df = pd.DataFrame(predictions).sort_values("Confidence", ascending=False)
                    top = pred_df.iloc[0]

                    # ---- Top Prediction Card ----
                    pred_col1, pred_col2 = st.columns(2)

                    with pred_col1:
                        st.markdown(metric_card("Detected Instrument", top["Instrument"], f"{top['Confidence']}% confidence"), unsafe_allow_html=True)

                        st.markdown("")

                        # Prediction bar chart
                        fig_pred = go.Figure(go.Bar(
                            x=pred_df["Confidence"], y=pred_df["Instrument"], orientation="h",
                            marker=dict(color=["#3fb950" if c >= threshold else "#f85149" for c in pred_df["Confidence"]]),
                            text=pred_df["Confidence"].astype(str) + "%", textposition="outside",
                            textfont=dict(size=10, color=text_color),
                        ))
                        fig_pred.add_vline(x=threshold, line=dict(color="#ffa657", width=2, dash="dash"),
                                      annotation_text=f"Threshold ({threshold}%)", annotation_font=dict(color="#ffa657", size=10))
                        fig_pred = apply_theme(fig_pred, is_dark, height=400, title="Prediction Confidence",
                            x_title="Confidence (%)", x_range=[0, 105], show_legend=False)
                        st.plotly_chart(fig_pred, use_container_width=True)

                    with pred_col2:
                        # ---- Present / Not Present Instruments ----
                        present_instr = [p for _, p in pred_df.iterrows() if p["Confidence"] >= threshold]
                        absent_instr = [p for _, p in pred_df.iterrows() if p["Confidence"] < threshold]

                        st.markdown(f"""
                            <div class="metric-card" style="margin-bottom:1rem;">
                                <div class="metric-label">✅ Present Instruments</div>
                                <div class="metric-value" style="font-size:1.4rem;">{len(present_instr)}</div>
                                <div class="metric-sub">Above {threshold}% threshold</div>
                            </div>
                        """, unsafe_allow_html=True)

                        for p in present_instr:
                            conf = p["Confidence"]
                            bar_color = "#3fb950"
                            st.markdown(f"""
                                <div style="background:rgba(63,185,80,0.06); border:1px solid rgba(63,185,80,0.2); border-left:3px solid #3fb950; border-radius:10px; padding:0.7rem 1rem; margin-bottom:0.5rem;">
                                    <div style="display:flex; justify-content:space-between; align-items:center;">
                                        <span style="font-weight:700; color:#e6edf3;">{p['Instrument']}</span>
                                        <span style="background:rgba(63,185,80,0.15); color:#3fb950; padding:2px 10px; border-radius:20px; font-size:0.65rem; font-weight:700; text-transform:uppercase;">Present</span>
                                    </div>
                                    <div style="color:#8b949e; font-size:0.8rem; margin:0.3rem 0 0.4rem;">Confidence: <strong style="color:#e6edf3;">{conf}%</strong></div>
                                    <div style="width:100%; height:5px; background:rgba(63,185,80,0.1); border-radius:3px; overflow:hidden;">
                                        <div style="width:{conf}%; height:100%; background:linear-gradient(90deg, #3fb950, #7ee787); border-radius:3px;"></div>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)

                        if absent_instr:
                            st.markdown(f"""
                                <div class="metric-card" style="margin-top:1rem; margin-bottom:0.5rem;">
                                    <div class="metric-label">❌ Not Present</div>
                                    <div class="metric-value" style="font-size:1.4rem;">{len(absent_instr)}</div>
                                    <div class="metric-sub-red">Below {threshold}% threshold</div>
                                </div>
                            """, unsafe_allow_html=True)
                            for p in absent_instr:
                                conf = p["Confidence"]
                                st.markdown(f"""
                                    <div style="background:rgba(248,81,73,0.04); border:1px solid rgba(248,81,73,0.15); border-left:3px solid #f85149; border-radius:10px; padding:0.7rem 1rem; margin-bottom:0.5rem; opacity:0.7;">
                                        <div style="display:flex; justify-content:space-between; align-items:center;">
                                            <span style="font-weight:700; color:#e6edf3;">{p['Instrument']}</span>
                                            <span style="background:rgba(248,81,73,0.15); color:#f85149; padding:2px 10px; border-radius:20px; font-size:0.65rem; font-weight:700; text-transform:uppercase;">Not Present</span>
                                        </div>
                                        <div style="color:#8b949e; font-size:0.8rem; margin:0.3rem 0 0.4rem;">Confidence: <strong style="color:#e6edf3;">{conf}%</strong></div>
                                        <div style="width:100%; height:5px; background:rgba(248,81,73,0.1); border-radius:3px; overflow:hidden;">
                                            <div style="width:{conf}%; height:100%; background:linear-gradient(90deg, #f85149, #ff7b72); border-radius:3px;"></div>
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)

                    # ---- Instrument Activity Timeline ----
                    if timeline_data.shape[0] > 1:
                        st.markdown('<div class="section-header">⏱ Instrument Activity Timeline</div>', unsafe_allow_html=True)
                        st.markdown(f"<p style='color:#8b949e; font-size:0.85rem;'>Confidence of detected instruments across time segments</p>", unsafe_allow_html=True)

                        segment_duration = preproc_config.get("duration_seconds", 3)
                        readable_names = [instr_names.get(c, c) for c in class_names]
                        n_seg = timeline_data.shape[0]
                        x_time = np.arange(n_seg) * segment_duration

                        fig_tl = go.Figure()
                        for i, (name, color) in enumerate(zip(readable_names, INSTRUMENT_COLORS[:len(readable_names)])):
                            fig_tl.add_trace(go.Scatter(
                                x=x_time, y=timeline_data[:, i] * 100, mode="lines+markers",
                                name=name, line=dict(width=2.5, color=color),
                                marker=dict(size=4),
                                hovertemplate=f"<b>{name}</b><br>Time: %{{x}}s<br>Confidence: %{{y:.1f}}%<extra></extra>",
                            ))
                        fig_tl.add_hline(y=threshold, line=dict(color="#ffa657", width=2, dash="dash"),
                                         annotation_text=f"Threshold ({threshold}%)", annotation_font=dict(color="#ffa657", size=11))
                        fig_tl = apply_theme(fig_tl, is_dark, height=450, title="Instrument Activity Over Time",
                                             x_title="Time (s)", y_title="Confidence (%)")
                        fig_tl.update_layout(legend=dict(font=dict(size=10, color=text_color), bgcolor="rgba(0,0,0,0)"))
                        st.plotly_chart(fig_tl, use_container_width=True)

                except Exception as e:
                    st.error(f"Error during prediction: {e}")
        elif not model_loaded:
            st.warning("⚠️ Model not loaded. Cannot make predictions.")

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#8b949e; font-size:0.75rem;'>"
        "🎵 InstruNet Dashboard • CNN-Based Music Instrument Recognition • EfficientNet-B0 • PyTorch</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
