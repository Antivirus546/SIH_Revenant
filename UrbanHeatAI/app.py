from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
MODEL_DIR = ROOT / "ml-modeling"
SIM_DIR = ROOT / "simulator-optimizer"

sys.path.insert(0, str(SIM_DIR))

from intervention_simulater import (
    load_ward1_grid,
    InterventionSimulator
)
from model_adapter import build_predict_fn

RAW_TO_C = 0.00341802
RAW_OFFSET = 149.0 - 273.15

FEATURES = [
    "bldg_area_sqm",
    "road_length_m",
    "ndvi",
    "ndbi",
    "albedo",
    "ndwi",
    "elevation"
]

FEATURE_LABELS = {
    "bldg_area_sqm": "Building Footprint (m²)",
    "road_length_m": "Road Length (m)",
    "ndvi": "Vegetation (NDVI)",
    "ndbi": "Built-up Index (NDBI)",
    "albedo": "Surface Albedo",
    "ndwi": "Water Index (NDWI)",
    "elevation": "Elevation (m)",
}

# =========================================================
# PAGE CONFIG  (must be the first Streamlit call)
# =========================================================

st.set_page_config(
    page_title="UrbanHeat AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# THEME / STYLING
# =========================================================

ACCENT = "#ff6b35"
INK = "#0b1220"

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;800&family=IBM+Plex+Mono:wght@500&display=swap');

      html, body, [class*="css"], .stApp { font-family: 'Manrope', system-ui, sans-serif; }

      .stApp {
        background:
          radial-gradient(1100px 520px at 12% -12%, rgba(255,107,53,.16), transparent 60%),
          radial-gradient(900px 480px at 92% 0%, rgba(45,212,191,.12), transparent 55%),
          #070c16;
        color: #e8eef7;
      }

      /* hide default chrome */
      #MainMenu, footer, header { visibility: hidden; }
      .block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1400px; }

      h1, h2, h3 { letter-spacing: -.02em; font-weight: 800; color: #f4f8ff; }
      p, li, label, .stMarkdown { color: #b9c6da; }

      /* ---------- sidebar ---------- */
      section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1220 0%, #070c16 100%);
        border-right: 1px solid rgba(255,255,255,.07);
      }
      section[data-testid="stSidebar"] .stRadio > label { display:none; }
      section[data-testid="stSidebar"] div[role="radiogroup"] > label {
        display:flex; align-items:center; gap:.6rem;
        padding:.62rem .8rem; margin-bottom:.28rem;
        border-radius:12px; border:1px solid transparent;
        transition:all .18s ease; cursor:pointer;
      }
      section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background: rgba(255,255,255,.05); border-color: rgba(255,255,255,.08);
      }
      section[data-testid="stSidebar"] div[role="radiogroup"] input:checked + div {
        color: #fff !important; font-weight: 700;
      }

      /* ---------- hero ---------- */
      .hero {
        border:1px solid rgba(255,255,255,.09);
        border-radius:22px; padding:2.1rem 2.2rem;
        background: linear-gradient(135deg, rgba(255,107,53,.14), rgba(45,212,191,.07));
        margin-bottom:1.6rem;
      }
      .hero .eyebrow {
        font-family:'IBM Plex Mono',monospace; font-size:.72rem; letter-spacing:.22em;
        text-transform:uppercase; color:#ffb094;
      }
      .hero h1 { font-size:2.6rem; margin:.35rem 0 .5rem; }
      .hero p  { font-size:1.02rem; max-width:70ch; margin:0; }

      /* ---------- metric cards ---------- */
      div[data-testid="stMetric"] {
        background: rgba(255,255,255,.035);
        border:1px solid rgba(255,255,255,.09);
        border-radius:16px; padding:1.05rem 1.15rem;
        transition: border-color .2s ease, transform .2s ease;
      }
      div[data-testid="stMetric"]:hover { border-color: rgba(255,107,53,.45); transform: translateY(-2px); }
      div[data-testid="stMetricLabel"] p {
        font-size:.72rem !important; letter-spacing:.14em; text-transform:uppercase;
        color:#8fa1ba !important; font-weight:600;
      }
      div[data-testid="stMetricValue"] {
        font-size:1.85rem !important; font-weight:800 !important; color:#fff !important;
      }

      /* ---------- generic surface card ---------- */
      .card {
        background: rgba(255,255,255,.035);
        border:1px solid rgba(255,255,255,.09);
        border-radius:16px; padding:1.25rem 1.35rem; height:100%;
      }
      .card .ico { font-size:1.5rem; }
      .card h4 { margin:.5rem 0 .3rem; font-size:1.02rem; color:#f4f8ff; font-weight:800; }
      .card p  { margin:0; font-size:.9rem; color:#9fb0c7; }

      .section-title { font-size:1.25rem; font-weight:800; margin:.4rem 0 .1rem; color:#f4f8ff; }
      .section-sub { font-size:.86rem; color:#8fa1ba; margin-bottom:.9rem; }

      .pill {
        display:inline-flex; align-items:center; gap:.45rem;
        font-family:'IBM Plex Mono',monospace; font-size:.72rem; letter-spacing:.1em;
        padding:.3rem .7rem; border-radius:999px;
        border:1px solid rgba(45,212,191,.35); color:#7ff0dc; background:rgba(45,212,191,.09);
      }
      .pill.bad { border-color:rgba(255,99,99,.35); color:#ff9a9a; background:rgba(255,99,99,.09); }

      /* ---------- buttons ---------- */
      .stButton > button {
        border-radius:12px; font-weight:700; letter-spacing:.01em;
        border:1px solid rgba(255,255,255,.12);
        background: linear-gradient(135deg, #ff6b35, #f7931e);
        color:#0b1220; padding:.65rem 1rem;
        transition: filter .18s ease, transform .18s ease;
      }
      .stButton > button:hover { filter:brightness(1.08); transform:translateY(-1px); color:#0b1220; }

      /* ---------- inputs ---------- */
      div[data-baseweb="select"] > div, .stTextInput input {
        background: rgba(255,255,255,.04) !important;
        border-color: rgba(255,255,255,.12) !important; border-radius:12px !important;
      }
      .stSlider [data-baseweb="slider"] div[role="slider"] { border-color:#ff6b35 !important; }

      /* ---------- radio (horizontal segmented) ---------- */
      div[role="radiogroup"][aria-orientation="horizontal"] { gap:.4rem; }

      /* ---------- dataframe ---------- */
      div[data-testid="stDataFrame"] {
        border:1px solid rgba(255,255,255,.09); border-radius:14px; overflow:hidden;
      }

      hr { border-color: rgba(255,255,255,.08) !important; }
    </style>
    """,
    unsafe_allow_html=True
)

CHART_BG = "transparent"


def theme_chart(chart, height=380):
    """Consistent dark chart styling."""
    return (
        chart.properties(height=height)
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor="#8fa1ba",
            titleColor="#b9c6da",
            gridColor="rgba(255,255,255,0.06)",
            domainColor="rgba(255,255,255,0.12)",
            labelFontSize=11,
            titleFontSize=12,
            labelFont="Manrope",
            titleFont="Manrope",
        )
        .configure_legend(
            labelColor="#b9c6da", titleColor="#b9c6da",
            labelFont="Manrope", titleFont="Manrope",
        )
        .configure(background=CHART_BG)
    )


def hero(eyebrow, title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
          <div class="eyebrow">{eyebrow}</div>
          <h1>{title}</h1>
          <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def section(title, subtitle=""):
    st.markdown(
        f'<div class="section-title">{title}</div>'
        f'<div class="section-sub">{subtitle}</div>',
        unsafe_allow_html=True
    )


# =========================================================
# DATA
# =========================================================

@st.cache_data
def load_data():
    df = pd.read_csv(
        DATA_DIR / "ward1_processed.csv"
    )

    df["observed_temp_c"] = (
        df["target_temp"] * RAW_TO_C
        + RAW_OFFSET
    )

    return df

# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_model():
    model_path = MODEL_DIR / "heat_model.pkl"
    if model_path.exists():
        return joblib.load(model_path), "saved"

    # Fallback to model_adapter if primary path is missing
    predict_fn, info = build_predict_fn()
    if predict_fn is not None:
        return predict_fn, "adapter"

    raise FileNotFoundError("Authoritative heat_model.pkl not found.")


@st.cache_data
def get_predictions(df):
    model, source = load_model()
    if source == "adapter":
        # model is a predict_fn callable
        preds = model(df[FEATURES].values)
    else:
        # model is a scikit-learn / XGBoost estimator
        preds = model.predict(df[FEATURES])

    return pd.Series(preds, index=df.index)

# =========================================================
# SIMULATOR
# =========================================================

@st.cache_resource
def get_simulator():
    grid = load_ward1_grid(
        DATA_DIR / "ward1_processed.csv"
    )
    predict_fn, _ = build_predict_fn()
    return InterventionSimulator(grid, model_predict_fn=predict_fn, feature_order=FEATURES)

# =========================================================
# HELPERS
# =========================================================

def create_grid(df, value_column):
    grid = df.copy()
    grid["col"] = grid["grid_id"] // 34
    grid["row"] = grid["grid_id"] % 34
    grid["value"] = grid[value_column]
    return grid


def get_target_cells(
    df,
    mode,
    percentage
):
    if mode == "All cells":
        return None

    count = max(
        1,
        int(len(df) * percentage / 100)
    )

    hottest = df.nlargest(
        count,
        "observed_temp_c"
    )

    return [
        f"Ward_1_C{int(i):04d}"
        for i in hottest["grid_id"]
    ]


def run_simulation(
    simulator,
    plan,
    target_cells
):
    return simulator.estimate_impact(
        plan,
        target_cells=target_cells
    )


def format_crore(value):
    return f"₹{value / 1e7:.2f} Cr"


# =========================================================
# LOAD DATA + MODEL
# =========================================================

df = load_data()
try:
    model, model_source = load_model()
except Exception:
    model, model_source = None, "missing"

df["predicted_temp_c"] = get_predictions(df)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding:.4rem 0 1rem">
          <div style="font-size:1.35rem;font-weight:800;color:#fff;letter-spacing:-.02em">
            UrbanHeat <span style="color:#ff8a5c">AI</span>
          </div>
          <div style="font-family:'IBM Plex Mono',monospace;font-size:.68rem;
                      letter-spacing:.2em;text-transform:uppercase;color:#7d8ea6;margin-top:.25rem">
            From heat maps to action
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Heat Map",
            "Heat Drivers",
            "Intervention Simulator",
            "Optimization",
            "Methodology"
        ],
        label_visibility="collapsed"
    )

    st.markdown("<hr/>", unsafe_allow_html=True)

    status = (
        '<span class="pill">● MODEL ONLINE</span>'
        if model_source in ["saved", "adapter"]
        else '<span class="pill bad">● MODEL OFFLINE</span>'
    )
    st.markdown(status, unsafe_allow_html=True)
    st.markdown(
        f"<div style='margin-top:.7rem;font-size:.78rem;color:#7d8ea6'>"
        f"Ward 1 · {len(df):,} grid cells<br/>Source: <code>{model_source}</code></div>",
        unsafe_allow_html=True
    )

# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":
    hero(
        "Space-enabled urban climate intelligence",
        "UrbanHeat AI",
        "Detect urban heat hotspots, explain their drivers, simulate cooling "
        "interventions, and optimize where action should be taken — grounded in "
        "satellite observation and machine learning."
    )

    hotspots = df.nlargest(
        max(1, int(len(df) * 0.10)),
        "observed_temp_c"
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Heat Hotspots", f"{len(hotspots):,}")
    with col2:
        st.metric("Mean Heat", f"{df.observed_temp_c.mean():.2f} °C")
    with col3:
        st.metric("Peak Heat", f"{df.observed_temp_c.max():.2f} °C")
    with col4:
        st.metric(
            "Thermal Range",
            f"{df.observed_temp_c.max() - df.observed_temp_c.min():.2f} °C"
        )

    st.write("")
    section("How UrbanHeat AI works", "Four stages, one continuous decision pipeline.")

    steps = [
        ("", "Detect", "Identify priority urban heat hotspots from satellite thermal data."),
        ("", "Explain", "Attribute local heating to vegetation, built form and surface properties."),
        ("", "Simulate", "Test cooling interventions before a rupee is committed."),
        ("", "Optimize", "Trade impact against cost with multi-objective search."),
    ]
    for col, (ico, title, body) in zip(st.columns(4), steps):
        with col:
            st.markdown(
                f'<div class="card"><div class="ico">{ico}</div>'
                f'<h4>{title}</h4><p>{body}</p></div>',
                unsafe_allow_html=True
            )

    st.write("")
    left, right = st.columns([1.15, 1])

    with left:
        section("Temperature distribution", "Observed land-surface temperature across Ward 1.")
        hist = (
            alt.Chart(df)
            .mark_bar(color=ACCENT, opacity=0.85, cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
            .encode(
                x=alt.X("observed_temp_c:Q", bin=alt.Bin(maxbins=40), title="Observed temperature (°C)"),
                y=alt.Y("count():Q", title="Grid cells"),
                tooltip=[alt.Tooltip("count():Q", title="Cells")]
            )
        )
        st.altair_chart(theme_chart(hist, 320), use_container_width=True)

    with right:
        section("Hottest areas", "Top 10 cells by observed temperature.")
        st.dataframe(
            hotspots[
                [
                    "grid_id",
                    "observed_temp_c",
                    "ndvi",
                    "ndbi",
                    "albedo"
                ]
            ]
            .head(10)
            .round(3)
            .rename(columns={
                "grid_id": "Grid",
                "observed_temp_c": "Temp °C",
                "ndvi": "NDVI",
                "ndbi": "NDBI",
                "albedo": "Albedo",
            }),
            use_container_width=True,
            hide_index=True,
            height=320
        )

# =========================================================
# HEAT MAP
# =========================================================

elif page == "Heat Map":
    hero("Stage 01 — Detect", "Heat Map", "Locate priority heat-stress areas across the ward grid.")

    ctrl, spacer = st.columns([1.4, 2])
    with ctrl:
        layer = st.radio(
            "Heat layer",
            ["Observed Temperature", "Predicted Temperature"],
            horizontal=True
        )

    value_column = "observed_temp_c" if layer == "Observed Temperature" else "predicted_temp_c"
    grid = create_grid(df, value_column)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Layer Mean", f"{grid['value'].mean():.2f} °C")
    with m2:
        st.metric("Layer Peak", f"{grid['value'].max():.2f} °C")
    with m3:
        st.metric("Layer Minimum", f"{grid['value'].min():.2f} °C")

    st.write("")
    heatmap = (
        alt.Chart(grid)
        .mark_rect(stroke=None)
        .encode(
            x=alt.X("col:O", title="Grid Column", axis=alt.Axis(labelAngle=0, labels=False, ticks=False)),
            y=alt.Y("row:O", sort="descending", title="Grid Row",
                    axis=alt.Axis(labels=False, ticks=False)),
            color=alt.Color(
                "value:Q",
                title="Temp °C",
                scale=alt.Scale(scheme="inferno"),
                legend=alt.Legend(orient="right", gradientLength=280)
            ),
            tooltip=[
                alt.Tooltip("grid_id:Q", title="Grid ID"),
                alt.Tooltip("value:Q", title="Temperature °C", format=".2f"),
                alt.Tooltip("ndvi:Q", title="NDVI", format=".3f"),
                alt.Tooltip("ndbi:Q", title="NDBI", format=".3f")
            ]
        )
    )
    st.altair_chart(theme_chart(heatmap, 620), use_container_width=True)
    st.caption("Brighter cells indicate higher land-surface temperature. Hover any cell for detail.")

# =========================================================
# HEAT DRIVERS
# =========================================================

elif page == "Heat Drivers":
    hero("Stage 02 — Explain", "Heat Drivers", "Understand what makes a specific place hot.")

    if model_source == "saved":
        section("Global feature importance", "Learned contribution of each predictor across the ward.")
        importance = pd.DataFrame(
            {
                "Feature": FEATURES,
                "Importance": model.feature_importances_
            }
        ).sort_values("Importance", ascending=False)
        importance["Label"] = importance["Feature"].map(FEATURE_LABELS)

        chart = (
            alt.Chart(importance)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("Importance:Q", title="Model importance"),
                # Set labelLimit to 300 to prevent label truncation
                y=alt.Y("Label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=300)),
                color=alt.Color(
                    "Importance:Q", 
                    scale=alt.Scale(range=["#f7931e", "#f7931e"]), 
                    legend=None
                ),
                tooltip=["Label:N", alt.Tooltip("Importance:Q", format=".4f")]
            )
        )
        
        st.altair_chart(theme_chart(chart, 330), use_container_width=True)
        st.markdown("<hr/>", unsafe_allow_html=True)

    section("Local driver analysis", "Sensitivity of the prediction to each feature at one grid cell.")

    selected_grid = st.selectbox(
        "Grid cell",
        df["grid_id"].astype(int).tolist(),
        index=int(df["observed_temp_c"].idxmax())
    )

    row = df[df.grid_id == selected_grid].iloc[0]
    base_input = row[FEATURES].to_frame().T

    def predict_single(X_df):
        if model_source == "adapter":
            return model(X_df.values)[0]
        else:
            return model.predict(X_df)[0]

    baseline_prediction = float(predict_single(base_input))
    effects = []

    for feature in FEATURES:
        modified = base_input.copy()
        current = float(modified.iloc[0][feature])

        if feature in ["bldg_area_sqm", "road_length_m"]:
            change = max(abs(current) * 0.10, 1)
        elif feature == "elevation":
            change = 20
        else:
            change = 0.05

        modified.loc[modified.index[0], feature] = current + change
        new_prediction = float(predict_single(modified))

        effects.append({
            "Feature": feature,
            "Current Value": current,
            "Prediction Change (°C)": new_prediction - baseline_prediction,
            "Absolute Effect": abs(new_prediction - baseline_prediction)
        })

    local = pd.DataFrame(effects).sort_values("Absolute Effect", ascending=False)
    local["Label"] = local["Feature"].map(FEATURE_LABELS)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Grid ID", int(selected_grid))
    with col2:
        st.metric("Observed", f"{row.observed_temp_c:.2f} °C")
    with col3:
        st.metric(
            "Model Prediction",
            f"{baseline_prediction:.2f} °C",
            delta=f"{baseline_prediction - row.observed_temp_c:+.2f} °C vs observed"
        )

    st.write("")
    chart_col, table_col = st.columns([1.1, 1])

    with chart_col:
        tornado = (
            alt.Chart(local)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("Prediction Change (°C):Q", title="Δ prediction (°C)"),
                y=alt.Y("Label:N", sort="-x", title=None),
                color=alt.condition(
                    alt.datum["Prediction Change (°C)"] > 0,
                    alt.value("#ff6b35"),
                    alt.value("#2dd4a8")
                ),
                tooltip=["Label:N", alt.Tooltip("Prediction Change (°C):Q", format=".4f")]
            )
        )
        st.altair_chart(theme_chart(tornado, 330), use_container_width=True)

    with table_col:
        st.dataframe(
            local[["Label", "Current Value", "Prediction Change (°C)"]]
            .round(4)
            .rename(columns={"Label": "Feature"}),
            use_container_width=True,
            hide_index=True,
            height=330
        )

    st.caption(
        "Warm bars increase predicted temperature when the feature rises; "
        "cool bars reduce it. Values are local sensitivity tests, not causal effects."
    )

    shap_file = MODEL_DIR / "shap_summary.png"
    if shap_file.exists():
        st.markdown("<hr/>", unsafe_allow_html=True)
        section("SHAP summary", "Global explanation exported from the training pipeline.")
        st.image(str(shap_file), use_container_width=True)

# =========================================================
# INTERVENTION SIMULATOR
# =========================================================

elif page == "Intervention Simulator":
    hero("Stage 03 — Simulate", "Intervention Simulator", "Test cooling strategies before committing budget.")

    simulator = get_simulator()

    cfg, plan_col = st.columns([1, 1.25])

    with cfg:
        section("Target area", "Where the interventions are applied.")
        target_mode = st.radio(
            "Apply interventions to",
            ["All cells", "Top hotspot cells"],
            horizontal=True,
            label_visibility="collapsed"
        )

        hotspot_percentage = 25
        if target_mode == "Top hotspot cells":
            hotspot_percentage = st.slider("Hotspot coverage (%)", 5, 100, 25, 5)

        target_cells = get_target_cells(df, target_mode, hotspot_percentage)
        st.markdown(
            f"<div style='font-size:.82rem;color:#7d8ea6'>Cells targeted: "
            f"<b style='color:#e8eef7'>{len(df) if target_cells is None else len(target_cells):,}</b></div>",
            unsafe_allow_html=True
        )

    with plan_col:
        section("Intervention scenario", "Dial in the mix of cooling measures.")
        c1, c2 = st.columns(2)
        with c1:
            green_cover = st.slider("Green Cover Increase (%)", 0, 100, 25, 5)
            cool_roofs = st.slider("Cool Roof Coverage (%)", 0, 100, 50, 5)
        with c2:
            albedo = st.slider("Albedo Improvement", 0.0, 1.0, 0.25, 0.05)
            water_bodies = st.slider("Water Body Increase (%)", 0, 100, 10, 5)

    plan = {
        "green_cover": green_cover / 100,
        "cool_roofs": cool_roofs / 100,
        "albedo_boost": albedo,
        "water_bodies": water_bodies / 100
    }

    st.write("")
    if st.button("Run Simulation", type="primary", use_container_width=True):
        with st.spinner("Running intervention simulation…"):
            result = run_simulation(simulator, plan, target_cells)
        st.session_state["simulation_result"] = result
        st.session_state["simulation_plan"] = plan

    result = st.session_state.get("simulation_result")

    if result is not None:
        st.markdown("<hr/>", unsafe_allow_html=True)
        section("Simulation results", "Projected cooling and cost for the configured plan.")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Cooling", f"{result['mean_cooling']:.3f} °C")
        with col2:
            st.metric("Population Benefit", f"{result['pop_weighted_cooling']:.3f} °C")
        with col3:
            st.metric("Maximum Cooling", f"{result['max_cooling']:.3f} °C")
        with col4:
            st.metric("Estimated Cost", format_crore(result["total_cost_inr"]))

        cooling_df = pd.DataFrame({
            "grid_id": df["grid_id"],
            "cooling_c": result["delta_per_cell"]
        })

        st.write("")
        cooling_chart = (
            alt.Chart(cooling_df)
            .mark_area(
                line={"color": "#2dd4a8"},
                color=alt.Gradient(
                    gradient="linear",
                    stops=[
                        alt.GradientStop(color="rgba(45,212,168,0.45)", offset=0),
                        alt.GradientStop(color="rgba(45,212,168,0.02)", offset=1),
                    ],
                    x1=1, x2=1, y1=1, y2=0,
                ),
            )
            .encode(
                x=alt.X("grid_id:Q", title="Grid ID"),
                y=alt.Y("cooling_c:Q", title="Cooling (°C)"),
                tooltip=[
                    alt.Tooltip("grid_id:Q", title="Grid"),
                    alt.Tooltip("cooling_c:Q", title="Cooling °C", format=".3f")
                ]
            )
        )
        st.altair_chart(theme_chart(cooling_chart, 340), use_container_width=True)

# =========================================================
# OPTIMIZATION
# =========================================================

elif page == "Optimization":
    hero("Stage 04 — Optimize", "Optimization", "Search the trade-off frontier between cooling impact and cost.")

    simulator = get_simulator()

    left, right = st.columns([1, 1])

    with left:
        section("Target & budget", "Constrain the search space.")
        target_mode = st.radio(
            "Optimization target",
            ["All cells", "Top hotspot cells"],
            horizontal=True,
            label_visibility="collapsed"
        )

        hotspot_percentage = 25
        if target_mode == "Top hotspot cells":
            hotspot_percentage = st.slider("Hotspot coverage (%)", 5, 100, 25, 5)

        target_cells = get_target_cells(df, target_mode, hotspot_percentage)
        budget = st.slider("Maximum Budget (₹ Cr)", 10, 500, 200, 10)

    with right:
        section("Search settings", "Higher values improve quality and cost runtime.")
        pop_size = st.slider("Population Size", 10, 50, 20, 10)
        n_gen = st.slider("Generations", 10, 50, 20, 10)

    st.write("")
    if st.button("Run AI Optimization", type="primary", use_container_width=True):
        with st.spinner("Running NSGA-II evolutionary search… this may take a minute."):
            from optimizer import NSGA2Optimizer

            opt = NSGA2Optimizer(
                simulator=simulator,
                target_cells=target_cells,
                pop_size=pop_size,
                n_generations=n_gen,
                seed=42
            )

            opt_result = opt.optimize()
            pareto_solutions = opt_result.pareto_solutions

            results = []
            for sol in pareto_solutions:
                plan_data = sol.get("plan", {})

                results.append({
                    "cool_roofs": plan_data.get("cool_roofs", 0.0),
                    "green_cover": plan_data.get("green_cover", 0.0),
                    "albedo_boost": plan_data.get("albedo_boost", 0.0),
                    "water_bodies": plan_data.get("water_bodies", 0.0),
                    "cooling_c": sol.get("mean_cooling", 0.0),
                    "population_cooling_c": sol.get("cooling", 0.0),
                    "cost_cr": sol.get("cost", 0.0) / 1e7
                })

            st.session_state["optimization_results"] = pd.DataFrame(results)
        st.success(f"Optimization complete — {len(results)} Pareto-optimal plans found.")

    results_df = st.session_state.get("optimization_results")

    if results_df is not None and not results_df.empty:
        st.markdown("<hr/>", unsafe_allow_html=True)

        feasible = results_df[results_df["cost_cr"] <= budget].copy()

        if feasible.empty:
            st.warning("No solutions fit this budget. Try increasing the maximum budget.")
        else:
            feasible["score"] = feasible["population_cooling_c"] / feasible["cost_cr"].clip(lower=0.01)

            recommended = feasible.sort_values(
                ["population_cooling_c", "score"], ascending=False
            ).iloc[0]

            section("Recommended plan", f"Best cooling available within ₹{budget} Cr.")

            met1, met2, met3 = st.columns(3)
            with met1:
                st.metric("Cooling Impact", f"{recommended['population_cooling_c']:.3f} °C")
            with met2:
                st.metric("Estimated Cost", f"₹{recommended['cost_cr']:.2f} Cr")
            with met3:
                st.metric("Cooling per Crore", f"{recommended['score']:.4f}")

            st.write("")
            section("Intervention setup", "The mix that produces the recommended outcome.")
            plan_cols = st.columns(4)
            plan_cols[0].metric("Cool Roofs", f"{recommended['cool_roofs'] * 100:.0f}%")
            plan_cols[1].metric("Green Cover", f"{recommended['green_cover'] * 100:.0f}%")
            plan_cols[2].metric("Albedo Boost", f"{recommended['albedo_boost'] * 100:.0f}%")
            plan_cols[3].metric("Water Bodies", f"{recommended['water_bodies'] * 100:.0f}%")

            st.write("")
            section("Pareto frontier", "Every non-dominated plan; green points fit the budget.")
            scatter_df = results_df.copy()
            scatter_df["within_budget"] = np.where(
                scatter_df["cost_cr"] <= budget, "Within budget", "Over budget"
            )

            scatter = (
                alt.Chart(scatter_df)
                .mark_circle(size=110, opacity=0.85)
                .encode(
                    x=alt.X("cost_cr:Q", title="Cost (₹ Cr)"),
                    y=alt.Y("population_cooling_c:Q", title="Population cooling (°C)"),
                    color=alt.Color(
                        "within_budget:N",
                        title=None,
                        scale=alt.Scale(
                            domain=["Within budget", "Over budget"],
                            range=["#2dd4a8", "#6b7a90"]
                        )
                    ),
                    tooltip=[
                        alt.Tooltip("cost_cr:Q", title="Cost ₹Cr", format=".2f"),
                        alt.Tooltip("population_cooling_c:Q", title="Cooling °C", format=".3f"),
                        alt.Tooltip("green_cover:Q", title="Green", format=".2f"),
                        alt.Tooltip("cool_roofs:Q", title="Cool roofs", format=".2f"),
                    ]
                )
            )
            st.altair_chart(theme_chart(scatter, 380), use_container_width=True)

            with st.expander("All Pareto-optimal plans"):
                st.dataframe(
                    results_df.round(4),
                    use_container_width=True,
                    hide_index=True
                )

# =========================================================
# METHODOLOGY
# =========================================================

elif page == "Methodology":
    hero("Reference", "Methodology", "How data, models and optimization fit together.")

    stages = [
        ("", "Data harmonization",
         "Satellite thermal bands, spectral indices (NDVI, NDBI, NDWI, albedo), "
         "elevation and urban-form metrics are aligned onto a common ward grid."),
        ("", "Heat model",
         "A gradient-boosted XGBoost regressor learns land-surface temperature "
         "from urban-form and surface-property predictors."),
        ("", "Intervention simulation",
         "Cooling plans perturb the feature space (vegetation, roof reflectivity, "
         "albedo, water) and the model re-scores each grid cell."),
        ("", "Multi-objective optimization",
         "NSGA-II searches plan space for the Pareto frontier balancing "
         "population-weighted cooling against capital cost."),
    ]

    for col, (ico, title, body) in zip(st.columns(2), stages[:2]):
        with col:
            st.markdown(
                f'<div class="card"><div class="ico">{ico}</div>'
                f'<h4>{title}</h4><p>{body}</p></div>',
                unsafe_allow_html=True
            )
    st.write("")
    for col, (ico, title, body) in zip(st.columns(2), stages[2:]):
        with col:
            st.markdown(
                f'<div class="card"><div class="ico">{ico}</div>'
                f'<h4>{title}</h4><p>{body}</p></div>',
                unsafe_allow_html=True
            )

    st.markdown("<hr/>", unsafe_allow_html=True)
    section("Temperature conversion", "Raw thermal digital numbers to degrees Celsius.")
    st.code(
        f"observed_temp_c = target_temp * {RAW_TO_C} + ({RAW_OFFSET:.2f})",
        language="python"
    )

    section("Model features", "Predictors used by the heat model.")
    st.dataframe(
        pd.DataFrame(
            {"Feature": FEATURES, "Description": [FEATURE_LABELS[f] for f in FEATURES]}
        ),
        use_container_width=True,
        hide_index=True
    )