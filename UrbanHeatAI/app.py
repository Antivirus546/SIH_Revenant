import streamlit as st

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="UrbanHeat AI",
    page_icon="🌡️",
    layout="wide"
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🌡️ UrbanHeat AI")
st.sidebar.caption("From Heat Maps to Action")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "🌡️ Heat Map",
        "🔍 Heat Drivers",
        "🌳 Intervention Simulator",
        "🎯 Optimization",
        "📊 Methodology"
    ]
)

# -----------------------------
# Overview
# -----------------------------
if page == "🏠 Overview":

    st.title("🌡️ UrbanHeat AI")
    st.subheader("From Heat Maps to Action")

    st.write(
        "A space-enabled AI platform for detecting urban heat hotspots, "
        "explaining their drivers, simulating cooling interventions, "
        "and optimizing where action should be taken."
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Heat Hotspots", "—")

    with col2:
        st.metric("Heat Stress", "—")

    with col3:
        st.metric("Scenarios", "4")

    with col4:
        st.metric("Optimization", "Ready")

    st.divider()

    st.subheader("How UrbanHeat AI Works")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("### 🌡️ Detect")
        st.write("Identify priority urban heat hotspots.")

    with c2:
        st.markdown("### 🔍 Explain")
        st.write("Understand the factors driving local heating.")

    with c3:
        st.markdown("### 🌳 Simulate")
        st.write("Test different cooling interventions.")

    with c4:
        st.markdown("### 🎯 Optimize")
        st.write("Find the best strategy based on impact, cost and reach.")


# -----------------------------
# Heat Map / Detect
# -----------------------------
elif page == "🌡️ Heat Map":

    st.title("🌡️ Heat Map")
    st.caption("Detect — Identify priority heat-stress areas")

    st.info(
        "The interactive Bengaluru heat map will appear here."
    )

    # Folium map will be added here later.


# -----------------------------
# Heat Drivers / Explain
# -----------------------------
elif page == "🔍 Heat Drivers":

    st.title("🔍 Heat Drivers")
    st.caption("Explain — Understand why an area is hot")

    st.info(
        "SHAP-based driver attribution will appear here."
    )

    # SHAP chart will be added here later.


# -----------------------------
# Intervention Simulator
# -----------------------------
elif page == "🌳 Intervention Simulator":

    st.title("🌳 Intervention Simulator")
    st.caption("Simulate — Test cooling strategies")

    st.subheader("Select Intervention Scenario")

    green_cover = st.slider(
        "🌳 Green Cover Increase (%)",
        0,
        50,
        10
    )

    cool_roofs = st.slider(
        "🏠 Cool Roof Coverage (%)",
        0,
        50,
        10
    )

    albedo = st.slider(
        "☀️ Albedo Improvement",
        0.0,
        1.0,
        0.1
    )

    water_bodies = st.slider(
        "💧 Water Body Increase (%)",
        0,
        20,
        5
    )

    if st.button("Run Simulation", type="primary"):

        st.success("Scenario submitted for simulation.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Estimated Heat Change", "—")

        with col2:
            st.metric("Population Benefit", "—")

        with col3:
            st.metric("Estimated Cost", "—")


# -----------------------------
# Optimization
# -----------------------------
elif page == "🎯 Optimization":

    st.title("🎯 Optimization")
    st.caption("Optimize — Select the best cooling strategy")

    st.info(
        "Pareto-based optimization results will appear here."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Cooling Impact", "—")

    with col2:
        st.metric("Estimated Cost", "—")

    with col3:
        st.metric("Population Reach", "—")

    st.subheader("Recommended Plan")

    st.write(
        "The optimized intervention plan will be displayed here."
    )


# -----------------------------
# Methodology
# -----------------------------
elif page == "📊 Methodology":

    st.title("📊 Methodology")
    st.caption("About the UrbanHeat AI system")

    st.subheader("End-to-End Pipeline")

    st.markdown("""
    **1. Acquire**

    Satellite, weather and urban-form data are collected.

    **2. Harmonize**

    Data are spatially and temporally aligned and transformed into model features.

    **3. Model**

    Machine learning predicts LST / heat stress.

    **4. Explain**

    SHAP identifies the major drivers of local heating.

    **5. Simulate**

    Cooling interventions such as green cover, cool roofs, albedo
    and water bodies are tested.

    **6. Optimize**

    Cooling impact, cost and population benefit are considered to
    identify effective intervention strategies.
    """)

    st.divider()

    st.subheader("Technology Stack")

    st.write(
        "Python • Google Earth Engine • GeoPandas • Rasterio • "
        "XGBoost / Random Forest • SHAP • NSGA-II / pymoo • "
        "Streamlit • Folium"
    )