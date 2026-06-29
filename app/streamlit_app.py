"""
streamlit_app.py
----------------
Rapido — Intelligent Mobility Insights Dashboard
Streamlit multi-page application.

Tabs:
  1. 📊 Overview & EDA
  2. 🔮 Predictions
  3. 🗺  Demand & Locations
  4. 📈 Model Performance
  5. 🔍 Data Explorer

Run: streamlit run app/streamlit_app.py
"""

import os
import sys
import pickle
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ── path setup ────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.data_loader import load_all
from utils.preprocessing import build_master_dataset, clean_bookings
from utils.eda_utils import (
    plot_ride_volume_by_hour, plot_ride_volume_by_weekday, plot_ride_volume_by_city,
    plot_cancellation_heatmap, plot_distance_vs_fare, plot_rating_distribution,
    plot_booking_status_breakdown, plot_payment_method, plot_traffic_weather_cancellation,
    plot_surge_behavior, plot_cancellations_by_hour, plot_customer_vs_driver_cancel,
    plot_feature_importance, plot_demand_heatmap,
)
from utils.model_training import (
    MODEL_DIR, OUTCOME_FEATURES, FARE_FEATURES, CANCEL_RISK_FEATURES, DRIVER_DELAY_FEATURES
)
from utils.sql_manager import (
    build_database, run_query,
    query_ride_volume_by_hour, query_cancellation_by_city,
    query_fare_by_vehicle, query_peak_cancellation_windows,
    query_high_risk_customers, query_driver_reliability,
    query_surge_demand_pattern, query_traffic_weather_impact,
    query_booking_summary,
)

# ── Streamlit config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rapido Intelligent Mobility Insights",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {font-size:2rem; font-weight:700; color:#FF6B35;}
    .metric-card {background:#1e1e2e; padding:1rem; border-radius:10px; border-left:4px solid #FF6B35;}
    .stTabs [data-baseweb="tab-list"] {gap:8px;}
    .stTabs [data-baseweb="tab"] {border-radius:6px 6px 0 0; padding:8px 16px;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Data loading (cached) ─────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading datasets…")
def get_data():
    data = load_all()
    master = build_master_dataset(
        data["bookings"], data["customers"], data["drivers"], data["location_demand"]
    )
    return data, master


@st.cache_resource(show_spinner="Loading models…")
def get_models():
    models = {}
    model_names = [
        "ride_outcome_model", "fare_prediction_model",
        "customer_cancel_model", "driver_delay_model",
        "ride_outcome_feature_importance", "fare_feature_importance",
        "customer_cancel_feature_importance", "driver_delay_feature_importance",
    ]
    for name in model_names:
        path = os.path.join(MODEL_DIR, f"{name}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
    return models


# ── Load ──────────────────────────────────────────────────────────────────
data, master = get_data()
models = get_models()

bookings = data["bookings"]
customers = data["customers"]
drivers = data["drivers"]
location_demand = data["location_demand"]
time_features = data["time_features_csv"]

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛵 Rapido Insights")
    st.markdown("---")

    cities = ["All"] + sorted(bookings["city"].unique().tolist())
    sel_city = st.selectbox("🌆 Filter by City", cities)

    vehicle_types = ["All"] + sorted(bookings["vehicle_type"].unique().tolist())
    sel_vehicle = st.selectbox("🚗 Filter by Vehicle Type", vehicle_types)

    date_range = st.date_input(
        "📅 Date Range",
        value=(pd.to_datetime("2025-01-01"), pd.to_datetime("2025-12-31")),
    )

    st.markdown("---")
    st.markdown("**Dataset Stats**")
    st.markdown(f"- **Bookings:** {len(bookings):,}")
    st.markdown(f"- **Customers:** {len(customers):,}")
    st.markdown(f"- **Drivers:** {len(drivers):,}")
    st.markdown(f"- **Models:** {'✅ Loaded' if models else '⚠️ Not trained'}")
    st.markdown("---")
    if st.button("🔄 Retrain Models", use_container_width=True):
        st.info("Run `python train.py` from the project root to retrain models.")

# ── Apply filters ─────────────────────────────────────────────────────────
filtered = bookings.copy()
if sel_city != "All":
    filtered = filtered[filtered["city"] == sel_city]
if sel_vehicle != "All":
    filtered = filtered[filtered["vehicle_type"] == sel_vehicle]


# ==========================================================================
# TABS
# ==========================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview & EDA",
    "🔮 Predictions",
    "🗺 Demand & Locations",
    "📈 Model Performance",
    "🗄 SQL Analytics",
    "🔍 Data Explorer",
])


# --------------------------------------------------------------------------
# TAB 1 — Overview & EDA
# --------------------------------------------------------------------------
with tab1:
    st.markdown('<p class="main-header">📊 Rapido — Ride Intelligence Dashboard</p>', unsafe_allow_html=True)

    # KPI row
    total = len(filtered)
    completed = (filtered["booking_status"] == "Completed").sum()
    cancelled = (filtered["booking_status"] == "Cancelled").sum()
    incomplete = (filtered["booking_status"] == "Incomplete").sum()
    avg_fare = filtered[filtered["booking_status"] == "Completed"]["booking_value"].mean()
    avg_dist = filtered["ride_distance_km"].mean()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Bookings", f"{total:,}")
    c2.metric("Completed", f"{completed:,}", f"{completed/total*100:.1f}%" if total else "")
    c3.metric("Cancelled", f"{cancelled:,}", f"-{cancelled/total*100:.1f}%" if total else "")
    c4.metric("Incomplete", f"{incomplete:,}")
    c5.metric("Avg Fare (₹)", f"₹{avg_fare:.0f}")
    c6.metric("Avg Distance", f"{avg_dist:.1f} km")

    st.markdown("---")

    # Row 1
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_booking_status_breakdown(filtered), use_container_width=True)
    with col2:
        st.plotly_chart(plot_ride_volume_by_city(filtered), use_container_width=True)

    # Row 2
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_ride_volume_by_hour(filtered), use_container_width=True)
    with col2:
        st.plotly_chart(plot_ride_volume_by_weekday(filtered), use_container_width=True)

    # Row 3
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_cancellation_heatmap(filtered), use_container_width=True)
    with col2:
        st.plotly_chart(plot_cancellations_by_hour(filtered), use_container_width=True)

    # Row 4
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_distance_vs_fare(filtered), use_container_width=True)
    with col2:
        st.plotly_chart(plot_traffic_weather_cancellation(filtered), use_container_width=True)

    # Row 5
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_rating_distribution(filtered, customers, drivers), use_container_width=True)
    with col2:
        st.plotly_chart(plot_surge_behavior(filtered), use_container_width=True)

    # Row 6
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_customer_vs_driver_cancel(filtered), use_container_width=True)
    with col2:
        st.plotly_chart(plot_payment_method(filtered), use_container_width=True)


# --------------------------------------------------------------------------
# TAB 2 — Predictions
# --------------------------------------------------------------------------
with tab2:
    st.markdown("## 🔮 Real-Time Predictions")

    if not models:
        st.warning("⚠️ No trained models found. Please run `python train.py` first.")
    else:
        pred_tab1, pred_tab2, pred_tab3, pred_tab4 = st.tabs([
            "🚕 Ride Outcome", "💰 Fare Estimator", "👤 Customer Risk", "🚗 Driver Delay"
        ])

        # ── Ride Outcome ──────────────────────────────────────────────────
        with pred_tab1:
            st.markdown("### Predict Booking Outcome")
            col1, col2, col3 = st.columns(3)
            with col1:
                p_city = st.selectbox("City", sorted(bookings["city"].unique()), key="ro_city")
                p_vehicle = st.selectbox("Vehicle Type", sorted(bookings["vehicle_type"].unique()), key="ro_veh")
                p_distance = st.slider("Distance (km)", 1.0, 50.0, 8.0, key="ro_dist")
                p_hour = st.slider("Hour of Day", 0, 23, 9, key="ro_hour")
            with col2:
                p_traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High"], key="ro_traf")
                p_weather = st.selectbox("Weather", sorted(bookings["weather_condition"].unique()), key="ro_wea")
                p_surge = st.slider("Surge Multiplier", 1.0, 3.0, 1.2, key="ro_surge")
                p_base_fare = st.number_input("Base Fare (₹)", 50.0, 1000.0, 150.0, key="ro_fare")
            with col3:
                p_driver_rel = st.slider("Driver Reliability Score", 0.0, 10.0, 7.0, key="ro_drel")
                p_cust_loyal = st.slider("Customer Loyalty Score", 0.0, 10.0, 6.0, key="ro_cloy")
                p_est_time = st.slider("Estimated Ride Time (min)", 5.0, 120.0, 30.0, key="ro_time")
                p_cancel_rate = st.slider("Customer Cancel Rate", 0.0, 1.0, 0.2, key="ro_cr")

            if st.button("🔮 Predict Outcome", key="btn_ro"):
                le_city = {c: i for i, c in enumerate(sorted(bookings["city"].unique()))}
                le_veh = {v: i for i, v in enumerate(sorted(bookings["vehicle_type"].unique()))}
                le_traf = {"Low": 0, "Medium": 1, "High": 2}
                le_wea = {w: i for i, w in enumerate(sorted(bookings["weather_condition"].unique()))}

                bv = p_base_fare * p_surge
                row = {
                    "ride_distance_km": p_distance, "estimated_ride_time_min": p_est_time,
                    "actual_ride_time_min": p_est_time * 1.1,
                    "base_fare": p_base_fare, "surge_multiplier": p_surge,
                    "booking_value": bv, "is_weekend": 0, "hour_of_day": p_hour,
                    "Rush_Hour_Flag": 1 if p_hour in [7,8,9,17,18,19] else 0,
                    "Long_Distance_Flag": 1 if p_distance >= 15 else 0,
                    "Fare_per_KM": bv / max(p_distance, 0.1),
                    "Fare_per_Min": bv / max(p_est_time, 0.1),
                    "Driver_Reliability_Score": p_driver_rel,
                    "Customer_Loyalty_Score": p_cust_loyal,
                    "city_enc": le_city.get(p_city, 0),
                    "vehicle_type_enc": le_veh.get(p_vehicle, 0),
                    "traffic_level_enc": le_traf.get(p_traffic, 1),
                    "weather_condition_enc": le_wea.get(p_weather, 0),
                    "acceptance_rate": 0.8, "delay_rate": 0.1,
                    "cancellation_rate": p_cancel_rate,
                    "avg_demand_requests": 5, "avg_wait_time": 10, "avg_surge": p_surge,
                }
                X = pd.DataFrame([row])[OUTCOME_FEATURES]
                model = models.get("ride_outcome_model")
                if model:
                    pred = model.predict(X)[0]
                    prob = model.predict_proba(X)[0]
                    label_map = {0: "✅ Completed", 1: "❌ Cancelled", 2: "⚠️ Incomplete"}
                    color_map = {0: "success", 1: "error", 2: "warning"}
                    outcome = label_map[pred]
                    st.markdown(f"### Predicted Outcome: **{outcome}**")

                    fig = go.Figure(go.Bar(
                        x=["Completed", "Cancelled", "Incomplete"], y=prob,
                        marker_color=["#00CC96", "#EF553B", "#FFA15A"]
                    ))
                    fig.update_layout(title="Prediction Probabilities", yaxis_title="Probability")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Ride outcome model not found. Run train.py first.")

        # ── Fare Estimator ────────────────────────────────────────────────
        with pred_tab2:
            st.markdown("### Estimate Ride Fare")
            col1, col2 = st.columns(2)
            with col1:
                f_city = st.selectbox("City", sorted(bookings["city"].unique()), key="fare_city")
                f_vehicle = st.selectbox("Vehicle Type", sorted(bookings["vehicle_type"].unique()), key="fare_veh")
                f_distance = st.slider("Distance (km)", 1.0, 50.0, 10.0, key="fare_dist")
                f_time = st.slider("Estimated Time (min)", 5.0, 120.0, 35.0, key="fare_time")
            with col2:
                f_surge = st.slider("Surge Multiplier", 1.0, 3.0, 1.5, key="fare_surge")
                f_traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High"], key="fare_traf")
                f_weather = st.selectbox("Weather", sorted(bookings["weather_condition"].unique()), key="fare_wea")
                f_hour = st.slider("Hour of Day", 0, 23, 10, key="fare_hour")

            if st.button("💰 Estimate Fare", key="btn_fare"):
                le_city = {c: i for i, c in enumerate(sorted(bookings["city"].unique()))}
                le_veh = {v: i for i, v in enumerate(sorted(bookings["vehicle_type"].unique()))}
                le_traf = {"Low": 0, "Medium": 1, "High": 2}
                le_wea = {w: i for i, w in enumerate(sorted(bookings["weather_condition"].unique()))}

                base = bookings[bookings["vehicle_type"] == f_vehicle]["base_fare"].median()
                row = {
                    "ride_distance_km": f_distance, "estimated_ride_time_min": f_time,
                    "surge_multiplier": f_surge, "base_fare": base,
                    "is_weekend": 0, "hour_of_day": f_hour,
                    "Rush_Hour_Flag": 1 if f_hour in [7,8,9,17,18,19] else 0,
                    "Long_Distance_Flag": 1 if f_distance >= 15 else 0,
                    "city_enc": le_city.get(f_city, 0),
                    "vehicle_type_enc": le_veh.get(f_vehicle, 0),
                    "traffic_level_enc": le_traf.get(f_traffic, 1),
                    "weather_condition_enc": le_wea.get(f_weather, 0),
                    "Driver_Reliability_Score": 7.0,
                    "avg_demand_requests": 5, "avg_surge": f_surge,
                }
                X = pd.DataFrame([row])[FARE_FEATURES]
                model = models.get("fare_prediction_model")
                if model:
                    pred_fare = model.predict(X)[0]
                    st.success(f"### 💰 Estimated Fare: ₹{pred_fare:.2f}")

                    # Fare breakdown
                    base_est = base
                    surge_add = pred_fare - base_est
                    fig = go.Figure(go.Bar(
                        x=["Base Fare", "Surge Component", "Predicted Total"],
                        y=[base_est, max(surge_add, 0), pred_fare],
                        marker_color=["#636EFA", "#FFA15A", "#00CC96"]
                    ))
                    fig.update_layout(title="Fare Breakdown (₹)", yaxis_title="Amount (₹)")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Fare model not found. Run train.py first.")

        # ── Customer Risk ─────────────────────────────────────────────────
        with pred_tab3:
            st.markdown("### Customer Cancellation Risk Assessment")
            col1, col2 = st.columns(2)
            with col1:
                c_cancel_rate = st.slider("Historical Cancel Rate", 0.0, 1.0, 0.25, key="cr_crate")
                c_rating = st.slider("Avg Customer Rating", 1.0, 5.0, 3.5, key="cr_rating")
                c_age = st.slider("Customer Age", 18, 70, 32, key="cr_age")
                c_days = st.slider("Account Age (days)", 1, 1000, 200, key="cr_days")
            with col2:
                c_vehicle = st.selectbox("Vehicle Type", sorted(bookings["vehicle_type"].unique()), key="cr_veh")
                c_city = st.selectbox("City", sorted(bookings["city"].unique()), key="cr_city")
                c_surge = st.slider("Current Surge", 1.0, 3.0, 1.8, key="cr_surge")
                c_hour = st.slider("Hour of Day", 0, 23, 8, key="cr_hour")

            if st.button("👤 Assess Risk", key="btn_cr"):
                le_city = {c: i for i, c in enumerate(sorted(bookings["city"].unique()))}
                le_veh = {v: i for i, v in enumerate(sorted(bookings["vehicle_type"].unique()))}
                row = {
                    "cancellation_rate": c_cancel_rate, "avg_customer_rating": c_rating,
                    "customer_age": c_age, "customer_signup_days_ago": c_days,
                    "is_weekend": 0, "hour_of_day": c_hour,
                    "Rush_Hour_Flag": 1 if c_hour in [7,8,9,17,18,19] else 0,
                    "surge_multiplier": c_surge,
                    "vehicle_type_enc": le_veh.get(c_vehicle, 0),
                    "city_enc": le_city.get(c_city, 0),
                    "traffic_level_enc": 1, "weather_condition_enc": 0,
                    "avg_demand_requests": 5, "avg_wait_time": 10,
                }
                X = pd.DataFrame([row])[CANCEL_RISK_FEATURES]
                model = models.get("customer_cancel_model")
                if model:
                    risk_prob = model.predict_proba(X)[0][1]
                    risk_label = "🔴 HIGH RISK" if risk_prob > 0.6 else ("🟡 MEDIUM RISK" if risk_prob > 0.3 else "🟢 LOW RISK")
                    st.markdown(f"### {risk_label}  —  Cancel Probability: **{risk_prob*100:.1f}%**")

                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=risk_prob * 100,
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={"text": "Cancellation Risk %"},
                        gauge={"axis": {"range": [0, 100]},
                               "bar": {"color": "#EF553B" if risk_prob > 0.5 else "#00CC96"},
                               "steps": [
                                   {"range": [0, 30], "color": "#d4f7dc"},
                                   {"range": [30, 60], "color": "#fff3cd"},
                                   {"range": [60, 100], "color": "#fde8e8"},
                               ]}
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Customer cancel model not found. Run train.py first.")

        # ── Driver Delay ──────────────────────────────────────────────────
        with pred_tab4:
            st.markdown("### Driver Delay Risk Assessment")
            col1, col2 = st.columns(2)
            with col1:
                d_accept = st.slider("Acceptance Rate", 0.0, 1.0, 0.75, key="dd_acc")
                d_delay = st.slider("Historical Delay Rate", 0.0, 1.0, 0.15, key="dd_delay")
                d_rating = st.slider("Avg Driver Rating", 1.0, 5.0, 4.0, key="dd_rating")
                d_pickup_delay = st.slider("Avg Pickup Delay (min)", 0.0, 30.0, 5.0, key="dd_pd")
            with col2:
                d_exp = st.slider("Experience (years)", 0, 15, 3, key="dd_exp")
                d_distance = st.slider("Distance (km)", 1.0, 50.0, 8.0, key="dd_dist")
                d_traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High"], key="dd_traf")
                d_hour = st.slider("Hour of Day", 0, 23, 9, key="dd_hour")

            if st.button("🚗 Predict Delay", key="btn_dd"):
                le_traf = {"Low": 0, "Medium": 1, "High": 2}
                row = {
                    "acceptance_rate": d_accept, "delay_rate": d_delay,
                    "avg_driver_rating": d_rating, "avg_pickup_delay_min": d_pickup_delay,
                    "driver_experience_years": d_exp, "ride_distance_km": d_distance,
                    "estimated_ride_time_min": d_distance * 3.5,
                    "traffic_level_enc": le_traf.get(d_traffic, 1),
                    "weather_condition_enc": 0, "hour_of_day": d_hour,
                    "Rush_Hour_Flag": 1 if d_hour in [7,8,9,17,18,19] else 0,
                    "avg_demand_requests": 5, "avg_surge": 1.2,
                }
                X = pd.DataFrame([row])[DRIVER_DELAY_FEATURES]
                model = models.get("driver_delay_model")
                if model:
                    delay_prob = model.predict_proba(X)[0][1]
                    delay_label = "🔴 LIKELY TO DELAY" if delay_prob > 0.5 else "🟢 RELIABLE DRIVER"
                    st.markdown(f"### {delay_label}  —  Delay Probability: **{delay_prob*100:.1f}%**")

                    col1, col2 = st.columns(2)
                    with col1:
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=delay_prob * 100,
                            title={"text": "Delay Risk %"},
                            gauge={"axis": {"range": [0, 100]},
                                   "bar": {"color": "#EF553B" if delay_prob > 0.5 else "#00CC96"},
                                   "steps": [
                                       {"range": [0, 30], "color": "#d4f7dc"},
                                       {"range": [30, 60], "color": "#fff3cd"},
                                       {"range": [60, 100], "color": "#fde8e8"},
                                   ]}
                        ))
                        fig.update_layout(height=280)
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        reliability = (d_accept * d_rating * 2 - d_delay * 5 - d_pickup_delay * 0.05)
                        reliability = max(0, min(10, reliability))
                        fig2 = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=reliability,
                            title={"text": "Reliability Score (0–10)"},
                            gauge={"axis": {"range": [0, 10]},
                                   "bar": {"color": "#636EFA"}}
                        ))
                        fig2.update_layout(height=280)
                        st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.error("Driver delay model not found. Run train.py first.")


# --------------------------------------------------------------------------
# TAB 3 — Demand & Locations
# --------------------------------------------------------------------------
with tab3:
    st.markdown("## 🗺 Demand & Location Analytics")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_demand_heatmap(location_demand), use_container_width=True)
    with col2:
        # Top pickup locations
        top_pickup = (
            location_demand.groupby("pickup_location")["total_requests"]
            .sum().sort_values(ascending=False).head(15).reset_index()
        )
        fig = px.bar(top_pickup, x="total_requests", y="pickup_location", orientation="h",
                     title="📍 Top 15 High-Demand Pickup Locations",
                     color="total_requests", color_continuous_scale="Oranges")
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        # Vehicle demand split
        veh_demand = location_demand.groupby("vehicle_type")["total_requests"].sum().reset_index()
        fig = px.pie(veh_demand, names="vehicle_type", values="total_requests",
                     title="🚗 Vehicle Type Demand Share", color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        # Demand by hour (line)
        hr_demand = location_demand.groupby("hour_of_day")["total_requests"].sum().reset_index()
        fig = px.line(hr_demand, x="hour_of_day", y="total_requests", markers=True,
                      title="⏱ Total Ride Requests by Hour",
                      color_discrete_sequence=["#FF6B35"],
                      labels={"hour_of_day": "Hour", "total_requests": "Requests"})
        st.plotly_chart(fig, use_container_width=True)

    # Surge vs demand
    st.markdown("### ⚡ Surge vs Demand Level")
    fig = px.scatter(
        location_demand.sample(min(3000, len(location_demand)), random_state=42),
        x="avg_surge_multiplier", y="total_requests",
        color="demand_level", size="cancelled_rides",
        color_discrete_sequence=px.colors.qualitative.Bold,
        title="Surge Multiplier vs Total Requests",
        labels={"avg_surge_multiplier": "Avg Surge", "total_requests": "Requests"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Time features analysis
    st.markdown("### 📅 Temporal Features Analysis")
    col1, col2 = st.columns(2)
    with col1:
        season_cnt = time_features["season"].value_counts().reset_index()
        season_cnt.columns = ["season", "hours"]
        fig = px.bar(season_cnt, x="season", y="hours", title="🌤 Hours per Season",
                     color="season", color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        peak_dist = time_features.groupby("hour_of_day")["peak_time_flag"].mean().reset_index()
        fig = px.bar(peak_dist, x="hour_of_day", y="peak_time_flag",
                     title="📈 Peak Time Flag by Hour",
                     color="peak_time_flag", color_continuous_scale="Reds",
                     labels={"peak_time_flag": "Peak Probability"})
        st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------
# TAB 4 — Model Performance
# --------------------------------------------------------------------------
with tab4:
    st.markdown("## 📈 Model Performance Dashboard")

    if not models:
        st.warning("⚠️ Models not trained yet. Run `python train.py` to train all models.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            fi = models.get("ride_outcome_feature_importance")
            if fi is not None:
                st.plotly_chart(plot_feature_importance(fi, "Ride Outcome — Feature Importance"),
                                use_container_width=True)
            fi = models.get("customer_cancel_feature_importance")
            if fi is not None:
                st.plotly_chart(plot_feature_importance(fi, "Customer Cancel Risk — Feature Importance"),
                                use_container_width=True)

        with col2:
            fi = models.get("fare_feature_importance")
            if fi is not None:
                st.plotly_chart(plot_feature_importance(fi, "Fare Prediction — Feature Importance"),
                                use_container_width=True)
            fi = models.get("driver_delay_feature_importance")
            if fi is not None:
                st.plotly_chart(plot_feature_importance(fi, "Driver Delay — Feature Importance"),
                                use_container_width=True)

    # Model cards
    st.markdown("### 🏆 Target Industry Benchmarks")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Ride Outcome Accuracy", "85–90%", "Target")
    mc2.metric("Fare RMSE", "±10% of Mean", "Target")
    mc3.metric("Cancel Risk AUC", "> 0.80", "Target")
    mc4.metric("Driver Delay AUC", "> 0.80", "Target")

    st.markdown("### 📋 Algorithm Summary")
    algo_df = pd.DataFrame({
        "Model": ["Ride Outcome", "Fare Prediction", "Customer Cancel Risk", "Driver Delay"],
        "Type": ["Multi-Class Classification", "Regression", "Binary Classification", "Binary Classification"],
        "Algorithm": ["Random Forest", "Gradient Boosting", "Random Forest", "Random Forest"],
        "Target": ["booking_status_encoded", "booking_value", "customer_cancel_flag", "driver_delay_flag"],
        "Evaluation": ["Accuracy, F1, AUC, CM", "RMSE, MAE, R²", "Accuracy, F1, AUC", "Accuracy, F1, AUC"],
    })
    st.dataframe(algo_df, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------
# TAB 5 — Data Explorer
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# TAB 5 — SQL Analytics
# --------------------------------------------------------------------------
with tab5:
    st.markdown("""## 🗄 SQL Analytics — Data Management""")
    st.markdown("All queries run on the **rapido.db** SQLite database with indexed tables.")

    # ── MySQL Connection status ──────────────────────────────────────────
    from utils.sql_manager import MYSQL_CONFIG, DB_NAME
    try:
        import mysql.connector
        test_conn = mysql.connector.connect(**MYSQL_CONFIG)
        test_conn.close()
        st.success(
            f"✅ MySQL Connected  |  Host: {MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}  |  "
            f"Database:   |  Tables: bookings, customers, drivers, location_demand, time_features"
        )
    except Exception as e:
        st.error(f"❌ MySQL connection failed: {e}")
        st.info("👉 Make sure MySQL is running and your .env file has correct credentials. Then run: python utils/sql_manager.py")
        st.stop()

    # ── Preset queries ────────────────────────────────────────────────────
    query_choice = st.selectbox("📋 Select a Pre-built Analytical Query", [
        "Booking Summary KPIs",
        "Cancellation Rate by City",
        "Fare by Vehicle Type (Completed Rides)",
        "Top Peak Cancellation Windows",
        "Top High-Risk Customers",
        "Driver Reliability Rankings",
        "Surge & Demand Patterns",
        "Traffic & Weather Impact on Cancellations",
        "Custom SQL Query",
    ])

    df_sql = pd.DataFrame()

    if query_choice == "Booking Summary KPIs":
        df_sql = query_booking_summary()
        st.dataframe(df_sql, use_container_width=True)

    elif query_choice == "Cancellation Rate by City":
        df_sql = query_cancellation_by_city()
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(df_sql, use_container_width=True)
        with col2:
            fig = px.bar(df_sql, x="city", y="cancel_rate_pct",
                         title="Cancellation Rate % by City",
                         color="cancel_rate_pct", color_continuous_scale="Reds")
            st.plotly_chart(fig, use_container_width=True)

    elif query_choice == "Fare by Vehicle Type (Completed Rides)": 
        df_sql = query_fare_by_vehicle()
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(df_sql, use_container_width=True)
        with col2:
            fig = px.bar(df_sql, x="vehicle_type", y="avg_fare",
                         title="Average Fare (₹) by Vehicle Type",
                         color="vehicle_type",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)

    elif query_choice == "Top Peak Cancellation Windows":
        df_sql = query_peak_cancellation_windows()
        st.dataframe(df_sql, use_container_width=True)
        fig = px.scatter(df_sql, x="hour_of_day", y="cancel_rate_pct",
                         size="total", color="city",
                         title="Peak Cancellation Windows (size = total bookings)",
                         color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig, use_container_width=True)

    elif query_choice == "Top High-Risk Customers":
        n = st.slider("Number of customers", 5, 50, 20)
        df_sql = query_high_risk_customers(n)
        st.dataframe(df_sql, use_container_width=True)
        fig = px.bar(df_sql.head(15), x="customer_id", y="cancellation_rate",
                     title="Top High-Risk Customers by Cancellation Rate",
                     color="cancellation_rate", color_continuous_scale="Reds")
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    elif query_choice == "Driver Reliability Rankings":
        df_sql = query_driver_reliability()
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(df_sql.head(20), use_container_width=True)
        with col2:
            fig = px.histogram(df_sql, x="reliability_score", nbins=20,
                               title="Driver Reliability Score Distribution",
                               color_discrete_sequence=["#636EFA"])
            st.plotly_chart(fig, use_container_width=True)

    elif query_choice == "Surge & Demand Patterns":
        df_sql = query_surge_demand_pattern()
        st.dataframe(df_sql, use_container_width=True)
        fig = px.bar(df_sql, x="city", y="avg_surge", color="demand_level",
                     barmode="group", title="Avg Surge by City & Demand Level",
                     color_discrete_sequence=px.colors.qualitative.Set1)
        st.plotly_chart(fig, use_container_width=True)

    elif query_choice == "Traffic & Weather Impact on Cancellations":
        df_sql = query_traffic_weather_impact()
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(df_sql, use_container_width=True)
        with col2:
            fig = px.bar(df_sql, x="traffic_level", y="cancel_rate_pct",
                         color="weather_condition", barmode="group",
                         title="Cancel Rate % by Traffic × Weather",
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)

    elif query_choice == "Custom SQL Query":
        st.markdown("**Write your own SQL query on the rapido.db database:**")
        st.markdown("Tables: , , , , ")
        custom_sql = st.text_area("SQL Query",
            value="SELECT city, COUNT(*) AS rides FROM bookings GROUP BY city ORDER BY rides DESC",
            height=120)
        if st.button("▶ Run Query"):
            try:
                df_sql = run_query(custom_sql)
                st.success(f"✅ Returned {len(df_sql)} rows")
                st.dataframe(df_sql, use_container_width=True)
                csv = df_sql.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Result", csv, "query_result.csv", "text/csv")
            except Exception as e:
                st.error(f"Query error: {e}")

    # SQL DDL reference
    with st.expander("📋 View Schema & Indexes"):
        st.code("""
-- Tables
bookings       (100,000 rows) — booking_id PK, booking_date, city, vehicle_type, booking_status, ...
customers      (10,000 rows)  — customer_id PK, cancellation_rate, avg_customer_rating, ...
drivers        (5,000 rows)   — driver_id PK, delay_rate, acceptance_rate, avg_driver_rating, ...
location_demand(17,941 rows)  — city, pickup_location, hour_of_day, demand_level, ...
time_features  (8,760 rows)   — datetime PK, hour_of_day, peak_time_flag, season, ...

-- Indexes (14 total for query performance)
idx_bookings_city, idx_bookings_status, idx_bookings_date
idx_bookings_customer, idx_bookings_driver, idx_bookings_hour
idx_customers_city, idx_customers_flag
idx_drivers_city, idx_drivers_flag
idx_location_city, idx_location_hour
idx_time_hour
""", language="sql")


with tab6:
    st.markdown("## 🔍 Data Explorer")

    dataset_choice = st.selectbox("Choose Dataset", [
        "bookings", "customers", "drivers", "location_demand",
        "time_features_csv", "time_features_xlsx", "time_features_xlsx_v2"
    ])

    df_show = data[dataset_choice].copy()

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{df_show.shape[0]:,}")
    col2.metric("Columns", df_show.shape[1])
    col3.metric("Missing Values", f"{df_show.isnull().sum().sum():,}")

    # Search/filter
    search_col = st.selectbox("Filter by Column", ["None"] + list(df_show.columns))
    if search_col != "None" and df_show[search_col].dtype == object:
        search_val = st.text_input(f"Search in {search_col}")
        if search_val:
            df_show = df_show[df_show[search_col].str.contains(search_val, case=False, na=False)]

    n_rows = st.slider("Rows to display", 10, 500, 50)
    st.dataframe(df_show.head(n_rows), use_container_width=True)

    # Missing values chart
    missing = df_show.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        fig = px.bar(x=missing.index, y=missing.values,
                     title="Missing Values per Column",
                     labels={"x": "Column", "y": "Missing Count"},
                     color=missing.values, color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ No missing values in this dataset!")

    # Numeric stats
    st.markdown("### 📊 Statistical Summary")
    st.dataframe(df_show.describe(), use_container_width=True)

    # Download
    csv = df_show.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Filtered Data as CSV",
        data=csv,
        file_name=f"{dataset_choice}_filtered.csv",
        mime="text/csv",
    )
