"""
eda_utils.py
------------
Step 2 — Exploratory Data Analysis helpers.
Returns Plotly figures for use in Streamlit.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── colour palette ────────────────────────────────────────────────────────
COLORS = px.colors.qualitative.Set2
CANCEL_COLOR = "#EF553B"
COMPLETE_COLOR = "#00CC96"
INCOMPLETE_COLOR = "#FFA15A"


def plot_ride_volume_by_hour(bookings: pd.DataFrame) -> go.Figure:
    """Bar chart: ride volume by hour of day."""
    vol = bookings.groupby("hour_of_day").size().reset_index(name="count")
    fig = px.bar(vol, x="hour_of_day", y="count", title="🕐 Ride Volume by Hour of Day",
                 color="count", color_continuous_scale="Blues",
                 labels={"hour_of_day": "Hour", "count": "Bookings"})
    fig.update_layout(showlegend=False)
    return fig


def plot_ride_volume_by_weekday(bookings: pd.DataFrame) -> go.Figure:
    """Bar chart: ride volume by weekday."""
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    vol = bookings.groupby("day_of_week").size().reset_index(name="count")
    vol["day_of_week"] = pd.Categorical(vol["day_of_week"], categories=order, ordered=True)
    vol = vol.sort_values("day_of_week")
    fig = px.bar(vol, x="day_of_week", y="count", title="📅 Ride Volume by Weekday",
                 color="count", color_continuous_scale="Teal",
                 labels={"day_of_week": "Day", "count": "Bookings"})
    fig.update_layout(showlegend=False)
    return fig


def plot_ride_volume_by_city(bookings: pd.DataFrame) -> go.Figure:
    """Pie chart: ride distribution across cities."""
    vol = bookings["city"].value_counts().reset_index()
    vol.columns = ["city", "count"]
    fig = px.pie(vol, names="city", values="count", title="🌆 Ride Distribution by City",
                 color_discrete_sequence=COLORS)
    return fig


def plot_cancellation_heatmap(bookings: pd.DataFrame) -> go.Figure:
    """Heatmap: cancellation rate by city × hour."""
    df = bookings.copy()
    df["is_cancelled"] = (df["booking_status"] == "Cancelled").astype(int)
    pivot = df.pivot_table(values="is_cancelled", index="city",
                           columns="hour_of_day", aggfunc="mean", fill_value=0)
    fig = px.imshow(pivot, title="🔥 Cancellation Rate Heatmap (City × Hour)",
                    color_continuous_scale="Reds",
                    labels={"x": "Hour", "y": "City", "color": "Cancel Rate"},
                    aspect="auto")
    return fig


def plot_distance_vs_fare(bookings: pd.DataFrame) -> go.Figure:
    """Scatter: distance vs fare coloured by vehicle type."""
    df = bookings[bookings["booking_status"] == "Completed"]
    df = df.sample(min(5000, len(df)), random_state=42) if len(df) > 0 else df
    fig = px.scatter(df, x="ride_distance_km", y="booking_value",
                     color="vehicle_type", opacity=0.5,
                     title="📏 Distance vs Fare (Completed Rides)",
                     labels={"ride_distance_km": "Distance (km)", "booking_value": "Fare (₹)"},
                     color_discrete_sequence=COLORS)
    return fig


def plot_rating_distribution(bookings: pd.DataFrame, customers: pd.DataFrame, drivers: pd.DataFrame) -> go.Figure:
    """Histogram overlay: customer vs driver ratings."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=customers["avg_customer_rating"], name="Customer Rating",
                                opacity=0.7, marker_color="#636EFA", nbinsx=20))
    fig.add_trace(go.Histogram(x=drivers["avg_driver_rating"], name="Driver Rating",
                                opacity=0.7, marker_color="#EF553B", nbinsx=20))
    fig.update_layout(barmode="overlay", title="⭐ Rating Distribution: Customer vs Driver",
                       xaxis_title="Rating", yaxis_title="Count")
    return fig


def plot_booking_status_breakdown(bookings: pd.DataFrame) -> go.Figure:
    """Donut chart: booking status breakdown."""
    counts = bookings["booking_status"].value_counts().reset_index()
    counts.columns = ["status", "count"]
    color_map = {"Completed": COMPLETE_COLOR, "Cancelled": CANCEL_COLOR, "Incomplete": INCOMPLETE_COLOR}
    fig = px.pie(counts, names="status", values="count", hole=0.45,
                 title="📊 Booking Status Breakdown",
                 color="status", color_discrete_map=color_map)
    return fig


def plot_payment_method(bookings: pd.DataFrame) -> go.Figure:
    """Payment method usage — fallback to vehicle_type if no payment column."""
    if "payment_method" in bookings.columns:
        col = "payment_method"
        title = "💳 Payment Method Usage"
    else:
        col = "vehicle_type"
        title = "🚗 Bookings by Vehicle Type"
    vol = bookings[col].value_counts().reset_index()
    vol.columns = [col, "count"]
    fig = px.bar(vol, x=col, y="count", title=title, color=col,
                 color_discrete_sequence=COLORS)
    fig.update_layout(showlegend=False)
    return fig


def plot_traffic_weather_cancellation(bookings: pd.DataFrame) -> go.Figure:
    """Grouped bar: cancellation rate by traffic × weather."""
    df = bookings.copy()
    df["is_cancelled"] = (df["booking_status"] == "Cancelled").astype(int)
    grp = df.groupby(["traffic_level", "weather_condition"])["is_cancelled"].mean().reset_index()
    grp.columns = ["traffic_level", "weather_condition", "cancel_rate"]
    fig = px.bar(grp, x="traffic_level", y="cancel_rate", color="weather_condition",
                 barmode="group", title="🌦 Traffic & Weather vs Cancellation Rate",
                 labels={"cancel_rate": "Cancellation Rate"},
                 color_discrete_sequence=COLORS)
    return fig


def plot_surge_behavior(bookings: pd.DataFrame) -> go.Figure:
    """Box plot: surge multiplier by city."""
    fig = px.box(bookings, x="city", y="surge_multiplier", color="city",
                 title="⚡ Surge Multiplier by City",
                 labels={"surge_multiplier": "Surge Multiplier"},
                 color_discrete_sequence=COLORS)
    fig.update_layout(showlegend=False)
    return fig


def plot_cancellations_by_hour(bookings: pd.DataFrame) -> go.Figure:
    """Line chart: cancellations by hour."""
    df = bookings[bookings["booking_status"] == "Cancelled"]
    vol = df.groupby("hour_of_day").size().reset_index(name="cancellations")
    fig = px.line(vol, x="hour_of_day", y="cancellations",
                  title="⏱ Cancellations by Hour of Day",
                  markers=True, color_discrete_sequence=[CANCEL_COLOR],
                  labels={"hour_of_day": "Hour", "cancellations": "Cancellations"})
    return fig


def plot_customer_vs_driver_cancel(bookings: pd.DataFrame) -> go.Figure:
    """Compare booking status between customer and driver behaviour signals."""
    status_hour = bookings.groupby(["hour_of_day", "booking_status"]).size().reset_index(name="count")
    fig = px.line(status_hour, x="hour_of_day", y="count", color="booking_status",
                  title="🚦 Booking Outcomes by Hour",
                  color_discrete_map={"Completed": COMPLETE_COLOR,
                                       "Cancelled": CANCEL_COLOR,
                                       "Incomplete": INCOMPLETE_COLOR},
                  labels={"hour_of_day": "Hour", "count": "Count", "booking_status": "Status"})
    return fig


def plot_feature_importance(feat_imp: pd.Series, title: str = "Feature Importance") -> go.Figure:
    """Horizontal bar chart for feature importance."""
    top = feat_imp.head(15).sort_values()
    fig = px.bar(x=top.values, y=top.index, orientation="h",
                 title=f"🔑 {title}",
                 labels={"x": "Importance", "y": "Feature"},
                 color=top.values, color_continuous_scale="Viridis")
    fig.update_layout(showlegend=False, height=450)
    return fig


def plot_demand_heatmap(location_demand: pd.DataFrame) -> go.Figure:
    """Demand level heatmap by city and hour."""
    pivot = location_demand.pivot_table(
        values="total_requests", index="city", columns="hour_of_day", aggfunc="sum", fill_value=0
    )
    fig = px.imshow(pivot, title="📍 Demand Heatmap (City × Hour)",
                    color_continuous_scale="YlOrRd",
                    labels={"x": "Hour", "y": "City", "color": "Requests"},
                    aspect="auto")
    return fig