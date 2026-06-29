"""
preprocessing.py
----------------
Step 1 – Data Cleaning
Step 3 – Feature Engineering

Handles missing values, type conversions, encoding, and
creation of all derived features required by the project spec.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Step 1 — Data Cleaning
# ---------------------------------------------------------------------------

def clean_bookings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the bookings DataFrame:
    - Parse datetime columns
    - Fill / flag missing values
    - Encode categoricals
    """
    df = df.copy()

    # ── datetime ──────────────────────────────────────────────────────────
    df["booking_date"] = pd.to_datetime(df["booking_date"], errors="coerce")
    df["booking_datetime"] = pd.to_datetime(
        df["booking_date"].astype(str) + " " + df["booking_time"].astype(str),
        errors="coerce",
    )

    # ── missing: actual_ride_time_min (NaN for cancelled/incomplete) ───────
    # Fill with median per vehicle type for completed rides
    median_time = (
        df[df["booking_status"] == "Completed"]
        .groupby("vehicle_type")["actual_ride_time_min"]
        .median()
    )
    df["actual_ride_time_min"] = df.apply(
        lambda r: median_time.get(r["vehicle_type"], np.nan)
        if pd.isna(r["actual_ride_time_min"])
        else r["actual_ride_time_min"],
        axis=1,
    )

    # ── missing: incomplete_ride_reason ───────────────────────────────────
    df["incomplete_ride_reason"] = df["incomplete_ride_reason"].fillna("None")

    # ── encode booking_status ─────────────────────────────────────────────
    status_map = {"Completed": 0, "Cancelled": 1, "Incomplete": 2}
    df["booking_status_encoded"] = df["booking_status"].map(status_map)

    # ── encode categoricals ───────────────────────────────────────────────
    cat_cols = ["city", "pickup_location", "drop_location", "vehicle_type",
                "traffic_level", "weather_condition", "day_of_week"]
    le = LabelEncoder()
    for col in cat_cols:
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))

    return df


def clean_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Clean customers DataFrame."""
    df = df.copy()
    df["cancellation_rate"] = df["cancellation_rate"].clip(0, 1)
    df["avg_customer_rating"] = df["avg_customer_rating"].fillna(df["avg_customer_rating"].median())
    return df


def clean_drivers(df: pd.DataFrame) -> pd.DataFrame:
    """Clean drivers DataFrame."""
    df = df.copy()
    df["avg_pickup_delay_min"] = df["avg_pickup_delay_min"].fillna(0)
    df["avg_driver_rating"] = df["avg_driver_rating"].fillna(df["avg_driver_rating"].median())
    return df


def clean_location_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Clean location_demand DataFrame."""
    df = df.copy()
    df["avg_wait_time_min"] = df["avg_wait_time_min"].fillna(df["avg_wait_time_min"].median())
    demand_map = {"Low": 0, "Medium": 1, "High": 2}
    df["demand_level_enc"] = df["demand_level"].map(demand_map).fillna(1)
    return df


def clean_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean time_features DataFrame."""
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    season_map = {"Winter": 0, "Spring": 1, "Summer": 2, "Monsoon": 3, "Autumn": 4, "Fall": 4}
    df["season_enc"] = df["season"].map(season_map).fillna(0).astype(int)
    return df


# ---------------------------------------------------------------------------
# Step 3 — Feature Engineering
# ---------------------------------------------------------------------------

def engineer_features(bookings: pd.DataFrame,
                       customers: pd.DataFrame,
                       drivers: pd.DataFrame) -> pd.DataFrame:
    """
    Create all derived features:
    - Fare_per_KM, Fare_per_Min
    - Rush_Hour_Flag, Long_Distance_Flag
    - City_Pair
    - Driver_Reliability_Score
    - Customer_Loyalty_Score
    """
    df = bookings.copy()

    # ── fare-based features ───────────────────────────────────────────────
    df["Fare_per_KM"] = df["booking_value"] / df["ride_distance_km"].replace(0, np.nan)
    df["Fare_per_Min"] = df["booking_value"] / df["actual_ride_time_min"].replace(0, np.nan)
    df["Fare_per_KM"] = df["Fare_per_KM"].fillna(0)
    df["Fare_per_Min"] = df["Fare_per_Min"].fillna(0)

    # ── time-based flags ──────────────────────────────────────────────────
    rush_hours = [7, 8, 9, 17, 18, 19, 20]
    df["Rush_Hour_Flag"] = df["hour_of_day"].isin(rush_hours).astype(int)

    # ── distance flag (top quartile) ──────────────────────────────────────
    q75 = df["ride_distance_km"].quantile(0.75)
    df["Long_Distance_Flag"] = (df["ride_distance_km"] >= q75).astype(int)

    # ── city pair ─────────────────────────────────────────────────────────
    df["City_Pair"] = df["pickup_location"] + "_" + df["drop_location"]

    # ── Driver_Reliability_Score ──────────────────────────────────────────
    driver_score = drivers[["driver_id", "acceptance_rate", "delay_rate",
                             "avg_driver_rating", "avg_pickup_delay_min"]].copy()
    # Score = acceptance_rate * rating - delay_rate penalty (0–10 scale)
    driver_score["Driver_Reliability_Score"] = (
        driver_score["acceptance_rate"] * driver_score["avg_driver_rating"] * 2
        - driver_score["delay_rate"] * 5
        - driver_score["avg_pickup_delay_min"] * 0.05
    ).clip(0, 10)
    df = df.merge(driver_score[["driver_id", "Driver_Reliability_Score"]],
                  on="driver_id", how="left")
    df["Driver_Reliability_Score"] = df["Driver_Reliability_Score"].fillna(5.0)

    # ── Customer_Loyalty_Score ────────────────────────────────────────────
    cust_score = customers[["customer_id", "cancellation_rate",
                             "avg_customer_rating", "total_bookings",
                             "customer_signup_days_ago"]].copy()
    cust_score["Customer_Loyalty_Score"] = (
        (1 - cust_score["cancellation_rate"]) * cust_score["avg_customer_rating"]
        + np.log1p(cust_score["total_bookings"]) * 0.3
        + np.log1p(cust_score["customer_signup_days_ago"]) * 0.1
    ).clip(0, 10)
    df = df.merge(cust_score[["customer_id", "Customer_Loyalty_Score"]],
                  on="customer_id", how="left")
    df["Customer_Loyalty_Score"] = df["Customer_Loyalty_Score"].fillna(5.0)

    return df


def build_master_dataset(bookings: pd.DataFrame,
                          customers: pd.DataFrame,
                          drivers: pd.DataFrame,
                          location_demand: pd.DataFrame) -> pd.DataFrame:
    """
    Merge all tables into a single modelling-ready master dataset.
    """
    # Clean each table
    bk = clean_bookings(bookings)
    cu = clean_customers(customers)
    dr = clean_drivers(drivers)
    ld = clean_location_demand(location_demand)

    # Engineer features
    master = engineer_features(bk, cu, dr)

    # Merge customer features
    cu_cols = ["customer_id", "customer_age", "customer_signup_days_ago",
               "cancellation_rate", "avg_customer_rating", "customer_cancel_flag"]
    master = master.merge(cu[cu_cols], on="customer_id", how="left")

    # Merge driver features
    dr_cols = ["driver_id", "driver_experience_years", "acceptance_rate",
               "delay_rate", "avg_driver_rating", "avg_pickup_delay_min", "driver_delay_flag"]
    master = master.merge(dr[dr_cols], on="driver_id", how="left")

    # Merge location demand (avg per pickup_location + vehicle_type)
    ld_agg = (
        ld.groupby(["city", "pickup_location", "vehicle_type"])
        .agg(avg_demand_requests=("total_requests", "mean"),
             avg_wait_time=("avg_wait_time_min", "mean"),
             avg_surge=("avg_surge_multiplier", "mean"))
        .reset_index()
    )
    master = master.merge(ld_agg, on=["city", "pickup_location", "vehicle_type"], how="left")

    # Fill remaining NaNs
    num_cols = master.select_dtypes(include=[np.number]).columns
    master[num_cols] = master[num_cols].fillna(master[num_cols].median())

    print(f"Master dataset: {master.shape[0]:,} rows × {master.shape[1]} columns")
    return master
