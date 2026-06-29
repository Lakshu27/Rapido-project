"""
data_loader.py
--------------
Utility functions for loading all Rapido datasets (CSV + XLSX).
Loads 5 CSV files and 2 XLSX files as specified in the project.
"""

import os
import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_bookings() -> pd.DataFrame:
    """Load core transactional bookings data."""
    path = os.path.join(DATA_DIR, "bookings.csv")
    df = pd.read_csv(path, parse_dates=["booking_date"])
    df["booking_time"] = pd.to_datetime(df["booking_time"], format="%H:%M:%S", errors="coerce")
    return df


def load_customers() -> pd.DataFrame:
    """Load customer behaviour and historical cancellation signals."""
    path = os.path.join(DATA_DIR, "customers.csv")
    return pd.read_csv(path)


def load_drivers() -> pd.DataFrame:
    """Load driver performance, delay, and reliability metrics."""
    path = os.path.join(DATA_DIR, "drivers.csv")
    return pd.read_csv(path)


def load_location_demand() -> pd.DataFrame:
    """Load aggregated demand patterns by location and time."""
    path = os.path.join(DATA_DIR, "location_demand.csv")
    return pd.read_csv(path)


def load_time_features_csv() -> pd.DataFrame:
    """Load enriched temporal signals from CSV."""
    path = os.path.join(DATA_DIR, "time_features.csv")
    df = pd.read_csv(path, parse_dates=["datetime"])
    return df


def load_time_features_xlsx() -> pd.DataFrame:
    """Load enriched temporal signals from XLSX (version 1)."""
    path = os.path.join(DATA_DIR, "time_features.xlsx")
    df = pd.read_excel(path, parse_dates=["datetime"])
    return df


def load_time_features_xlsx_v2() -> pd.DataFrame:
    """Load enriched temporal signals from XLSX (version 2)."""
    path = os.path.join(DATA_DIR, "time_features_v2.xlsx")
    df = pd.read_excel(path, parse_dates=["datetime"])
    return df


def load_all() -> dict:
    """
    Load all 7 data files (5 CSV + 2 XLSX) and return as a dict.

    Returns
    -------
    dict with keys:
        bookings, customers, drivers, location_demand,
        time_features_csv, time_features_xlsx, time_features_xlsx_v2
    """
    print("Loading all datasets...")
    data = {
        "bookings": load_bookings(),
        "customers": load_customers(),
        "drivers": load_drivers(),
        "location_demand": load_location_demand(),
        "time_features_csv": load_time_features_csv(),
        "time_features_xlsx": load_time_features_xlsx(),
        "time_features_xlsx_v2": load_time_features_xlsx_v2(),
    }
    for name, df in data.items():
        print(f"  ✔ {name}: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return data
