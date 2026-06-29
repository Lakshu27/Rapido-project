"""
train.py
--------
Entry point: loads data, builds master dataset, trains all four models.

Usage:
    python train.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from utils.data_loader import load_all
from utils.preprocessing import build_master_dataset
from utils.model_training import train_all_models


def main():
    print("=" * 60)
    print("  RAPIDO — INTELLIGENT MOBILITY INSIGHTS")
    print("  Model Training Pipeline")
    print("=" * 60)

    # ── Step 1: Load all 7 files ──────────────────────────────────────────
    data = load_all()

    bookings = data["bookings"]
    customers = data["customers"]
    drivers = data["drivers"]
    location_demand = data["location_demand"]
    time_csv = data["time_features_csv"]
    time_xlsx = data["time_features_xlsx"]
    time_xlsx_v2 = data["time_features_xlsx_v2"]

    print(f"\ntime_features CSV  : {time_csv.shape}")
    print(f"time_features XLSX : {time_xlsx.shape}")
    print(f"time_features v2   : {time_xlsx_v2.shape}")

    # ── Step 2: Build master dataset ──────────────────────────────────────
    print("\nBuilding master dataset (cleaning + feature engineering)...")
    master = build_master_dataset(bookings, customers, drivers, location_demand)

    # ── Step 3: Train all models ──────────────────────────────────────────
    results = train_all_models(master)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE — Summary")
    print("=" * 60)

    m = results["ride_outcome"]["metrics"]
    print(f"  Ride Outcome    → Accuracy: {m['accuracy']:.3f}  F1: {m['f1']:.3f}")

    m = results["fare"]["metrics"]
    print(f"  Fare Model      → RMSE: {m['rmse']:.2f}  R²: {m['r2']:.3f}  (%RMSE: {m['pct_rmse']:.1f}%)")

    m = results["customer_cancel"]["metrics"]
    print(f"  Cancel Risk     → Accuracy: {m['accuracy']:.3f}  F1: {m['f1']:.3f}")

    m = results["driver_delay"]["metrics"]
    print(f"  Driver Delay    → Accuracy: {m['accuracy']:.3f}  F1: {m['f1']:.3f}")

    print("\nModels saved to /models/ directory.")
    print("Run 'streamlit run app/streamlit_app.py' to launch the dashboard.\n")


if __name__ == "__main__":
    main()
