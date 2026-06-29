"""
model_training.py
-----------------
Step 4 — Model Training
Step 5 — Model Evaluation

Four models:
  1. Ride Outcome Prediction   – Multi-Class Classification (RF)
  2. Fare Prediction           – Regression (GBM)
  3. Customer Cancellation Risk – Binary Classification (RF)
  4. Driver Delay Prediction   – Binary Classification (RF)
"""

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report, mean_squared_error, mean_absolute_error, r2_score,
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def save_model(model, name: str):
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  ✔ Saved {name}.pkl")


def load_model(name: str):
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def eval_classifier(y_test, y_pred, y_prob=None, label="Model"):
    """Print classification evaluation metrics."""
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    if y_prob is not None:
        try:
            if y_prob.ndim == 2:
                auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
            else:
                auc = roc_auc_score(y_test, y_prob)
            print(f"  AUC      : {auc:.4f}")
        except Exception:
            pass
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")
    return {"accuracy": acc, "f1": f1, "confusion_matrix": cm}


def eval_regressor(y_test, y_pred, label="Regression Model"):
    """Print regression evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    pct_rmse = (rmse / y_test.mean()) * 100 if y_test.mean() != 0 else 0
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  RMSE     : {rmse:.4f}  ({pct_rmse:.1f}% of mean)")
    print(f"  MAE      : {mae:.4f}")
    print(f"  R²       : {r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2, "pct_rmse": pct_rmse}


# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------

OUTCOME_FEATURES = [
    "ride_distance_km", "estimated_ride_time_min", "actual_ride_time_min",
    "base_fare", "surge_multiplier", "booking_value", "is_weekend", "hour_of_day",
    "Rush_Hour_Flag", "Long_Distance_Flag", "Fare_per_KM", "Fare_per_Min",
    "Driver_Reliability_Score", "Customer_Loyalty_Score",
    "city_enc", "vehicle_type_enc", "traffic_level_enc", "weather_condition_enc",
    "acceptance_rate", "delay_rate", "cancellation_rate",
    "avg_demand_requests", "avg_wait_time", "avg_surge",
]

FARE_FEATURES = [
    "ride_distance_km", "estimated_ride_time_min", "surge_multiplier", "base_fare",
    "is_weekend", "hour_of_day", "Rush_Hour_Flag", "Long_Distance_Flag",
    "city_enc", "vehicle_type_enc", "traffic_level_enc", "weather_condition_enc",
    "Driver_Reliability_Score", "avg_demand_requests", "avg_surge",
]

CANCEL_RISK_FEATURES = [
    "cancellation_rate", "avg_customer_rating", "customer_age",
    "customer_signup_days_ago", "is_weekend", "hour_of_day",
    "Rush_Hour_Flag", "surge_multiplier", "vehicle_type_enc",
    "city_enc", "traffic_level_enc", "weather_condition_enc",
    "avg_demand_requests", "avg_wait_time",
]

DRIVER_DELAY_FEATURES = [
    "acceptance_rate", "delay_rate", "avg_driver_rating", "avg_pickup_delay_min",
    "driver_experience_years", "ride_distance_km", "estimated_ride_time_min",
    "traffic_level_enc", "weather_condition_enc", "hour_of_day",
    "Rush_Hour_Flag", "avg_demand_requests", "avg_surge",
]


def _safe_features(df, feature_list):
    """Return only features that exist in df, filling missing with 0."""
    available = [f for f in feature_list if f in df.columns]
    missing = [f for f in feature_list if f not in df.columns]
    if missing:
        print(f"  ⚠ Missing features (will use 0): {missing}")
    result = df[available].copy()
    for m in missing:
        result[m] = 0
    return result[feature_list]


# ---------------------------------------------------------------------------
# Model 1 — Ride Outcome Prediction (Multi-Class)
# ---------------------------------------------------------------------------

def train_ride_outcome_model(master: pd.DataFrame) -> dict:
    """
    Predict booking_status: Completed (0), Cancelled (1), Incomplete (2).
    """
    print("\n[1] Training Ride Outcome Model (Multi-Class Classification)...")
    df = master.dropna(subset=["booking_status_encoded"])
    X = _safe_features(df, OUTCOME_FEATURES)
    y = df["booking_status_encoded"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    metrics = eval_classifier(y_test, y_pred, y_prob, "Ride Outcome Prediction")

    # Feature importance
    feat_imp = pd.Series(model.feature_importances_, index=OUTCOME_FEATURES).sort_values(ascending=False)
    print("\nTop 10 Feature Importances:")
    print(feat_imp.head(10).to_string())

    save_model(model, "ride_outcome_model")
    save_model(feat_imp, "ride_outcome_feature_importance")
    return {"model": model, "metrics": metrics, "feature_importance": feat_imp}


# ---------------------------------------------------------------------------
# Model 2 — Fare Prediction (Regression)
# ---------------------------------------------------------------------------

def train_fare_model(master: pd.DataFrame) -> dict:
    """
    Predict booking_value using GradientBoostingRegressor.
    """
    print("\n[2] Training Fare Prediction Model (Regression)...")
    df = master[master["booking_status"] == "Completed"].copy()
    df = df.dropna(subset=["booking_value"])
    X = _safe_features(df, FARE_FEATURES)
    y = df["booking_value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = eval_regressor(y_test, y_pred, "Fare Prediction Model")

    feat_imp = pd.Series(model.feature_importances_, index=FARE_FEATURES).sort_values(ascending=False)
    print("\nTop 10 Feature Importances:")
    print(feat_imp.head(10).to_string())

    save_model(model, "fare_prediction_model")
    save_model(feat_imp, "fare_feature_importance")
    return {"model": model, "metrics": metrics, "feature_importance": feat_imp}


# ---------------------------------------------------------------------------
# Model 3 — Customer Cancellation Risk (Binary)
# ---------------------------------------------------------------------------

def train_customer_cancel_model(master: pd.DataFrame) -> dict:
    """
    Predict customer_cancel_flag (1 = likely to cancel).
    """
    print("\n[3] Training Customer Cancellation Risk Model (Binary)...")
    df = master.dropna(subset=["customer_cancel_flag"])
    X = _safe_features(df, CANCEL_RISK_FEATURES)
    y = df["customer_cancel_flag"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150, max_depth=10, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = eval_classifier(y_test, y_pred, y_prob, "Customer Cancellation Risk")

    feat_imp = pd.Series(model.feature_importances_, index=CANCEL_RISK_FEATURES).sort_values(ascending=False)
    print("\nTop 10 Feature Importances:")
    print(feat_imp.head(10).to_string())

    save_model(model, "customer_cancel_model")
    save_model(feat_imp, "customer_cancel_feature_importance")
    return {"model": model, "metrics": metrics, "feature_importance": feat_imp}


# ---------------------------------------------------------------------------
# Model 4 — Driver Delay Prediction (Binary)
# ---------------------------------------------------------------------------

def train_driver_delay_model(master: pd.DataFrame) -> dict:
    """
    Predict driver_delay_flag (1 = likely to cause delay).
    """
    print("\n[4] Training Driver Delay Prediction Model (Binary)...")
    df = master.dropna(subset=["driver_delay_flag"])
    X = _safe_features(df, DRIVER_DELAY_FEATURES)
    y = df["driver_delay_flag"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150, max_depth=10, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = eval_classifier(y_test, y_pred, y_prob, "Driver Delay Prediction")

    feat_imp = pd.Series(model.feature_importances_, index=DRIVER_DELAY_FEATURES).sort_values(ascending=False)
    print("\nTop 10 Feature Importances:")
    print(feat_imp.head(10).to_string())

    save_model(model, "driver_delay_model")
    save_model(feat_imp, "driver_delay_feature_importance")
    return {"model": model, "metrics": metrics, "feature_importance": feat_imp}


# ---------------------------------------------------------------------------
# Train all
# ---------------------------------------------------------------------------

def train_all_models(master: pd.DataFrame) -> dict:
    """Train all four models and return results dict."""
    print("\n" + "=" * 60)
    print("   RAPIDO — MODEL TRAINING PIPELINE")
    print("=" * 60)
    results = {}
    results["ride_outcome"] = train_ride_outcome_model(master)
    results["fare"] = train_fare_model(master)
    results["customer_cancel"] = train_customer_cancel_model(master)
    results["driver_delay"] = train_driver_delay_model(master)
    print("\n✅ All models trained and saved to /models/")
    return results
