# 🛵 Rapido — Intelligent Mobility Insights
## Ride Patterns, Cancellations & Fare Forecasting

> **Domain:** Mobility & Transportation Analytics  
> **Stack:** Python · Scikit-Learn · Streamlit · Plotly  

---

## 📁 Project Structure

```
rapido_project/
├── data/                        # All 7 data files (5 CSV + 2 XLSX)
│   ├── bookings.csv             # Core transactional data (100,000 rows)
│   ├── customers.csv            # Customer behaviour signals (10,000 rows)
│   ├── drivers.csv              # Driver performance metrics (5,000 rows)
│   ├── location_demand.csv      # Demand patterns by location & time
│   ├── time_features.csv        # Temporal signals (hourly, 8,760 rows)
│   ├── time_features.xlsx       # Same — Excel version 1
│   └── time_features_v2.xlsx    # Same — Excel version 2
│
├── models/                      # Trained model .pkl files (auto-created)
│
├── utils/
│   ├── data_loader.py           # Load all 7 files (CSV + XLSX)
│   ├── preprocessing.py         # Step 1: Cleaning + Step 3: Feature Engineering
│   ├── model_training.py        # Step 4 & 5: Train & Evaluate all 4 models
│   └── eda_utils.py             # Step 2: EDA chart helpers (Plotly)
│
├── app/
│   └── streamlit_app.py         # Step 6: Full Streamlit Dashboard
│
├── train.py                     # Entry point — runs full ML pipeline
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

```bash
# 1. Clone / unzip the project
cd rapido_project

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Step 1 — Train all models
```bash
python train.py
```
This will:
- Load all 7 files (5 CSV + 2 XLSX)
- Clean and engineer features
- Train 4 ML models
- Save `.pkl` files to `/models/`

### Step 2 — Launch the dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## 🤖 ML Models

| # | Model | Type | Algorithm | Target |
|---|-------|------|-----------|--------|
| 1 | Ride Outcome Prediction | Multi-Class Classification | Random Forest | booking_status |
| 2 | Fare Prediction | Regression | Gradient Boosting | booking_value |
| 3 | Customer Cancel Risk | Binary Classification | Random Forest | customer_cancel_flag |
| 4 | Driver Delay Prediction | Binary Classification | Random Forest | driver_delay_flag |

### Evaluation Targets
- Classification Accuracy → **85–90%**
- Regression RMSE → **within ±10% of actual fare**

---

## 📊 Features Engineered

| Feature | Description |
|---------|-------------|
| `Fare_per_KM` | booking_value / ride_distance_km |
| `Fare_per_Min` | booking_value / actual_ride_time_min |
| `Rush_Hour_Flag` | 1 if hour ∈ {7,8,9,17,18,19,20} |
| `Long_Distance_Flag` | 1 if distance ≥ 75th percentile |
| `City_Pair` | pickup_location + "_" + drop_location |
| `Driver_Reliability_Score` | Composite score (0–10) |
| `Customer_Loyalty_Score` | Composite score (0–10) |

---

## 🖥️ Dashboard Tabs

| Tab | Content |
|-----|---------|
| 📊 Overview & EDA | KPI cards, ride volumes, cancellation heatmaps, rating distributions |
| 🔮 Predictions | Live inference for all 4 models with interactive inputs |
| 🗺 Demand & Locations | Demand heatmaps, surge patterns, temporal features |
| 📈 Model Performance | Feature importance charts, algorithm summary, benchmarks |
| 🔍 Data Explorer | Browse, filter, and download any of the 7 datasets |

---

## 🗄️ Data Files Loaded

| File | Format | Rows | Description |
|------|--------|------|-------------|
| bookings.csv | CSV | 100,000 | Core booking transactions |
| customers.csv | CSV | 10,000 | Customer behaviour |
| drivers.csv | CSV | 5,000 | Driver metrics |
| location_demand.csv | CSV | 17,941 | Demand by location |
| time_features.csv | CSV | 8,760 | Temporal signals |
| time_features.xlsx | XLSX | 8,760 | Same — Excel v1 |
| time_features_v2.xlsx | XLSX | 8,760 | Same — Excel v2 |

---

## 👥 Business Use Cases

1. **Reduce Cancellations by 20%** — Customer cancel risk model
2. **Improve ETA Accuracy** — Driver delay prediction
3. **Dynamic Pricing** — Fare prediction + demand patterns
4. **Driver Reliability Scoring** — Composite reliability score

---

## 📄 License
Internal project — GUVI × HCL Capstone
