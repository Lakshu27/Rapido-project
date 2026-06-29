"""
sql_manager.py
--------------
Data Management using LOCAL MySQL.

Project spec: "Data Management using SQL"
- Connects to local MySQL server
- Creates rapido database + 5 normalized tables with indexes
- Loads all 5 CSV files into MySQL
- Provides analytical query functions used by Streamlit app

Usage:
    python utils/sql_manager.py        # creates DB + loads all data
    from utils.sql_manager import get_connection, run_query
"""

import os
import mysql.connector
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# ── Load .env credentials ─────────────────────────────────────────────────
load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# ── MySQL Config (reads from .env or falls back to defaults) ──────────────
MYSQL_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", 3306)),
    "user":     os.getenv("DB_USER",     "root"),
    "password": os.getenv("DB_PASSWORD", "Lakshu@27"),
    "database": os.getenv("DB_NAME",     "rapido_db"),
    "charset":  "utf8mb4",
    "autocommit": True,
}

DB_NAME = MYSQL_CONFIG["database"]


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def get_connection(with_db: bool = True) -> mysql.connector.MySQLConnection:
    """
    Return a MySQL connection.

    Parameters
    ----------
    with_db : bool — if False, connects without selecting a database
                     (used during initial DB creation)
    """
    try:
        cfg = MYSQL_CONFIG.copy()
        if not with_db:
            cfg.pop("database", None)
        conn = mysql.connector.connect(**cfg)
        return conn
    except mysql.connector.Error as e:
        raise ConnectionError(f"MySQL connection failed: {e}\n"
                              f"Check your .env file or MYSQL_CONFIG in sql_manager.py")


def run_query(sql: str, params=None) -> pd.DataFrame:
    """
    Execute a SELECT query and return a DataFrame.

    Parameters
    ----------
    sql    : str   — SQL query
    params : tuple — optional bind parameters (%s placeholders)
    """
    try:
        conn = get_connection()
        df = pd.read_sql(sql, conn, params=params)
        conn.close()
        return df
    except mysql.connector.Error as e:
        print(f"SQL Error: {e}")
        return pd.DataFrame()


def execute_ddl(sql: str, conn) -> None:
    """Execute a DDL/DML statement (no result set)."""
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        cursor.close()
    except mysql.connector.Error as e:
        print(f"  ⚠ DDL Error: {e}")


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

DDL_CREATE_DB = f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"

DDL_TABLES = {
    "bookings": """
        CREATE TABLE IF NOT EXISTS bookings (
            booking_id              VARCHAR(20)  NOT NULL,
            booking_date            DATE,
            booking_time            TIME,
            day_of_week             VARCHAR(10),
            is_weekend              TINYINT(1),
            hour_of_day             TINYINT,
            city                    VARCHAR(50),
            pickup_location         VARCHAR(20),
            drop_location           VARCHAR(20),
            vehicle_type            VARCHAR(20),
            ride_distance_km        FLOAT,
            estimated_ride_time_min FLOAT,
            actual_ride_time_min    FLOAT,
            traffic_level           VARCHAR(10),
            weather_condition       VARCHAR(20),
            base_fare               FLOAT,
            surge_multiplier        FLOAT,
            booking_value           FLOAT,
            booking_status          VARCHAR(15),
            incomplete_ride_reason  VARCHAR(100),
            customer_id             VARCHAR(15),
            driver_id               VARCHAR(15),
            PRIMARY KEY (booking_id),
            INDEX idx_city          (city),
            INDEX idx_status        (booking_status),
            INDEX idx_date          (booking_date),
            INDEX idx_customer      (customer_id),
            INDEX idx_driver        (driver_id),
            INDEX idx_hour          (hour_of_day),
            INDEX idx_vehicle       (vehicle_type)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,

    "customers": """
        CREATE TABLE IF NOT EXISTS customers (
            customer_id              VARCHAR(15)  NOT NULL,
            customer_gender          VARCHAR(15),
            customer_age             TINYINT,
            customer_city            VARCHAR(50),
            customer_signup_days_ago INT,
            preferred_vehicle_type   VARCHAR(20),
            total_bookings           INT,
            completed_rides          INT,
            cancelled_rides          INT,
            incomplete_rides         INT,
            cancellation_rate        FLOAT,
            avg_customer_rating      FLOAT,
            customer_cancel_flag     TINYINT(1),
            PRIMARY KEY (customer_id),
            INDEX idx_city           (customer_city),
            INDEX idx_cancel_flag    (customer_cancel_flag)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,

    "drivers": """
        CREATE TABLE IF NOT EXISTS drivers (
            driver_id               VARCHAR(15)  NOT NULL,
            driver_age              TINYINT,
            driver_city             VARCHAR(50),
            vehicle_type            VARCHAR(20),
            driver_experience_years TINYINT,
            total_assigned_rides    INT,
            accepted_rides          INT,
            incomplete_rides        INT,
            delay_count             INT,
            acceptance_rate         FLOAT,
            delay_rate              FLOAT,
            avg_driver_rating       FLOAT,
            avg_pickup_delay_min    FLOAT,
            driver_delay_flag       TINYINT(1),
            PRIMARY KEY (driver_id),
            INDEX idx_city          (driver_city),
            INDEX idx_delay_flag    (driver_delay_flag)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,

    "location_demand": """
        CREATE TABLE IF NOT EXISTS location_demand (
            id                   INT          NOT NULL AUTO_INCREMENT,
            city                 VARCHAR(50),
            pickup_location      VARCHAR(20),
            hour_of_day          TINYINT,
            vehicle_type         VARCHAR(20),
            total_requests       INT,
            completed_rides      INT,
            cancelled_rides      INT,
            avg_wait_time_min    FLOAT,
            avg_surge_multiplier FLOAT,
            demand_level         VARCHAR(10),
            PRIMARY KEY (id),
            INDEX idx_city       (city),
            INDEX idx_hour       (hour_of_day)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,

    "time_features": """
        CREATE TABLE IF NOT EXISTS time_features (
            datetime       VARCHAR(25)  NOT NULL,
            hour_of_day    TINYINT,
            day_of_week    VARCHAR(10),
            is_weekend     TINYINT(1),
            is_holiday     TINYINT(1),
            peak_time_flag TINYINT(1),
            season         VARCHAR(10),
            PRIMARY KEY (datetime),
            INDEX idx_hour (hour_of_day)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
}


# ---------------------------------------------------------------------------
# Build database
# ---------------------------------------------------------------------------

def build_database(force: bool = False) -> None:
    """
    Create MySQL database + tables + load all 5 CSV files.

    Parameters
    ----------
    force : bool — if True, DROP and recreate all tables
    """
    print(f"Connecting to MySQL at {MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']} "
          f"as '{MYSQL_CONFIG['user']}'...")

    # Step 1: Create database if not exists
    conn_no_db = get_connection(with_db=False)
    execute_ddl(DDL_CREATE_DB, conn_no_db)
    execute_ddl(f"USE `{DB_NAME}`", conn_no_db)
    print(f"  ✔ Database `{DB_NAME}` ready")

    # Step 2: Create tables
    if force:
        for table in reversed(list(DDL_TABLES.keys())):
            execute_ddl(f"DROP TABLE IF EXISTS `{table}`", conn_no_db)
        print("  ✔ Old tables dropped")

    for table, ddl in DDL_TABLES.items():
        execute_ddl(ddl, conn_no_db)
        print(f"  ✔ Table `{table}` ready")

    conn_no_db.close()

    # Step 3: Load CSV files into MySQL
    csv_table_map = {
        "bookings.csv":        "bookings",
        "customers.csv":       "customers",
        "drivers.csv":         "drivers",
        "location_demand.csv": "location_demand",
        "time_features.csv":   "time_features",
    }

    conn = get_connection()
    cursor = conn.cursor()

    def nan_to_none(val):
        """Convert NaN / numpy nan / float nan → None for MySQL NULL."""
        if val is None:
            return None
        try:
            import math
            if isinstance(val, float) and (math.isnan(val) or val != val):
                return None
        except Exception:
            pass
        return val

    for filename, table in csv_table_map.items():
        path = os.path.join(DATA_DIR, filename)
        df = pd.read_csv(path)

        # Convert to object dtype first, then replace NaN with None
        df = df.astype(object).where(pd.notnull(df), None)

        # Check if already loaded
        cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
        existing = cursor.fetchone()[0]
        if existing > 0 and not force:
            print(f"  ⏭ `{table}` already has {existing:,} rows — skipping")
            continue

        # Truncate + reload
        cursor.execute(f"TRUNCATE TABLE `{table}`")

        # Batch insert (1000 rows at a time for performance)
        cols = ", ".join([f"`{c}`" for c in df.columns])
        placeholders = ", ".join(["%s"] * len(df.columns))
        insert_sql = f"INSERT INTO `{table}` ({cols}) VALUES ({placeholders})"

        batch_size = 1000
        total = len(df)
        for i in range(0, total, batch_size):
            batch = df.iloc[i:i + batch_size]
            # Apply nan_to_none on every cell to guarantee no NaN leaks into MySQL
            rows = [
                tuple(nan_to_none(v) for v in r)
                for r in batch.itertuples(index=False, name=None)
            ]
            cursor.executemany(insert_sql, rows)
        conn.commit()
        print(f"  ✔ Loaded {filename} → `{table}` ({total:,} rows)")

    cursor.close()
    conn.close()
    print(f"\n✅ MySQL database `{DB_NAME}` is ready!")


# ---------------------------------------------------------------------------
# Analytical queries
# ---------------------------------------------------------------------------

def query_booking_summary() -> pd.DataFrame:
    return run_query("""
        SELECT
            COUNT(*)                                                              AS total_bookings,
            SUM(booking_status = 'Completed')                                    AS completed,
            SUM(booking_status = 'Cancelled')                                    AS cancelled,
            SUM(booking_status = 'Incomplete')                                   AS incomplete,
            ROUND(AVG(CASE WHEN booking_status='Completed' THEN booking_value END), 2) AS avg_fare,
            ROUND(AVG(ride_distance_km), 2)                                      AS avg_distance_km,
            ROUND(AVG(surge_multiplier), 3)                                      AS avg_surge
        FROM bookings
    """)


def query_ride_volume_by_hour() -> pd.DataFrame:
    return run_query("""
        SELECT hour_of_day,
               COUNT(*)                          AS total_bookings,
               SUM(booking_status='Completed')   AS completed,
               SUM(booking_status='Cancelled')   AS cancelled
        FROM   bookings
        GROUP  BY hour_of_day
        ORDER  BY hour_of_day
    """)


def query_cancellation_by_city() -> pd.DataFrame:
    return run_query("""
        SELECT city,
               COUNT(*)                                              AS total,
               SUM(booking_status = 'Cancelled')                    AS cancelled,
               ROUND(100.0 * SUM(booking_status='Cancelled') / COUNT(*), 2) AS cancel_rate_pct
        FROM   bookings
        GROUP  BY city
        ORDER  BY cancel_rate_pct DESC
    """)


def query_fare_by_vehicle() -> pd.DataFrame:
    return run_query("""
        SELECT vehicle_type,
               ROUND(AVG(booking_value), 2)    AS avg_fare,
               ROUND(AVG(ride_distance_km), 2) AS avg_distance_km,
               ROUND(AVG(surge_multiplier), 3) AS avg_surge,
               COUNT(*)                        AS total_rides
        FROM   bookings
        WHERE  booking_status = 'Completed'
        GROUP  BY vehicle_type
        ORDER  BY avg_fare DESC
    """)


def query_peak_cancellation_windows() -> pd.DataFrame:
    return run_query("""
        SELECT city, hour_of_day, day_of_week,
               COUNT(*)                                                      AS total,
               SUM(booking_status='Cancelled')                               AS cancelled,
               ROUND(100.0 * SUM(booking_status='Cancelled') / COUNT(*), 2) AS cancel_rate_pct
        FROM   bookings
        GROUP  BY city, hour_of_day, day_of_week
        HAVING total >= 10
        ORDER  BY cancel_rate_pct DESC
        LIMIT  20
    """)


def query_high_risk_customers(limit: int = 20) -> pd.DataFrame:
    return run_query("""
        SELECT c.customer_id, c.customer_city,
               c.cancellation_rate, c.avg_customer_rating,
               c.total_bookings, c.customer_cancel_flag,
               COUNT(b.booking_id) AS recent_bookings
        FROM   customers c
        LEFT JOIN bookings b ON c.customer_id = b.customer_id
        WHERE  c.customer_cancel_flag = 1
        GROUP  BY c.customer_id, c.customer_city,
                  c.cancellation_rate, c.avg_customer_rating,
                  c.total_bookings, c.customer_cancel_flag
        ORDER  BY c.cancellation_rate DESC
        LIMIT  %s
    """, params=(limit,))


def query_driver_reliability() -> pd.DataFrame:
    return run_query("""
        SELECT driver_id, driver_city, vehicle_type,
               acceptance_rate, delay_rate,
               avg_driver_rating, avg_pickup_delay_min,
               driver_delay_flag,
               ROUND(
                   acceptance_rate * avg_driver_rating * 2
                   - delay_rate * 5
                   - avg_pickup_delay_min * 0.05,
               3) AS reliability_score
        FROM   drivers
        ORDER  BY reliability_score DESC
        LIMIT  50
    """)


def query_surge_demand_pattern() -> pd.DataFrame:
    return run_query("""
        SELECT city, vehicle_type, demand_level,
               ROUND(AVG(avg_surge_multiplier), 3) AS avg_surge,
               ROUND(AVG(avg_wait_time_min), 2)    AS avg_wait_min,
               SUM(total_requests)                  AS total_requests
        FROM   location_demand
        GROUP  BY city, vehicle_type, demand_level
        ORDER  BY avg_surge DESC
    """)


def query_traffic_weather_impact() -> pd.DataFrame:
    return run_query("""
        SELECT traffic_level, weather_condition,
               COUNT(*)                                                      AS total,
               SUM(booking_status='Cancelled')                               AS cancelled,
               ROUND(100.0 * SUM(booking_status='Cancelled') / COUNT(*), 2) AS cancel_rate_pct,
               ROUND(AVG(booking_value), 2)                                  AS avg_fare
        FROM   bookings
        GROUP  BY traffic_level, weather_condition
        ORDER  BY cancel_rate_pct DESC
    """)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  RAPIDO — MySQL Database Builder")
    print("=" * 60)
    build_database(force=True)

    print("\n── Sample Queries ─────────────────────────────────────────")
    print("\n[1] Booking Summary KPIs:")
    print(query_booking_summary().to_string(index=False))

    print("\n[2] Cancellation Rate by City:")
    print(query_cancellation_by_city().to_string(index=False))

    print("\n[3] Avg Fare by Vehicle Type:")
    print(query_fare_by_vehicle().to_string(index=False))

    print("\n[4] Top Peak Cancellation Windows:")
    print(query_peak_cancellation_windows().head(5).to_string(index=False))

    print("\n[5] Traffic & Weather Impact:")
    print(query_traffic_weather_impact().to_string(index=False))

    print("\n✅ MySQL module verified.")