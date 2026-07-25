"""
Synthetic data generator for a small "customers / orders / marketing spend /
support tickets" analytics schema, used to exercise SQL and BI tooling
against realistic-but-fake data.

Generates 4 related CSVs under ./data and loads them into a SQLite database
at ./db/analytics.db.

DELIBERATE DATA QUALITY ISSUES (do not "fix" these — they exist on purpose
so downstream tooling can be tested against messy real-world data):

1. customers.region casing/format inconsistency
   ~30% of rows independently get a "messy" variant of their region
   ("west", "WEST", "W" instead of "West"). This simulates inconsistent
   data entry across source systems and is meant to break naive GROUP BY
   region queries unless the value is normalized first.

2. orders.channel nulls
   Exactly 15% of order rows have channel = NULL, simulating untracked /
   unattributed orders (e.g. direct traffic with no UTM tagging). Any
   channel-level revenue rollup must explicitly decide how to handle
   these nulls (drop, bucket as "Unknown", etc.) rather than assuming
   channel is always populated.

3. Granularity mismatch between orders and marketing_spend
   orders is DAILY grain (one row per order, with a full order_date).
   marketing_spend is MONTHLY grain (one row per channel per month).
   These are intentionally NOT pre-aggregated or aligned to the same
   grain — joining them for e.g. ROI analysis requires explicitly
   rolling orders up to month first. Do not "fix" this by adding a
   daily marketing_spend or monthly orders table.

Run with: python generate_synthetic_data.py
"""

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
Faker.seed(SEED)
fake = Faker()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = BASE_DIR / "db" / "analytics.db"

N_CUSTOMERS = 800
N_ORDERS = 4000
N_SUPPORT_TICKETS = 600

CHANNELS = ["Paid Search", "Organic", "Referral", "Email"]
REGIONS = ["West", "East", "North", "South"]
PLAN_TIERS = ["Free", "Basic", "Pro", "Enterprise"]
PLAN_TIER_WEIGHTS = [0.55, 0.28, 0.13, 0.04]
TICKET_STATUSES = ["Open", "Closed", "Pending"]

SIGNUP_START = date(2023, 1, 1)
SIGNUP_END = date(2025, 12, 31)
DATA_END = date(2025, 12, 31)

REGION_ABBREV = {"West": "W", "East": "E", "North": "N", "South": "S"}


def random_date(start: date, end: date) -> date:
    delta_days = (end - start).days
    return start + timedelta(days=random.randint(0, delta_days))


def messy_region(region: str) -> str:
    """Return region unchanged 70% of the time; a randomly-cased/abbreviated
    variant the other 30%, independently per call (see module docstring)."""
    if random.random() >= 0.30:
        return region
    variant_kind = random.choice(["lower", "upper", "abbrev"])
    if variant_kind == "lower":
        return region.lower()
    if variant_kind == "upper":
        return region.upper()
    return REGION_ABBREV[region]


def generate_customers() -> pd.DataFrame:
    rows = []
    for customer_id in range(1, N_CUSTOMERS + 1):
        signup_date = random_date(SIGNUP_START, SIGNUP_END)
        region = messy_region(random.choice(REGIONS))
        plan_tier = np.random.choice(PLAN_TIERS, p=PLAN_TIER_WEIGHTS)
        rows.append(
            {
                "customer_id": customer_id,
                "signup_date": signup_date.isoformat(),
                "region": region,
                "plan_tier": plan_tier,
            }
        )
    return pd.DataFrame(rows)


def generate_orders(customers: pd.DataFrame) -> pd.DataFrame:
    signup_by_customer = dict(
        zip(customers["customer_id"], pd.to_datetime(customers["signup_date"]).dt.date)
    )
    customer_ids = customers["customer_id"].tolist()

    rows = []
    for order_id in range(1, N_ORDERS + 1):
        customer_id = random.choice(customer_ids)
        signup_date = signup_by_customer[customer_id]
        order_date = random_date(signup_date, DATA_END)
        revenue = float(np.clip(np.random.normal(loc=80, scale=60), 10, 500))
        channel = random.choice(CHANNELS)
        rows.append(
            {
                "order_id": order_id,
                "customer_id": customer_id,
                "order_date": order_date.isoformat(),
                "revenue": round(revenue, 2),
                "channel": channel,
            }
        )

    orders = pd.DataFrame(rows)

    null_count = int(round(len(orders) * 0.15))
    null_indices = np.random.choice(orders.index, size=null_count, replace=False)
    orders.loc[null_indices, "channel"] = None

    return orders


def generate_marketing_spend() -> pd.DataFrame:
    months = pd.period_range("2023-01", "2025-12", freq="M").astype(str).tolist()

    rows = []
    spend_id = 1
    for month in months:
        for channel in CHANNELS:
            rows.append(
                {
                    "spend_id": spend_id,
                    "month": month,
                    "channel": channel,
                    "amount_spent": round(random.uniform(1000, 20000), 2),
                }
            )
            spend_id += 1
    return pd.DataFrame(rows)


def generate_support_tickets(customers: pd.DataFrame) -> pd.DataFrame:
    signup_by_customer = dict(
        zip(customers["customer_id"], pd.to_datetime(customers["signup_date"]).dt.date)
    )
    customer_ids = customers["customer_id"].tolist()

    rows = []
    for ticket_id in range(1, N_SUPPORT_TICKETS + 1):
        customer_id = random.choice(customer_ids)
        signup_date = signup_by_customer[customer_id]
        created_at = random_date(signup_date, DATA_END)
        status = random.choice(TICKET_STATUSES)
        rows.append(
            {
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "created_at": created_at.isoformat(),
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def save_csvs(tables: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(DATA_DIR / f"{name}.csv", index=False)


def load_into_sqlite() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            signup_date DATE NOT NULL,
            region TEXT NOT NULL,
            plan_tier TEXT NOT NULL
        );

        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
            order_date DATE NOT NULL,
            revenue REAL NOT NULL,
            channel TEXT
        );

        CREATE TABLE marketing_spend (
            spend_id INTEGER PRIMARY KEY,
            month TEXT NOT NULL,
            channel TEXT NOT NULL,
            amount_spent REAL NOT NULL
        );

        CREATE TABLE support_tickets (
            ticket_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
            created_at DATE NOT NULL,
            status TEXT NOT NULL
        );
        """
    )
    conn.commit()

    for table_name in ["customers", "orders", "marketing_spend", "support_tickets"]:
        df = pd.read_csv(DATA_DIR / f"{table_name}.csv")
        df.to_sql(table_name, conn, if_exists="append", index=False)

    conn.commit()
    conn.close()


def print_verification() -> None:
    conn = sqlite3.connect(DB_PATH)
    for table_name in ["customers", "orders", "marketing_spend", "support_tickets"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"\n=== {table_name}: {count} rows ===")
        sample = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 5", conn)
        print(sample.to_string(index=False))
    conn.close()


def main() -> None:
    customers = generate_customers()
    orders = generate_orders(customers)
    marketing_spend = generate_marketing_spend()
    support_tickets = generate_support_tickets(customers)

    save_csvs(
        {
            "customers": customers,
            "orders": orders,
            "marketing_spend": marketing_spend,
            "support_tickets": support_tickets,
        }
    )

    load_into_sqlite()
    print_verification()


if __name__ == "__main__":
    main()
