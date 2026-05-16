# dataset.py
import random
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# =========================
# CONFIG
# =========================
NUM_TRANSACTIONS = 50000
FRAUD_RATIO = 0.035  # чуть повысили, чтобы модель лучше училась

OUTPUT_FILE = "ml/synthetic_transactions.csv"

# =========================
# CONSTANTS
# =========================
MERCHANT_CATEGORIES = ["electronics", "food", "gaming", "travel", "fashion", "services", "raw_materials", "equipment"]
COUNTRIES = ["RU", "US", "DE", "NL", "CN", "TR", "BY", "KZ"]
DEVICE_TYPES = ["desktop", "mobile", "tablet", "api"]


# =========================
# HELPER FUNCTIONS
# =========================
def generate_timestamp(base_date):
    days_back = random.randint(0, 60)
    hours_back = random.randint(0, 23)
    minutes_back = random.randint(0, 59)
    return base_date - timedelta(days=days_back, hours=hours_back, minutes=minutes_back)


def calculate_amount_deviation(amount, avg_amount_30d):
    if avg_amount_30d == 0:
        return 0.0
    return round(amount / avg_amount_30d, 2)


# =========================
# MAIN GENERATION
# =========================
random.seed(42)
np.random.seed(42)

transactions = []
base_date = datetime.now()

for i in range(NUM_TRANSACTIONS):
    is_fraud = random.random() < FRAUD_RATIO

    # --- NORMAL BEHAVIOR ---
    if not is_fraud:
        amount = round(random.uniform(500, 45000), 2)
        velocity_1h = random.randint(0, 5)
        avg_amount_30d = round(random.uniform(amount * 0.6, amount * 1.4), 2)
        new_merchant = random.random() < 0.12
        device_type = random.choice(DEVICE_TYPES)
        merchant_category = random.choice(MERCHANT_CATEGORIES)
        country = "RU" if random.random() < 0.75 else random.choice(COUNTRIES)

        # время в основном рабочее
        hour = random.choices([8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
                              weights=[5, 8, 10, 12, 15, 18, 20, 22, 25, 22, 18, 15, 10, 7])[0]

        amount_deviation = calculate_amount_deviation(amount, avg_amount_30d)

    # --- FRAUD BEHAVIOR ---
    else:
        fraud_type = random.choice(["account_takeover", "substitution", "structuring", "unusual_time"])

        if fraud_type == "account_takeover":
            amount = round(random.uniform(80000, 450000), 2)
            velocity_1h = random.randint(6, 18)
            new_merchant = True
            avg_amount_30d = round(random.uniform(5000, 35000), 2)
            hour = random.randint(0, 5)
            amount_deviation = round(random.uniform(3.5, 12.0), 2)
            country = random.choice(COUNTRIES) if random.random() < 0.6 else "RU"

        elif fraud_type == "substitution":  # подмена реквизитов
            amount = round(random.uniform(45000, 280000), 2)
            velocity_1h = random.randint(3, 9)
            new_merchant = True
            avg_amount_30d = round(random.uniform(8000, 45000), 2)
            hour = random.randint(9, 22)
            amount_deviation = round(random.uniform(2.5, 8.0), 2)

        elif fraud_type == "structuring":  # дробление
            amount = round(random.uniform(8500, 14500), 2)
            velocity_1h = random.randint(4, 12)
            new_merchant = random.choice([True, False])
            avg_amount_30d = round(random.uniform(15000, 60000), 2)
            hour = random.randint(8, 21)
            amount_deviation = round(random.uniform(0.6, 1.4), 2)

        else:  # unusual_time + high amount
            amount = round(random.uniform(60000, 350000), 2)
            velocity_1h = random.randint(4, 10)
            new_merchant = random.random() < 0.7
            avg_amount_30d = round(random.uniform(10000, 50000), 2)
            hour = random.randint(0, 6)
            amount_deviation = round(random.uniform(2.8, 9.5), 2)

        device_type = random.choice(DEVICE_TYPES)
        merchant_category = random.choice(MERCHANT_CATEGORIES)
        country = random.choice(COUNTRIES)

    timestamp = generate_timestamp(base_date)

    transaction = {
        "transaction_id": f"TX{1000000 + i}",
        "account_id": f"ACC{random.randint(1000, 9999)}",
        "amount": amount,
        "currency": "RUB",
        "merchant_id": f"MERCH{random.randint(10000, 99999)}",
        "merchant_category": merchant_category,
        "device_type": device_type,
        "location": country,  # используем country как location
        "velocity_1h": velocity_1h,
        "avg_amount_30d": avg_amount_30d,
        "new_merchant": new_merchant,
        "hour": hour,
        "amount_deviation": amount_deviation,
        "timestamp": timestamp,
        "is_fraud": int(is_fraud)
    }

    transactions.append(transaction)

# =========================
# CREATE DATAFRAME & SAVE
# =========================
df = pd.DataFrame(transactions)

# Дополнительные полезные признаки
df['amount_to_avg_ratio'] = df['amount_deviation']

print("Dataset generated successfully!")
print(f"Total transactions: {len(df)}")
print(f"Fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].mean() * 100:.2f}%)")
print("\nFraud distribution:")
print(df['is_fraud'].value_counts())

df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved to: {OUTPUT_FILE}")