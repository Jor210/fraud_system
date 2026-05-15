import random
import pandas as pd
from datetime import datetime, timedelta

# =========================
# CONFIG
# =========================

NUM_TRANSACTIONS = 50000
FRAUD_RATIO = 0.03

OUTPUT_FILE = "ml/synthetic_transactions.csv"

# =========================
# HELPERS
# =========================

MERCHANT_CATEGORIES = [
    "electronics",
    "food",
    "gaming",
    "travel",
    "fashion",
    "crypto",
    "services"
]

COUNTRIES = [
    "RU",
    "US",
    "DE",
    "NL",
    "CN",
    "TR"
]

DEVICE_TYPES = [
    "mobile",
    "desktop",
    "tablet"
]

# =========================
# GENERATION
# =========================

transactions = []

base_date = datetime.now()

for transaction_id in range(NUM_TRANSACTIONS):

    # -------------------------
    # FRAUD OR NORMAL
    # -------------------------

    is_fraud = random.random() < FRAUD_RATIO

    # -------------------------
    # NORMAL TRANSACTIONS
    # -------------------------

    if not is_fraud:

        amount = round(random.uniform(100, 5000), 2)

        hour = random.randint(8, 22)

        velocity_1h = random.randint(1, 3)

        new_receiver = random.choices(
            [0, 1],
            weights=[90, 10]
        )[0]

        device_changed = random.choices(
            [0, 1],
            weights=[95, 5]
        )[0]

        failed_attempts = random.randint(0, 1)

        country_risk = round(random.uniform(0.0, 0.3), 2)

        amount_deviation = round(random.uniform(0.0, 0.4), 2)

    # -------------------------
    # FRAUD TRANSACTIONS
    # -------------------------

    else:

        fraud_type = random.choice([
            "account_takeover",
            "structuring",
            "suspicious_transfer"
        ])

        # ACCOUNT TAKEOVER
        if fraud_type == "account_takeover":

            amount = round(random.uniform(5000, 30000), 2)

            hour = random.randint(0, 5)

            velocity_1h = random.randint(5, 15)

            new_receiver = 1

            device_changed = 1

            failed_attempts = random.randint(2, 8)

            country_risk = round(random.uniform(0.6, 1.0), 2)

            amount_deviation = round(random.uniform(0.5, 1.0), 2)

        # STRUCTURING
        elif fraud_type == "structuring":

            amount = round(random.uniform(8000, 9900), 2)

            hour = random.randint(6, 23)

            velocity_1h = random.randint(10, 25)

            new_receiver = random.choice([0, 1])

            device_changed = random.choice([0, 1])

            failed_attempts = random.randint(0, 2)

            country_risk = round(random.uniform(0.3, 0.8), 2)

            amount_deviation = round(random.uniform(0.4, 0.9), 2)

        # SUSPICIOUS TRANSFER
        else:

            amount = round(random.uniform(15000, 50000), 2)

            hour = random.randint(0, 4)

            velocity_1h = random.randint(3, 10)

            new_receiver = 1

            device_changed = random.choice([0, 1])

            failed_attempts = random.randint(1, 5)

            country_risk = round(random.uniform(0.5, 1.0), 2)

            amount_deviation = round(random.uniform(0.6, 1.0), 2)

    # -------------------------
    # COMMON FIELDS
    # -------------------------

    timestamp = base_date - timedelta(
        minutes=random.randint(0, 60 * 24 * 30)
    )

    transaction = {
        "transaction_id": transaction_id,

        "amount": amount,

        "hour": hour,

        "velocity_1h": velocity_1h,

        "new_receiver": new_receiver,

        "device_changed": device_changed,

        "failed_attempts": failed_attempts,

        "country_risk": country_risk,

        "amount_deviation": amount_deviation,

        "merchant_category": random.choice(MERCHANT_CATEGORIES),

        "country": random.choice(COUNTRIES),

        "device_type": random.choice(DEVICE_TYPES),

        "timestamp": timestamp,

        "is_fraud": int(is_fraud)
    }

    transactions.append(transaction)

# =========================
# DATAFRAME
# =========================

df = pd.DataFrame(transactions)

# =========================
# SAVE CSV
# =========================

df.to_csv(OUTPUT_FILE, index=False)

print("Dataset generated successfully!")
print(f"Saved to: {OUTPUT_FILE}")

print("\nFraud distribution:")
print(df["is_fraud"].value_counts())