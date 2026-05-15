import joblib
import pandas as pd

# =========================
# LOAD MODEL
# =========================

model = joblib.load("model.pkl")

encoders = joblib.load("encoders.pkl")

# =========================
# PREDICT FUNCTION
# =========================

def predict_transaction(transaction_data: dict):

    df = pd.DataFrame([transaction_data])

    # Encode categorical fields

    categorical_columns = [
        "merchant_category",
        "country",
        "device_type"
    ]

    for column in categorical_columns:

        encoder = encoders[column]

        df[column] = encoder.transform(df[column])

    # Prediction

    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0][1]

    return {
        "is_fraud": bool(prediction),
        "fraud_probability": round(float(probability), 4)
    }