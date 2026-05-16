# ml/predictor.py
import joblib
import pandas as pd
import os

model = None
encoders = None


def load_model():
    global model, encoders
    if model is None:
        model_path = "ml/model.pkl"
        encoders_path = "ml/encoders.pkl"

        if not os.path.exists(model_path):
            raise FileNotFoundError("Модель не найдена!")

        model = joblib.load(model_path)
        encoders = joblib.load(encoders_path)
        print("✅ ML-модель загружена")


def predict_transaction(transaction: dict) -> dict:
    load_model()

    data = pd.DataFrame([transaction])

    # === Безопасное добавление отсутствующих признаков ===
    if 'hour' not in data.columns:
        data['hour'] = 14  # дефолт — день

    if 'amount_deviation' not in data.columns:
        avg = data.get('avg_amount_30d', 20000.0)
        data['amount_deviation'] = round(data['amount'] / avg.iloc[0] if len(avg) > 0 else 1.0, 2)

    data['amount_to_avg_ratio'] = data['amount_deviation']
    data['velocity_per_hour_score'] = data['velocity_1h'] * (data['amount'] / 10000)

    categorical_features = ['merchant_category', 'device_type', 'location']
    numerical_features = [
        'amount', 'velocity_1h', 'avg_amount_30d', 'new_merchant',
        'hour', 'amount_deviation', 'amount_to_avg_ratio', 'velocity_per_hour_score'
    ]

    # Кодирование с обработкой ошибок
    for col in categorical_features:
        if col in data.columns and col in encoders:
            try:
                data[col] = encoders[col].transform(data[col].astype(str))
            except:
                data[col] = 0

    X = data[categorical_features + numerical_features]

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    return {
        "is_fraud": bool(prediction),
        "fraud_probability": round(float(probability), 4),
        "risk_score_ml": round(float(probability), 4)
    }