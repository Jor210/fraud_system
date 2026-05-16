"""
Модуль предсказания мошенничества.
Используется из FastAPI (app/main.py).

Пример вызова:
    from ml.predictor import predict_transaction

    result = predict_transaction({
        "amount": 150000.0,
        "merchant_category": "Электроника",
        "device_type": "mobile",
        "velocity_1h": 8,
        "avg_amount_30d": 30000.0,
        "new_merchant": 1,
        "hour": 2,
        "is_night": 1,
        "amount_deviation": 4.0
    })
    # {"is_fraud": True, "fraud_probability": 0.87, "risk_level": "HIGH"}
"""

import os
import joblib
import pandas as pd

# ─────────────────────────────────────────────
# ПУТИ — всегда относительно этого файла
# ─────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "model.pkl")
ENC_PATH    = os.path.join(BASE_DIR, "encoders.pkl")
FEAT_PATH   = os.path.join(BASE_DIR, "feature_names.pkl")

# ─────────────────────────────────────────────
# ЗАГРУЗКА МОДЕЛИ (один раз при импорте)
# ─────────────────────────────────────────────

model         = joblib.load(MODEL_PATH)
encoders      = joblib.load(ENC_PATH)
feature_names = joblib.load(FEAT_PATH)

# ─────────────────────────────────────────────
# ПРЕДСКАЗАНИЕ
# ─────────────────────────────────────────────

def predict_transaction(transaction_data: dict) -> dict:
    """
    Принимает словарь с признаками транзакции,
    возвращает результат классификации.

    Args:
        transaction_data: dict с ключами из feature_names

    Returns:
        {
            "is_fraud": bool,
            "fraud_probability": float,  # 0.0 – 1.0
            "risk_level": str            # LOW / MEDIUM / HIGH
        }
    """
    df = pd.DataFrame([transaction_data])

    # Заполняем пропуски на случай отсутствия поля
    df["avg_amount_30d"]   = df.get("avg_amount_30d",   pd.Series([0.0])).fillna(0.0)
    df["amount_deviation"] = df.get("amount_deviation", pd.Series([0.0])).fillna(0.0)

    # Кодируем категориальные признаки
    CATEGORICAL = ["merchant_category", "device_type"]
    for col in CATEGORICAL:
        if col in df.columns and col in encoders:
            enc = encoders[col]
            # Если пришло неизвестное значение — заменяем на первый известный класс
            df[col] = df[col].apply(
                lambda x: x if x in enc.classes_ else enc.classes_[0]
            )
            df[col] = enc.transform(df[col])

    # Оставляем только нужные признаки в правильном порядке
    df = df[feature_names]

    probability = float(model.predict_proba(df)[0][1])
    is_fraud    = probability >= 0.5

    # Уровень риска для отображения в дашборде
    if probability < 0.3:
        risk_level = "LOW"
    elif probability < 0.6:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    return {
        "is_fraud":          is_fraud,
        "fraud_probability": round(probability, 4),
        "risk_level":        risk_level,
    }