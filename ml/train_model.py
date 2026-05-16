"""
Обучение модели обнаружения мошенничества.

Запуск (из папки ml/):
    python train_model.py

Требует: synthetic_transactions.csv в той же папке.
Сохраняет: model.pkl, encoders.pkl, feature_names.pkl
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ─────────────────────────────────────────────
# ПУТИ — всегда относительно этого файла
# ─────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "synthetic_transactions.csv")
MODEL_PATH  = os.path.join(BASE_DIR, "model.pkl")
ENC_PATH    = os.path.join(BASE_DIR, "encoders.pkl")
FEAT_PATH   = os.path.join(BASE_DIR, "feature_names.pkl")

# ─────────────────────────────────────────────
# ЗАГРУЗКА
# ─────────────────────────────────────────────

print("Загружаю данные...")
df = pd.read_csv(DATA_PATH)
print(f"  Строк: {len(df):,}  |  Мошеннических: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.1f}%)")

# ─────────────────────────────────────────────
# ПОДГОТОВКА
# ─────────────────────────────────────────────

# Убираем колонки, которые не должны попасть в модель
DROP_COLS = ["transaction_id", "timestamp", "fraud_type"]
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

# Заполняем пропуски (намеренный шум из генератора)
df["avg_amount_30d"]   = df["avg_amount_30d"].fillna(df["avg_amount_30d"].median())
df["amount_deviation"] = df["amount_deviation"].fillna(0.0)

# ─────────────────────────────────────────────
# КОДИРОВАНИЕ КАТЕГОРИЙ
# ─────────────────────────────────────────────

CATEGORICAL = ["merchant_category", "device_type"]
encoders = {}

for col in CATEGORICAL:
    enc = LabelEncoder()
    df[col] = enc.fit_transform(df[col])
    encoders[col] = enc
    print(f"  Закодировано '{col}': {list(enc.classes_)}")

# ─────────────────────────────────────────────
# ПРИЗНАКИ И ЦЕЛЕВАЯ ПЕРЕМЕННАЯ
# ─────────────────────────────────────────────

FEATURE_COLS = [
    "amount",
    "merchant_category",
    "device_type",
    "velocity_1h",
    "avg_amount_30d",
    "new_merchant",
    "hour",
    "is_night",
    "amount_deviation",
]

X = df[FEATURE_COLS]
y = df["is_fraud"]

print(f"\nПризнаков: {len(FEATURE_COLS)}  →  {FEATURE_COLS}")

# ─────────────────────────────────────────────
# РАЗБИВКА
# ─────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # сохраняем пропорцию 3% в обоих сплитах
)

print(f"\nОбучающая выборка: {len(X_train):,}  |  Тестовая: {len(X_test):,}")

# ─────────────────────────────────────────────
# МОДЕЛЬ
# ─────────────────────────────────────────────

model = RandomForestClassifier(
    n_estimators=200,       # больше деревьев — стабильнее на дисбалансе
    max_depth=14,
    min_samples_leaf=10,    # не переобучаться на редких мошеннических случаях
    class_weight="balanced",# автоматически компенсирует дисбаланс 3%/97%
    random_state=42,
    n_jobs=-1               # использовать все ядра
)

print("\nОбучаю модель...")
model.fit(X_train, y_train)

# ─────────────────────────────────────────────
# МЕТРИКИ
# ─────────────────────────────────────────────

y_pred      = model.predict(X_test)
y_proba     = model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 50)
print("РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ")
print("=" * 50)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Легитимная", "Мошенническая"]))

print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

print("\nВажность признаков:")
importance_df = pd.DataFrame({
    "feature":    FEATURE_COLS,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)
print(importance_df.to_string(index=False))

# ─────────────────────────────────────────────
# СОХРАНЕНИЕ
# ─────────────────────────────────────────────

joblib.dump(model,        MODEL_PATH)
joblib.dump(encoders,     ENC_PATH)
joblib.dump(FEATURE_COLS, FEAT_PATH)   # predictor.py использует этот список

print(f"\nСохранено:")
print(f"  {MODEL_PATH}")
print(f"  {ENC_PATH}")
print(f"  {FEAT_PATH}")
print("\nГотово!")