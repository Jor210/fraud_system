# ml/train_model.py
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

# =========================
# LOAD DATA
# =========================
print("Загрузка данных...")
df = pd.read_csv("synthetic_transactions.csv")

print(f"Размер датасета: {df.shape}")
print(f"Мошенничество: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")

# =========================
# FEATURE ENGINEERING
# =========================
# Дополнительные полезные признаки
df['amount_to_avg_ratio'] = df['amount_deviation']
df['velocity_per_hour_score'] = df['velocity_1h'] * (df['amount'] / 10000)  # комбинированный признак

# =========================
# SELECT FEATURES
# =========================
categorical_features = ['merchant_category', 'device_type', 'location']
numerical_features = [
    'amount', 'velocity_1h', 'avg_amount_30d', 'new_merchant',
    'hour', 'amount_deviation', 'amount_to_avg_ratio', 'velocity_per_hour_score'
]

target = 'is_fraud'

X = df[categorical_features + numerical_features]
y = df[target]

print(f"Используется признаков: {len(X.columns)}")

# =========================
# ENCODE CATEGORICAL
# =========================
encoders = {}

for col in categorical_features:
    encoder = LabelEncoder()
    X[col] = encoder.fit_transform(X[col])
    encoders[col] = encoder

# =========================
# TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# MODEL TRAINING
# =========================
print("\nОбучение модели RandomForest...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=14,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',      # важно при дисбалансе
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =========================
# PREDICTION & EVALUATION
# =========================
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ МОДЕЛИ")
print("="*60)

print("\nAccuracy:", round(model.score(X_test, y_test), 4))
print("ROC-AUC:  ", round(roc_auc_score(y_test, y_pred_proba), 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nТОП-10 важных признаков:")
print(feature_importance.head(10))

# =========================
# SAVE MODEL AND ENCODERS
# =========================
joblib.dump(model, "model.pkl")
joblib.dump(encoders, "encoders.pkl")

print("\nМодель успешно сохранена!")
print("   → model.pkl")
print("   → encoders.pkl")