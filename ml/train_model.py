import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("synthetic_transactions.csv")

# =========================
# DROP UNUSED COLUMNS
# =========================

df = df.drop(columns=[
    "transaction_id",
    "timestamp"
])

# =========================
# ENCODE CATEGORICAL FEATURES
# =========================

categorical_columns = [
    "merchant_category",
    "country",
    "device_type"
]

encoders = {}

for column in categorical_columns:

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(df[column])

    encoders[column] = encoder

# =========================
# FEATURES / TARGET
# =========================

X = df.drop(columns=["is_fraud"])

y = df["is_fraud"]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# MODEL
# =========================

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    random_state=42,
    class_weight="balanced"
)

# =========================
# TRAIN
# =========================

model.fit(X_train, y_train)

# =========================
# PREDICTIONS
# =========================

y_pred = model.predict(X_test)

# =========================
# METRICS
# =========================

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================
# FEATURE IMPORTANCE
# =========================

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)

# =========================
# SAVE MODEL
# =========================

joblib.dump(model, "model.pkl")

joblib.dump(encoders, "encoders.pkl")

print("\nModel saved successfully!")