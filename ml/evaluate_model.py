"""
Анализ качества модели обнаружения мошенничества.
Генерирует графики и метрики для дипломного отчёта.

Запуск (из папки ml/):
    python evaluate_model.py

Требует в той же папке:
    model.pkl, encoders.pkl, feature_names.pkl, synthetic_transactions.csv

Результат:
    папка ml/report/ с PNG-графиками и файлом metrics.txt
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
    average_precision_score, roc_auc_score,
)
from sklearn.calibration import calibration_curve

matplotlib.rcParams["font.family"]  = "DejaVu Sans"
matplotlib.rcParams["figure.dpi"]   = 150
matplotlib.rcParams["savefig.dpi"]  = 200
matplotlib.rcParams["axes.spines.top"]   = False
matplotlib.rcParams["axes.spines.right"] = False

# ─────────────────────────────────────────────
# ПУТИ
# ─────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR  = os.path.join(BASE_DIR, "report")
os.makedirs(REPORT_DIR, exist_ok=True)

MODEL_PATH  = os.path.join(BASE_DIR, "model.pkl")
ENC_PATH    = os.path.join(BASE_DIR, "encoders.pkl")
FEAT_PATH   = os.path.join(BASE_DIR, "feature_names.pkl")
DATA_PATH   = os.path.join(BASE_DIR, "synthetic_transactions.csv")

# ─────────────────────────────────────────────
# ЗАГРУЗКА
# ─────────────────────────────────────────────

print("Загружаю модель и данные...")
model         = joblib.load(MODEL_PATH)
encoders      = joblib.load(ENC_PATH)
feature_names = joblib.load(FEAT_PATH)

df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["transaction_id", "timestamp", "fraud_type"], errors="ignore")
df["avg_amount_30d"]   = df["avg_amount_30d"].fillna(df["avg_amount_30d"].median())
df["amount_deviation"] = df["amount_deviation"].fillna(0.0)

for col in ["merchant_category", "device_type"]:
    if col in encoders:
        df[col] = encoders[col].transform(df[col])

X = df[feature_names]
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"  Тестовая выборка: {len(X_test):,} транзакций  "
      f"({y_test.sum():,} мошеннических)\n")

# ─────────────────────────────────────────────
# ЦВЕТОВАЯ СХЕМА
# ─────────────────────────────────────────────

C_FRAUD  = "#E05C5C"   # красный — мошенничество
C_LEGIT  = "#4C9EBE"   # синий  — легитимные
C_ACCENT = "#2E7D9E"
C_LIGHT  = "#EFF6FB"
PALETTE  = [C_LEGIT, C_FRAUD]

# ─────────────────────────────────────────────
# 1. МАТРИЦА ОШИБОК
# ─────────────────────────────────────────────

def plot_confusion_matrix():
    cm = confusion_matrix(y_test, y_pred)
    labels = [["TN\n(верно легитимные)", "FP\n(ложная тревога)"],
              ["FN\n(пропущенные)", "TP\n(верно мошеннические)"]]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=False, fmt="d", cmap="Blues",
        linewidths=1, linecolor="white", ax=ax,
        cbar_kws={"shrink": 0.8}
    )
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.38, f"{cm[i, j]:,}",
                    ha="center", va="center", fontsize=16, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() * 0.5 else "#333")
            ax.text(j + 0.5, i + 0.65, labels[i][j],
                    ha="center", va="center", fontsize=8, color="#555")

    ax.set_xticklabels(["Легитимная", "Мошенническая"], fontsize=10)
    ax.set_yticklabels(["Легитимная", "Мошенническая"], fontsize=10, rotation=0)
    ax.set_xlabel("Предсказанный класс", fontsize=11)
    ax.set_ylabel("Истинный класс", fontsize=11)
    ax.set_title("Матрица ошибок (Confusion Matrix)", fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "1_confusion_matrix.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {path}")

# ─────────────────────────────────────────────
# 2. ROC-КРИВАЯ
# ─────────────────────────────────────────────

def plot_roc():
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # Оптимальный порог (максимум Youden J)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thr = thresholds[best_idx]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=C_ACCENT, lw=2.5,
            label=f"ROC-кривая (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#aaa", lw=1.2, linestyle="--",
            label="Случайный классификатор")
    ax.scatter(fpr[best_idx], tpr[best_idx], s=80, color=C_FRAUD, zorder=5,
               label=f"Оптимальный порог = {best_thr:.2f}")
    ax.fill_between(fpr, tpr, alpha=0.08, color=C_ACCENT)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC-кривая", fontsize=13, fontweight="bold", pad=14)
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "2_roc_curve.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {path}")
    return roc_auc, best_thr

# ─────────────────────────────────────────────
# 3. PRECISION-RECALL КРИВАЯ
# ─────────────────────────────────────────────

def plot_pr_curve():
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    baseline = y_test.mean()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color=C_FRAUD, lw=2.5,
            label=f"PR-кривая (AP = {ap:.4f})")
    ax.axhline(baseline, color="#aaa", lw=1.2, linestyle="--",
               label=f"Базовый уровень ({baseline:.3f})")
    ax.fill_between(recall, precision, alpha=0.08, color=C_FRAUD)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("Recall (Полнота)", fontsize=11)
    ax.set_ylabel("Precision (Точность)", fontsize=11)
    ax.set_title("Precision-Recall кривая", fontsize=13, fontweight="bold", pad=14)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "3_pr_curve.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {path}")
    return ap

# ─────────────────────────────────────────────
# 4. ВАЖНОСТЬ ПРИЗНАКОВ
# ─────────────────────────────────────────────

def plot_feature_importance():
    importances = model.feature_importances_
    indices = np.argsort(importances)
    names   = [feature_names[i] for i in indices]
    values  = importances[indices]

    # Цвет: топ-3 — акцентный, остальные — светлые
    colors = [C_ACCENT if v >= sorted(values)[-3] else C_LIGHT for v in values]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(names, values, color=colors, edgecolor="white", height=0.6)

    for bar, val in zip(bars, values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9, color="#333")

    ax.set_xlabel("Важность (Gini impurity)", fontsize=11)
    ax.set_title("Важность признаков", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlim(0, max(values) * 1.2)
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "4_feature_importance.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {path}")

# ─────────────────────────────────────────────
# 5. РАСПРЕДЕЛЕНИЕ ВЕРОЯТНОСТЕЙ
# ─────────────────────────────────────────────

def plot_probability_distribution():
    proba_legit = y_proba[y_test == 0]
    proba_fraud = y_proba[y_test == 1]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(proba_legit, bins=60, alpha=0.65, color=C_LEGIT,
            label=f"Легитимные (n={len(proba_legit):,})", density=True)
    ax.hist(proba_fraud, bins=60, alpha=0.65, color=C_FRAUD,
            label=f"Мошеннические (n={len(proba_fraud):,})", density=True)
    ax.axvline(0.5, color="#555", lw=1.5, linestyle="--", label="Порог 0.5")

    ax.set_xlabel("Предсказанная вероятность мошенничества", fontsize=11)
    ax.set_ylabel("Плотность", fontsize=11)
    ax.set_title("Распределение вероятностей по классам",
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "5_probability_distribution.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {path}")

# ─────────────────────────────────────────────
# 6. PRECISION / RECALL / F1 ПО ПОРОГУ
# ─────────────────────────────────────────────

def plot_threshold_metrics():
    thresholds = np.linspace(0.1, 0.9, 200)
    precisions, recalls, f1s = [], [], []

    for thr in thresholds:
        pred = (y_proba >= thr).astype(int)
        tp = ((pred == 1) & (y_test == 1)).sum()
        fp = ((pred == 1) & (y_test == 0)).sum()
        fn = ((pred == 0) & (y_test == 1)).sum()
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f  = 2 * p * r / (p + r) if (p + r) > 0 else 0
        precisions.append(p); recalls.append(r); f1s.append(f)

    best_idx = np.argmax(f1s)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, precisions, color=C_ACCENT,  lw=2, label="Precision")
    ax.plot(thresholds, recalls,    color=C_FRAUD,   lw=2, label="Recall")
    ax.plot(thresholds, f1s,        color="#5BAD72", lw=2, label="F1-score")
    ax.axvline(thresholds[best_idx], color="#555", lw=1.5, linestyle="--",
               label=f"Лучший F1 при пороге {thresholds[best_idx]:.2f}")

    ax.set_xlabel("Порог классификации", fontsize=11)
    ax.set_ylabel("Значение метрики", fontsize=11)
    ax.set_title("Precision / Recall / F1 в зависимости от порога",
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "6_threshold_metrics.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {path}")
    return thresholds[best_idx], f1s[best_idx]

# ─────────────────────────────────────────────
# 7. КРОСС-ВАЛИДАЦИЯ
# ─────────────────────────────────────────────

def plot_cross_validation():
    print("  Кросс-валидация (5-fold)... это займёт ~1 мин")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    metrics = {}
    for metric in ["roc_auc", "f1", "precision", "recall"]:
        scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
        metrics[metric] = scores

    labels_ru = {
        "roc_auc":   "ROC-AUC",
        "f1":        "F1-score",
        "precision": "Precision",
        "recall":    "Recall",
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    positions = list(range(len(metrics)))
    colors_bar = [C_ACCENT, "#5BAD72", "#F0A030", C_FRAUD]

    for pos, (metric, scores) in zip(positions, metrics.items()):
        ax.bar(pos, scores.mean(), color=colors_bar[pos],
               alpha=0.85, width=0.5, label=labels_ru[metric],
               yerr=scores.std(), capsize=6, error_kw={"elinewidth": 1.5})
        ax.text(pos, scores.mean() + scores.std() + 0.01,
                f"{scores.mean():.3f}±{scores.std():.3f}",
                ha="center", fontsize=9, color="#333")

    ax.set_xticks(positions)
    ax.set_xticklabels([labels_ru[m] for m in metrics], fontsize=10)
    ax.set_ylabel("Значение метрики", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title("5-Fold кросс-валидация", fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "7_cross_validation.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {path}")
    return {k: (v.mean(), v.std()) for k, v in metrics.items()}

# ─────────────────────────────────────────────
# 8. ТЕКСТОВЫЙ ОТЧЁТ metrics.txt
# ─────────────────────────────────────────────

def save_metrics_txt(roc_auc, ap, best_thr, best_f1, cv_scores):
    report = classification_report(
        y_test, y_pred,
        target_names=["Легитимная", "Мошенническая"]
    )
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    lines = [
        "=" * 56,
        "  ОТЧЁТ О КАЧЕСТВЕ МОДЕЛИ",
        "  Система выявления мошенничества в транзакциях",
        "=" * 56,
        "",
        f"  Алгоритм:          Random Forest",
        f"  Деревьев:          {model.n_estimators}",
        f"  Макс. глубина:     {model.max_depth}",
        f"  Обучающая выб.:    {len(X_train):,} транзакций",
        f"  Тестовая выб.:     {len(X_test):,} транзакций",
        f"  Доля мошенничества:{y.mean()*100:.1f}%",
        "",
        "-" * 56,
        "  МЕТРИКИ НА ТЕСТОВОЙ ВЫБОРКЕ",
        "-" * 56,
        "",
        f"  ROC-AUC:           {roc_auc:.4f}",
        f"  Average Precision: {ap:.4f}",
        f"  Оптимальный порог: {best_thr:.2f}  (по F1 = {best_f1:.4f})",
        "",
        f"  Confusion Matrix:",
        f"    Верно легитимные  (TN): {tn:,}",
        f"    Ложная тревога    (FP): {fp:,}",
        f"    Пропущенные       (FN): {fn:,}",
        f"    Верно мошеннич.   (TP): {tp:,}",
        "",
        "  Classification Report:",
        "",
        report,
        "",
        "-" * 56,
        "  5-FOLD КРОСС-ВАЛИДАЦИЯ",
        "-" * 56,
        "",
    ]
    for metric, (mean, std) in cv_scores.items():
        name = {"roc_auc": "ROC-AUC", "f1": "F1-score",
                "precision": "Precision", "recall": "Recall"}[metric]
        lines.append(f"  {name:<18} {mean:.4f} ± {std:.4f}")

    lines += [
        "",
        "-" * 56,
        "  ПРИЗНАКИ МОДЕЛИ",
        "-" * 56,
        "",
    ]
    for i, feat in enumerate(feature_names, 1):
        imp = model.feature_importances_[feature_names.index(feat)]
        lines.append(f"  {i}. {feat:<22} важность: {imp:.4f}")

    lines += ["", "=" * 56]

    path = os.path.join(REPORT_DIR, "metrics.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Сохранено: {path}")

# ─────────────────────────────────────────────
# ЗАПУСК
# ─────────────────────────────────────────────

print("\nГенерирую графики для отчёта...\n")

plot_confusion_matrix()
roc_auc, best_thr = plot_roc()
ap                = plot_pr_curve()
plot_feature_importance()
plot_probability_distribution()
best_thr_f1, best_f1 = plot_threshold_metrics()
cv_scores         = plot_cross_validation()
save_metrics_txt(roc_auc, ap, best_thr_f1, best_f1, cv_scores)

print(f"""
{'=' * 50}
  Готово! Все файлы в папке: ml/report/

  1_confusion_matrix.png     — матрица ошибок
  2_roc_curve.png            — ROC-кривая
  3_pr_curve.png             — Precision-Recall
  4_feature_importance.png   — важность признаков
  5_probability_distribution — распределение вероятностей
  6_threshold_metrics.png    — метрики по порогу
  7_cross_validation.png     — кросс-валидация
  metrics.txt                — все числа для отчёта
{'=' * 50}

  ROC-AUC: {roc_auc:.4f}
  Лучший F1: {best_f1:.4f}  (порог {best_thr_f1:.2f})
""")