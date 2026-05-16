
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# КОНФИГ
# ─────────────────────────────────────────────

RANDOM_SEED     = 42
NUM_TOTAL       = 100_000
FRAUD_RATIO     = 0.03          # 3 % — реалистично (глава 1.2)
OUTPUT_FILE     = "synthetic_transactions.csv"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────
# СПРАВОЧНИКИ
# ─────────────────────────────────────────────

MERCHANT_CATEGORIES = [
    "Оборудование", "Электроника", "Сырье",
    "Услуги", "Транспорт", "Другое",
]

DEVICE_TYPES = ["desktop", "mobile", "tablet", "api"]

# Средняя сумма по категории — корреляция 1
CATEGORY_AMOUNT_MEAN = {
    "Оборудование": 85_000,
    "Электроника":  55_000,
    "Сырье":       120_000,
    "Услуги":       30_000,
    "Транспорт":    45_000,
    "Другое":       25_000,
}

# Типичное устройство по категории — корреляция 2
CATEGORY_DEVICE_WEIGHTS = {
    "Оборудование": [50, 20, 10, 20],
    "Электроника":  [30, 45, 10, 15],
    "Сырье":        [55, 15, 5,  25],
    "Услуги":       [35, 40, 15, 10],
    "Транспорт":    [30, 50, 10, 10],
    "Другое":       [25, 45, 20, 10],
}

# ─────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ─────────────────────────────────────────────

def legit_hour(quarter: int) -> int:
    """
    Концептуальный дрейф: рабочее время смещается по кварталам.
    Q1 — стандартное (пик 13ч), Q4 — более ранее (пик 11ч).
    """
    peak = {1: 13, 2: 13, 3: 14, 4: 11}[quarter]
    h = int(np.random.normal(peak, 2.5))
    return max(7, min(22, h))

def fraud_hour(fraud_type: str) -> int:
    """Ночные часы для account_takeover, рабочие для остальных."""
    if fraud_type == "account_takeover":
        return random.choice(list(range(0, 6)) + [22, 23])
    return random.randint(8, 20)

def amount_for_category(cat: str, sigma_factor: float = 1.0) -> float:
    """Логнормальная сумма с центром по категории."""
    mu = CATEGORY_AMOUNT_MEAN[cat]
    sigma = 0.9 * sigma_factor
    val = np.random.lognormal(mean=np.log(mu), sigma=sigma)
    return round(float(np.clip(val, 500, 3_000_000)), 2)

def device_for_category(cat: str) -> str:
    return random.choices(DEVICE_TYPES, weights=CATEGORY_DEVICE_WEIGHTS[cat])[0]

def amount_deviation(amount: float, avg: float) -> float:
    if avg <= 0:
        return 0.0
    return round(float(np.clip((amount - avg) / avg, -1.0, 15.0)), 4)

def inject_noise(row: dict) -> dict:
    """
    Шум и выбросы (~2 % строк):
      - случайный пропуск avg_amount_30d → NaN
      - аномальный velocity выброс
      - округление amount до "красивого" числа
    """
    r = random.random()
    if r < 0.008:
        row["avg_amount_30d"] = float("nan")   # пропуск как в реальных логах
        row["amount_deviation"] = float("nan")
    elif r < 0.013:
        row["velocity_1h"] = random.randint(20, 60)  # технический сбой — всплеск
    elif r < 0.020:
        # "красивые" суммы — характерны для ручного ввода
        row["amount"] = round(row["amount"] / 1000) * 1000
    return row

# ─────────────────────────────────────────────
# ГЕНЕРАЦИЯ ТРАНЗАКЦИЙ
# ─────────────────────────────────────────────

num_fraud = int(NUM_TOTAL * FRAUD_RATIO)
num_legit = NUM_TOTAL - num_fraud

# Равномерное распределение по трём типам мошенничества
fp = num_fraud // 3
fraud_labels = (
    ["invoice_fraud"]    * fp +
    ["account_takeover"] * fp +
    ["structuring"]      * (num_fraud - 2 * fp)
)
all_labels = fraud_labels + ["legit"] * num_legit
random.shuffle(all_labels)

base_date = datetime(2024, 1, 1)
records = []

for i, label in enumerate(all_labels):
    is_fraud = label != "legit"

    # Квартал для концептуального дрейфа
    day_offset = random.randint(0, 364)
    quarter = min((day_offset // 91) + 1, 4)

    cat = random.choice(MERCHANT_CATEGORIES)

    # ── ЛЕГИТИМНЫЕ ─────────────────────────────────────────────────────────
    if label == "legit":
        amount      = amount_for_category(cat)
        avg_30d     = round(amount * random.uniform(0.6, 1.4), 2)
        velocity    = int(np.random.choice([1, 2, 3, 4], p=[0.55, 0.30, 0.10, 0.05]))
        new_merch   = random.choices([0, 1], weights=[87, 13])[0]
        hour        = legit_hour(quarter)
        device      = device_for_category(cat)

    # ── ПОДМЕНА РЕКВИЗИТОВ (invoice fraud) ─────────────────────────────────
    # Новый получатель + сумма резко выше среднего + рабочие часы (маскировка)
    elif label == "invoice_fraud":
        avg_30d     = round(random.uniform(20_000, 150_000), 2)
        amount      = round(avg_30d * random.uniform(2.5, 7.0), 2)
        amount      = float(np.clip(amount, 80_000, 2_500_000))
        velocity    = random.randint(1, 2)      # не спешат, чтобы не бросаться в глаза
        new_merch   = 1
        hour        = fraud_hour(label)
        device      = random.choices(["desktop", "api"], weights=[55, 45])[0]
        cat         = random.choices(
            MERCHANT_CATEGORIES,
            weights=[20, 20, 25, 15, 10, 10]
        )[0]

    # ── НЕСАНКЦИОНИРОВАННЫЙ ПЕРЕВОД (account takeover) ──────────────────────
    # Ночь + высокий velocity + нетипичная сумма + mobile/api
    elif label == "account_takeover":
        avg_30d     = round(random.uniform(10_000, 100_000), 2)
        amount      = round(avg_30d * random.uniform(1.8, 6.0), 2)
        amount      = float(np.clip(amount, 5_000, 800_000))
        velocity    = random.randint(6, 18)     # много операций за короткое время
        new_merch   = 1
        hour        = fraud_hour(label)
        device      = random.choices(["mobile", "api"], weights=[50, 50])[0]
        cat         = random.choices(
            MERCHANT_CATEGORIES,
            weights=[5, 30, 5, 25, 5, 30]
        )[0]

    # ── ДРОБЛЕНИЕ (structuring) ─────────────────────────────────────────────
    # Суммы чуть ниже порога + высокий velocity + повторяемость
    elif label == "structuring":
        avg_30d     = round(random.uniform(8_000, 50_000), 2)
        amount      = round(random.uniform(8_500, 14_800), 2)   # под порогом 15k
        velocity    = random.randint(4, 14)
        new_merch   = random.choices([0, 1], weights=[45, 55])[0]
        hour        = random.randint(6, 22)
        device      = random.choices(DEVICE_TYPES, weights=[25, 30, 10, 35])[0]
        cat         = random.choices(
            MERCHANT_CATEGORIES,
            weights=[10, 15, 15, 30, 15, 15]
        )[0]

    # ── HARD NEGATIVES ──────────────────────────────────────────────────────
    # ~15 % легитимных получают "подозрительные" признаки, но остаются легитимными.
    # Это намеренно усложняет задачу классификации.
    if label == "legit" and random.random() < 0.15:
        tweak = random.randint(1, 4)
        if tweak == 1:
            # Командировка ночью — легитимно, но похоже на account_takeover
            hour = random.choice([0, 1, 2, 3, 4, 5, 23])
        elif tweak == 2:
            # Крупная разовая закупка — похоже на invoice_fraud
            amount  = round(avg_30d * random.uniform(2.0, 4.5), 2) if "avg_30d" in dir() else amount
            new_merch = 1
        elif tweak == 3:
            # Много мелких операций (сезонный закуп) — похоже на structuring
            velocity  = random.randint(4, 8)
            amount    = round(random.uniform(9_000, 14_500), 2)
        else:
            # API-интеграция с нетипичным устройством
            device = "api"

    # ── ОБЩИЕ ВЫЧИСЛЯЕМЫЕ ПРИЗНАКИ ──────────────────────────────────────────
    dev   = amount_deviation(amount, avg_30d)
    night = 1 if (hour < 6 or hour >= 22) else 0

    ts = base_date + timedelta(
        days=day_offset,
        hours=hour,
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59),
    )

    row = {
        "transaction_id":    f"TX{i:07d}",
        "timestamp":         ts.isoformat(),
        "amount":            round(amount, 2),
        "merchant_category": cat,
        "device_type":       device,
        "velocity_1h":       velocity,
        "avg_amount_30d":    round(avg_30d, 2),
        "new_merchant":      new_merch,
        "hour":              hour,
        "is_night":          night,
        "amount_deviation":  dev,
        "is_fraud":          int(is_fraud),
        "fraud_type":        label,    # для анализа, в обучение ML не идёт
    }

    row = inject_noise(row)
    records.append(row)

# ─────────────────────────────────────────────
# СОХРАНЕНИЕ И СТАТИСТИКА
# ─────────────────────────────────────────────

df = pd.DataFrame(records)
df.to_csv(OUTPUT_FILE, index=False)

print("=" * 50)
print(f"{'Датасет сохранён':>30}: {OUTPUT_FILE}")
print(f"{'Всего транзакций':>30}: {len(df):,}")
print(f"{'Мошеннических':>30}: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.1f}%)")
print()
print("Распределение по типам мошенничества:")
print(df[df['is_fraud']==1]['fraud_type'].value_counts().to_string())
print()
print("Строк с пропусками (avg_amount_30d):",
      df['avg_amount_30d'].isna().sum(), "  ← намеренный шум")
print("Строк с аномальным velocity (>20):",
      (df['velocity_1h'] > 20).sum(), "  ← намеренный выброс")
print()
print("Статистика ключевых признаков:")
print(df[['amount', 'velocity_1h', 'avg_amount_30d', 'amount_deviation']]
      .describe().round(2).to_string())