import streamlit as st
import pandas as pd
from sqlalchemy import create_engine

st.set_page_config(page_title="Fraud Detection", layout="wide", page_icon="🛡️")

st.title("🛡️ Hybrid Fraud Detection System")
st.markdown("**Дипломная работа — Гибридная система выявления мошенничества (Rules + ML)**")

# Подключение к базе
engine = create_engine("postgresql+psycopg2://postgres:258654357@localhost:5432/fraud_detection")

df = pd.read_sql("""
    SELECT * FROM transactions 
    ORDER BY timestamp DESC
""", engine)

# Метрики
col1, col2, col3, col4 = st.columns(4)
col1.metric("Всего транзакций", len(df), delta=None)
col2.metric("Подозрительных",
            len(df[df['predicted_fraud'] == True]),
            delta=f"{len(df[df['predicted_fraud'] == True])}")
col3.metric("Средний Risk Score", f"{df['risk_score'].mean():.3f}")
col4.metric("Мошенничество %", f"{df['predicted_fraud'].mean()*100:.1f}%")

st.divider()

# Фильтры
st.subheader("📋 Транзакции")
filter_col1, filter_col2 = st.columns([3, 1])
with filter_col1:
    search = st.text_input("Поиск по ID транзакции или merchant", "")
with filter_col2:
    show_only_fraud = st.checkbox("Только подозрительные", value=False)

# Фильтрация
display_df = df.copy()
if search:
    display_df = display_df[display_df['transaction_id'].str.contains(search, case=False) |
                           display_df['merchant_id'].str.contains(search, case=False)]
if show_only_fraud:
    display_df = display_df[display_df['predicted_fraud'] == True]

# Красивая таблица
st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "risk_score": st.column_config.ProgressColumn(
            "Risk Score", min_value=0, max_value=1, format="%.2f"
        ),
        "predicted_fraud": st.column_config.CheckboxColumn("Подозрительно"),
        "timestamp": st.column_config.DatetimeColumn("Дата и время", format="DD.MM.YYYY HH:mm"),
        "amount": st.column_config.NumberColumn("Сумма ₽", format="%.2f"),
    }
)

# Графики
st.divider()
st.subheader("📊 Аналитика")

tab1, tab2, tab3 = st.tabs(["Risk Score", "Суммы по статусу", "Динамика по времени"])

with tab1:
    st.bar_chart(df['risk_score'].value_counts(bins=5).sort_index())

with tab2:
    amount_by_status = df.groupby('predicted_fraud')['amount'].sum()
    st.bar_chart(amount_by_status)

with tab3:
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    daily = df.groupby('date').size()
    st.line_chart(daily)

st.caption("🧠 Гибридная система: Rule Engine + (в будущем) Machine Learning")