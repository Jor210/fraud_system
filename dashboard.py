import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="Fraud Detection System",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

st.title("🛡️ Hybrid Fraud Detection System")
st.markdown("**Дипломная работа — Гибридная система выявления мошенничества (Rules + ML)**")

# Подключение к БД
@st.cache_resource
def get_engine():
    return create_engine("postgresql+psycopg2://postgres:258654357@localhost:5432/fraud_detection")

engine = get_engine()

# Загрузка данных
@st.cache_data(ttl=10)
def load_data():
    return pd.read_sql("""
        SELECT * FROM transactions 
        ORDER BY timestamp DESC
    """, engine)

df = load_data()

# ==================== SIDEBAR ====================
st.sidebar.header("🔧 Управление")
if st.sidebar.button("🔄 Обновить данные"):
    st.cache_data.clear()
    st.rerun()

# ==================== МЕТРИКИ ====================
col1, col2, col3, col4, col5 = st.columns(5)

total_tx = len(df)
fraud_tx = len(df[df['predicted_fraud'] == True])
fraud_rate = (fraud_tx / total_tx * 100) if total_tx > 0 else 0
avg_risk = df['risk_score'].mean()

col1.metric("Всего транзакций", f"{total_tx:,}")
col2.metric("Подозрительных", f"{fraud_tx:,}", delta=f"{fraud_rate:.1f}%")
col3.metric("Средний Risk Score", f"{avg_risk:.3f}")
col4.metric("Требуют ML-проверки", len(df[df['needs_ml_check'] == True]))
col5.metric("Подтверждённое мошенничество", len(df[df['is_fraud'] == True]))

# ==================== ФИЛЬТРЫ ====================
st.subheader("📋 Транзакции")

col_f1, col_f2, col_f3 = st.columns([3, 2, 2])
with col_f1:
    search = st.text_input("🔍 Поиск по ID транзакции или merchant", "")
with col_f2:
    min_risk = st.slider("Минимальный Risk Score", 0.0, 1.0, 0.0, 0.05)
with col_f3:
    show_only_fraud = st.checkbox("Только подозрительные", value=False)

# Фильтрация
display_df = df.copy()
if search:
    display_df = display_df[
        display_df['transaction_id'].str.contains(search, case=False) |
        display_df['merchant_id'].str.contains(search, case=False)
    ]
display_df = display_df[display_df['risk_score'] >= min_risk]
if show_only_fraud:
    display_df = display_df[display_df['predicted_fraud'] == True]

# ==================== ТАБЛИЦА ====================
st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "risk_score": st.column_config.ProgressColumn(
            "Risk Score", min_value=0, max_value=1, format="%.2f"
        ),
        "predicted_fraud": st.column_config.CheckboxColumn("Подозрительно (Rule)"),
        "is_fraud": st.column_config.CheckboxColumn("Подтверждено мошенничество"),
        "timestamp": st.column_config.DatetimeColumn("Дата и время", format="DD.MM.YYYY HH:mm"),
        "amount": st.column_config.NumberColumn("Сумма ₽", format="%.2f"),
    }
)

# ==================== АНАЛИТИКА ====================
st.divider()
st.subheader("📊 Аналитика")

tab1, tab2, tab3, tab4 = st.tabs(["Risk Score", "Динамика во времени", "По категориям", "Топ подозрительных"])

with tab1:
    fig = px.histogram(df, x="risk_score", nbins=20, title="Распределение Risk Score")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    daily = df.groupby('date').agg({
        'id': 'count',
        'predicted_fraud': 'sum'
    }).rename(columns={'id': 'transactions', 'predicted_fraud': 'fraud'})
    daily['fraud_rate'] = daily['fraud'] / daily['transactions'] * 100
    st.line_chart(daily[['transactions', 'fraud_rate']])

with tab3:
    cat = df.groupby('merchant_category').agg({
        'id': 'count',
        'predicted_fraud': 'sum',
        'amount': 'sum'
    })
    cat['fraud_rate'] = cat['predicted_fraud'] / cat['id'] * 100
    st.dataframe(cat.sort_values('fraud_rate', ascending=False))

with tab4:
    high_risk = df.nlargest(10, 'risk_score')
    st.dataframe(high_risk[[
        'transaction_id', 'amount', 'merchant_id',
        'merchant_category', 'risk_score', 'predicted_fraud', 'timestamp'
    ]])

st.caption("🧠 Гибридная система: Rule Engine + Machine Learning (в разработке)")