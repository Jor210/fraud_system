import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="Fraud Detection System",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

st.title("🛡️ Hybrid Fraud Detection System")
st.markdown(
    "**Дипломная работа — Гибридная система выявления мошенничества (Rules + ML)**"
)

# ==================== DATABASE ====================

@st.cache_resource
def get_engine():
    return create_engine(
        "postgresql+psycopg2://postgres:258654357@localhost:5432/fraud_detection"
    )

engine = get_engine()

# ==================== LOAD DATA ====================

@st.cache_data(ttl=10)
def load_data():
    return pd.read_sql("""
        SELECT *
        FROM transactions
        ORDER BY timestamp DESC
    """, engine)

df = load_data()

# ==================== SIDEBAR ====================

st.sidebar.header("🔧 Управление")

if st.sidebar.button("🔄 Обновить данные"):
    st.cache_data.clear()
    st.rerun()

# ==================== METRICS ====================

col1, col2, col3, col4, col5, col6 = st.columns(6)

total_tx = len(df)

fraud_tx = len(
    df[df['predicted_fraud'] == True]
) if 'predicted_fraud' in df.columns and not df.empty else 0

fraud_rate = (
    fraud_tx / total_tx * 100
) if total_tx > 0 else 0

avg_risk = (
    df['risk_score'].mean()
) if 'risk_score' in df.columns and not df.empty else 0

ml_detected = len(
    df[df['ml_prediction'] == True]
) if 'ml_prediction' in df.columns and not df.empty else 0

real_fraud = len(
    df[df['is_fraud'] == True]
) if 'is_fraud' in df.columns and not df.empty else 0

needs_ml = len(
    df[df['needs_ml_check'] == True]
) if 'needs_ml_check' in df.columns and not df.empty else 0

col1.metric(
    "Всего транзакций",
    f"{total_tx:,}"
)

col2.metric(
    "Rule Fraud",
    f"{fraud_tx:,}",
    delta=f"{fraud_rate:.1f}%"
)

col3.metric(
    "ML Fraud",
    f"{ml_detected:,}"
)

col4.metric(
    "Средний Risk Score",
    f"{avg_risk:.3f}"
)

col5.metric(
    "Требуют ML-проверки",
    f"{needs_ml:,}"
)

col6.metric(
    "Подтверждённое мошенничество",
    f"{real_fraud:,}"
)

# ==================== FILTERS ====================

st.subheader("📋 Транзакции")

col_f1, col_f2, col_f3 = st.columns([3, 2, 2])

with col_f1:
    search = st.text_input(
        "🔍 Поиск по transaction_id или merchant",
        ""
    )

with col_f2:
    min_risk = st.slider(
        "Минимальный Risk Score",
        0.0,
        1.0,
        0.0,
        0.05
    )

with col_f3:
    show_only_fraud = st.checkbox(
        "Только подозрительные",
        value=False
    )

# ==================== FILTERING ====================

display_df = df.copy()

if search and not display_df.empty:

    transaction_filter = (
        display_df['transaction_id']
        .astype(str)
        .str.contains(search, case=False, na=False)
    ) if 'transaction_id' in display_df.columns else False

    merchant_filter = (
        display_df['merchant_id']
        .astype(str)
        .str.contains(search, case=False, na=False)
    ) if 'merchant_id' in display_df.columns else False

    display_df = display_df[
        transaction_filter | merchant_filter
    ]

if 'risk_score' in display_df.columns and not display_df.empty:
    display_df = display_df[
        display_df['risk_score'] >= min_risk
    ]

if show_only_fraud and 'predicted_fraud' in display_df.columns:
    display_df = display_df[
        display_df['predicted_fraud'] == True
    ]

# ==================== TABLE ====================

st.subheader("🚨 Таблица транзакций")

columns_to_show = [
    'transaction_id',
    'timestamp',
    'amount',
    'merchant_id',
    'merchant_category',

    'risk_score',

    'predicted_fraud',

    'ml_probability',
    'ml_prediction',
    'ml_risk_level',

    'needs_ml_check',

    'is_fraud'
]

existing_columns = [
    col for col in columns_to_show
    if col in display_df.columns
]

st.dataframe(
    display_df[existing_columns],
    width='stretch',
    hide_index=True,
    column_config={

        "timestamp": st.column_config.DatetimeColumn(
            "Дата и время",
            format="DD.MM.YYYY HH:mm"
        ),

        "amount": st.column_config.NumberColumn(
            "Сумма ₽",
            format="%.2f"
        ),

        "risk_score": st.column_config.ProgressColumn(
            "Hybrid Risk Score",
            min_value=0,
            max_value=1,
            format="%.2f"
        ),

        "ml_probability": st.column_config.ProgressColumn(
            "ML Probability",
            min_value=0,
            max_value=1,
            format="%.2f"
        ),

        "predicted_fraud": st.column_config.CheckboxColumn(
            "Rule Fraud"
        ),

        "ml_prediction": st.column_config.CheckboxColumn(
            "ML Fraud"
        ),

        "needs_ml_check": st.column_config.CheckboxColumn(
            "Needs ML"
        ),

        "is_fraud": st.column_config.CheckboxColumn(
            "Real Fraud"
        ),
    }
)

# ==================== ANALYTICS ====================

st.divider()

st.subheader("📊 Аналитика")

tab1, tab2, tab3, tab4 = st.tabs([
    "Risk Score",
    "Динамика",
    "Категории",
    "Top High Risk"
])

# ==================== TAB 1 ====================

with tab1:

    if 'risk_score' in df.columns and not df.empty:

        fig = px.histogram(
            df,
            x="risk_score",
            nbins=20,
            title="Распределение Hybrid Risk Score"
        )

        st.plotly_chart(
            fig,
            width='stretch'
        )

# ==================== TAB 2 ====================

with tab2:

    if 'timestamp' in df.columns and not df.empty:

        temp_df = df.copy()

        temp_df['date'] = pd.to_datetime(
            temp_df['timestamp']
        ).dt.date

        daily = temp_df.groupby('date').agg({
            'id': 'count',
            'predicted_fraud': 'sum'
        }).rename(columns={
            'id': 'transactions',
            'predicted_fraud': 'fraud'
        })

        daily['fraud_rate'] = (
            daily['fraud']
            / daily['transactions']
            * 100
        ).fillna(0)

        st.line_chart(
            daily[['transactions', 'fraud_rate']],
            width='stretch'
        )

# ==================== TAB 3 ====================

with tab3:

    if 'merchant_category' in df.columns and not df.empty:

        cat = df.groupby('merchant_category').agg({
            'id': 'count',
            'predicted_fraud': 'sum',
            'amount': 'sum'
        })

        cat['fraud_rate'] = (
            cat['predicted_fraud']
            / cat['id']
            * 100
        ).round(2)

        st.dataframe(
            cat.sort_values(
                'fraud_rate',
                ascending=False
            ),
            width='stretch'
        )

# ==================== TAB 4 ====================

with tab4:

    if 'risk_score' in df.columns and not df.empty:

        high_risk = df.nlargest(10, 'risk_score')

        top_columns = [
            col for col in [
                'transaction_id',
                'amount',
                'merchant_id',
                'merchant_category',
                'risk_score',
                'ml_probability',
                'predicted_fraud',
                'ml_prediction',
                'timestamp'
            ]
            if col in high_risk.columns
        ]

        st.dataframe(
            high_risk[top_columns],
            width='stretch'
        )

# ==================== DEBUG ====================

with st.expander("🛠️ Debug Columns"):
    st.write(df.columns.tolist())

# ==================== FOOTER ====================

st.caption(
    "🧠 Гибридная система: Rule Engine + Machine Learning"
)