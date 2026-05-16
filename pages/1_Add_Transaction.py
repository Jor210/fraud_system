import streamlit as st
import requests
from datetime import datetime

st.set_page_config(page_title="Добавить транзакцию", layout="centered", page_icon="➕")

st.title("➕ Добавление новой транзакции")
st.markdown("### Введите данные транзакции для анализа")

with st.form("transaction_form", clear_on_submit=True):
    col1, col2 = st.columns(2)

    with col1:
        transaction_id = st.text_input(
            "ID транзакции",
            value=f"TX{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
        account_id = st.text_input("ID аккаунта", value="ACC001")
        amount = st.number_input("Сумма (RUB)", min_value=100.0, value=285000.0, step=1000.0)
        merchant_id = st.text_input("ID получателя", value="MERCH_NEW777")

    with col2:
        merchant_category = st.selectbox(
            "Категория",
            ["electronics", "food", "gaming", "travel", "fashion", "crypto", "services", "other"]
        )
        device_type = st.selectbox("Устройство", ["desktop", "mobile", "tablet", "api"])
        location = st.text_input("Местоположение", value="Москва")
        velocity_1h = st.slider("Операций за последний час", 0, 30, 2)

    col3, col4 = st.columns(2)
    with col3:
        avg_amount_30d = st.number_input("Средняя сумма за 30 дней", min_value=0.0, value=45000.0, step=1000.0)
    with col4:
        new_merchant = st.checkbox("Новый получатель", value=True)

    submitted = st.form_submit_button("🔍 Проанализировать транзакцию", type="primary")

if submitted:
    transaction_data = {
        "transaction_id": transaction_id,
        "account_id": account_id,
        "amount": float(amount),
        "currency": "RUB",
        "merchant_id": merchant_id,
        "merchant_category": merchant_category,
        "device_type": device_type,
        "location": location,
        "velocity_1h": velocity_1h,
        "avg_amount_30d": float(avg_amount_30d),
        "new_merchant": new_merchant
    }

    with st.spinner("Анализ через Rule Engine..."):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/transactions/",
                json=transaction_data,
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()

                if result["is_fraud"]:
                    st.error(f"🚨 **ВЫСОКИЙ РИСК МОШЕННИЧЕСТВА!** Risk Score: **{result['risk_score']}**")
                else:
                    st.success(f"✅ Транзакция принята. Risk Score: **{result['risk_score']}**")

                st.subheader("Сработавшие правила:")
                for rule in result.get("explanation", []):
                    st.info(f"• {rule}")

                st.caption(f"Transaction ID: {result['transaction_id']}")
            else:
                st.error(f"Ошибка API: {response.status_code}")
        except Exception as e:
            st.error(f"Не удалось подключиться к API: {e}")
            st.info("Убедитесь, что FastAPI сервер запущен (`uvicorn app.main:app --reload`)")