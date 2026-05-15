import streamlit as st
import requests
from datetime import datetime

st.set_page_config(page_title="Добавить транзакцию", layout="centered", page_icon="➕")

st.title("➕ Добавление новой транзакции")
st.markdown("### Введите данные бизнес-транзакции")

with st.form("transaction_form", clear_on_submit=True):
    col1, col2 = st.columns(2)

    with col1:
        transaction_id = st.text_input("ID транзакции", value=f"TX{datetime.now().strftime('%Y%m%d%H%M%S')}")
        account_id = st.text_input("ID аккаунта/отправителя", value="ACC001")
        amount = st.number_input("Сумма (RUB)", min_value=100.0, value=285000.0, step=5000.0)
        merchant_id = st.text_input("ID получателя", value="MERCH_NEW777")

    with col2:
        merchant_category = st.selectbox("Категория",
                                         ["Оборудование", "Электроника", "Сырье", "Услуги", "Транспорт", "Другое"])
        device_type = st.selectbox("Устройство", ["desktop", "mobile", "tablet", "api"])
        location = st.text_input("Местоположение", value="Москва")
        velocity_1h = st.slider("Кол-во операций за последний час", 0, 25, 3)

    avg_amount_30d = st.number_input("Средняя сумма за 30 дней", min_value=0.0, value=62000.0)
    new_merchant = st.checkbox("Новый получатель (первый платёж)", value=True)

    col_a, col_b = st.columns(2)
    analyze = col_a.form_submit_button("🔍 Проанализировать", type="primary")
    save_only = col_b.form_submit_button("💾 Только сохранить")

if analyze or save_only:
    transaction_data = {
        "transaction_id": transaction_id,
        "account_id": account_id,
        "amount": amount,
        "currency": "RUB",
        "merchant_id": merchant_id,
        "merchant_category": merchant_category,
        "device_type": device_type,
        "location": location,
        "velocity_1h": velocity_1h,
        "avg_amount_30d": avg_amount_30d,
        "new_merchant": new_merchant
    }

    with st.spinner("Отправка транзакции..."):
        try:
            response = requests.post("http://127.0.0.1:8000/transactions/", json=transaction_data)

            if response.status_code == 200:
                result = response.json()

                if result["is_fraud"]:
                    st.error(f"🚨 ВЫСОКИЙ РИСК МОШЕННИЧЕСТВА! Risk Score: **{result['risk_score']}**")
                else:
                    st.success(f"✅ Транзакция принята. Risk Score: **{result['risk_score']}**")

                st.write("**Сработавшие правила:**")
                for rule in result.get("explanation", []):
                    st.info(f"• {rule}")

            else:
                st.error("Ошибка при обработке")
        except:
            st.error("Не удалось подключиться к API. Убедитесь, что FastAPI запущен.")