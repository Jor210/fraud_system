from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean
from datetime import datetime
from database.base import Base


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String, unique=True, index=True, nullable=False)

    # Основные данные транзакции
    account_id = Column(String, index=True)
    amount = Column(Float, nullable=False)
    currency = Column(String, default="RUB")

    merchant_id = Column(String)
    merchant_category = Column(String)

    # Время
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Признаки для анализа
    device_type = Column(String)  # desktop, mobile, etc.
    location = Column(String)

    # Поведенческие признаки
    velocity_1h = Column(Integer, default=0)  # сколько транзакций за последний час
    avg_amount_30d = Column(Float, default=0.0)  # средняя сумма за 30 дней
    new_merchant = Column(Boolean, default=False)

    # Результаты
    is_fraud = Column(Boolean, default=None)   # реальная метка (None = неизвестно, проставляется вручную или при разметке)
    predicted_fraud = Column(Boolean, default=None)  # предсказание системы (rule-based или ML)
    risk_score = Column(Float, default=0.0)    # итоговый скор риска от 0.0 до 1.0
    needs_ml_check = Column(Boolean, default=False)  # флаг: требует ли транзакция проверки ML-модулем

    def __repr__(self):
        return f"<Transaction {self.transaction_id} - {self.amount} RUB>"