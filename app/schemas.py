from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class TransactionBase(BaseModel):
    transaction_id: str
    account_id: str
    amount: float
    currency: str = "RUB"
    merchant_id: str
    merchant_category: Optional[str] = None
    device_type: Optional[str] = None
    location: Optional[str] = None

    # Поведенческие признаки
    velocity_1h: int = 0
    avg_amount_30d: float = 0.0
    new_merchant: bool = False


class TransactionCreate(TransactionBase):
    """Схема для создания новой транзакции"""
    pass


class TransactionResponse(TransactionBase):
    """Схема для ответа (что возвращает API)"""
    id: int
    timestamp: datetime
    is_fraud: bool
    predicted_fraud: Optional[bool] = None
    risk_score: float = 0.0

    class Config:
        from_attributes = True  # для работы с SQLAlchemy


class PredictionResponse(BaseModel):
    """Ответ системы после анализа транзакции"""
    transaction_id: str
    risk_score: float
    is_fraud: bool
    explanation: list[str]
    message: str = "Анализ выполнен"