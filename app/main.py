from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session

from app.models import Transaction
from app.schemas import TransactionCreate, PredictionResponse
from database.session import get_db

app = FastAPI(
    title="Hybrid Fraud Detection System",
    description="Дипломная работа: Гибридная система выявления мошенничества",
    version="0.1"
)


@app.get("/")
def home():
    return {
        "message": "Система выявления мошенничества в бизнес-транзакциях",
        "status": "работает",
        "version": "0.1"
    }


@app.post("/transactions/", response_model=PredictionResponse)
def create_transaction(tx: TransactionCreate, db: Session = Depends(get_db)):
    # Пока просто сохраняем транзакцию в базу (анализ добавим позже)
    new_tx = Transaction(
        transaction_id=tx.transaction_id,
        account_id=tx.account_id,
        amount=tx.amount,
        currency=tx.currency,
        merchant_id=tx.merchant_id,
        merchant_category=tx.merchant_category,
        device_type=tx.device_type,
        location=tx.location,
        velocity_1h=tx.velocity_1h,
        avg_amount_30d=tx.avg_amount_30d,
        new_merchant=tx.new_merchant,
        is_fraud=False,  # пока по умолчанию
        risk_score=0.0
    )

    db.add(new_tx)
    db.commit()
    db.refresh(new_tx)

    return {
        "transaction_id": new_tx.transaction_id,
        "risk_score": 0.0,
        "is_fraud": False,
        "explanation": ["Транзакция сохранена в базу"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}