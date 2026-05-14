from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from datetime import datetime
from typing import List

from rules.engine import RuleEngine

from app.models import Transaction
from app.schemas import TransactionCreate, PredictionResponse
from database.session import get_db

app = FastAPI(
    title="Hybrid Fraud Detection System",
    description="Дипломная работа: Гибридная система выявления мошенничества",
    version="0.3"
)

rule_engine = RuleEngine()


@app.get("/")
def home():
    return {"message": "Система выявления мошенничества работает", "version": "0.3"}


@app.post("/transactions/", response_model=PredictionResponse)
def create_transaction(tx: TransactionCreate, db: Session = Depends(get_db)):
    tx_dict = tx.dict()
    tx_dict["timestamp"] = datetime.utcnow().isoformat()

    # Анализ через правила
    rule_result = rule_engine.evaluate(tx_dict)

    # Сохранение в базу
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
        is_fraud=rule_result["is_fraud"],
        predicted_fraud=rule_result["is_fraud"],
        risk_score=rule_result["risk_score"]
    )

    db.add(new_tx)
    db.commit()
    db.refresh(new_tx)

    return {
        "transaction_id": new_tx.transaction_id,
        "risk_score": rule_result["risk_score"],
        "is_fraud": rule_result["is_fraud"],
        "explanation": rule_result["triggered_rules"],
        "message": "Анализ выполнен с использованием правил"
    }


# Новый endpoint — список всех транзакций
@app.get("/transactions/", response_model=List[dict])
def get_all_transactions(db: Session = Depends(get_db)):
    transactions = db.query(Transaction).order_by(Transaction.timestamp.desc()).all()

    result = []
    for t in transactions:
        result.append({
            "id": t.id,
            "transaction_id": t.transaction_id,
            "amount": t.amount,
            "merchant_id": t.merchant_id,
            "timestamp": t.timestamp,
            "risk_score": t.risk_score,
            "is_fraud": t.is_fraud
        })
    return result