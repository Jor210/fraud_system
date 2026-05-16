# rules/engine.py
from datetime import datetime
from typing import Dict
from ml.predictor import predict_transaction

class RuleEngine:
    def __init__(self):
        self.rules = self._load_rules()

    def _load_rules(self):
        return [
            {
                "name": "large_amount",
                "condition": lambda t: t["amount"] > 500000,
                "score": 0.85,
                "explanation": "Сумма превышает 500 000 руб."
            },
            {
                "name": "new_merchant_high_amount",
                "condition": lambda t: t.get("new_merchant", False) and t["amount"] > 150000,
                "score": 0.82,
                "explanation": "Первый платёж новому получателю на крупную сумму"
            },
            {
                "name": "high_velocity",
                "condition": lambda t: t.get("velocity_1h", 0) > 7,
                "score": 0.90,
                "explanation": "Высокая скорость операций (>7 за час)"
            },
            {
                "name": "unusual_time",
                "condition": lambda t: self._is_unusual_time(t),
                "score": 0.70,
                "explanation": "Операция в нетипичное время"
            },
            {
                "name": "amount_spike",
                "condition": lambda t: t["amount"] > t.get("avg_amount_30d", 0) * 5,
                "score": 0.78,
                "explanation": "Резкий скачок суммы относительно среднего"
            },
            {
                "name": "possible_takeover",
                "condition": lambda t: t.get("velocity_1h", 0) > 5 and t.get("new_merchant", False),
                "score": 0.88,
                "explanation": "Признаки компрометации аккаунта"
            },
        ]

    def _is_unusual_time(self, t: Dict) -> bool:
        try:
            ts = t.get("timestamp")
            if isinstance(ts, str):
                hour = datetime.fromisoformat(ts.replace("Z", "+00:00")).hour
            else:
                hour = datetime.utcnow().hour
            return hour < 6 or hour > 22
        except:
            return False

    def evaluate(self, transaction: Dict) -> Dict:
        triggered_rules = []
        total_score = 0.0

        for rule in self.rules:
            try:
                if rule["condition"](transaction):
                    triggered_rules.append(rule["explanation"])
                    total_score += rule["score"]
            except:
                continue

        rule_risk_score = min(total_score / 2, 1.0)   # нормализация

        # === ML Проверка ===
        try:
            ml_result = predict_transaction(transaction)
            ml_score = ml_result["risk_score_ml"]
        except Exception as e:
            print(f"ML prediction error: {e}")
            ml_score = 0.3  # fallback

        # Гибридный скоринг
        final_risk_score = round(0.55 * rule_risk_score + 0.45 * ml_score, 4)

        is_fraud = final_risk_score >= 0.65

        return {
            "risk_score": final_risk_score,
            "is_fraud": is_fraud,
            "rule_risk_score": round(rule_risk_score, 4),
            "ml_risk_score": ml_score,
            "triggered_rules": triggered_rules,
            "needs_ml_check": True,   # теперь ML всегда используется
            "explanation": triggered_rules
        }