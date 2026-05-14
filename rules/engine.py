from datetime import datetime
from typing import Dict, List


class RuleEngine:
    def __init__(self):
        self.rules = self._load_rules()

    def _load_rules(self):
        return [
            # 1. Крупные суммы
            {
                "name": "large_amount",
                "condition": lambda t: t["amount"] > 500000,
                "score": 0.85,
                "explanation": "Сумма транзакции превышает 500 000 руб."
            },
            # 2. Новый получатель + большая сумма
            {
                "name": "new_merchant_high_amount",
                "condition": lambda t: t.get("new_merchant", False) and t["amount"] > 150000,
                "score": 0.80,
                "explanation": "Первый платёж новому получателю на крупную сумму"
            },
            # 3. Высокая скорость (velocity)
            {
                "name": "high_velocity",
                "condition": lambda t: t.get("velocity_1h", 0) > 7,
                "score": 0.90,
                "explanation": "Высокая частота операций (более 7 за час)"
            },
            # 4. Нетипичное время суток
            {
                "name": "unusual_time",
                "condition": lambda t: self._is_unusual_time(t),
                "score": 0.65,
                "explanation": "Операция в нетипичное время (ночь)"
            },
            # 5. Подозрительное изменение суммы
            {
                "name": "amount_spike",
                "condition": lambda t: t["amount"] > t.get("avg_amount_30d", 0) * 5,
                "score": 0.75,
                "explanation": "Резкий скачок суммы по сравнению со средним значением"
            },
            # 6. Подмена реквизитов (один из самых частых видов мошенничества)
            {
                "name": "possible_account_takeover",
                "condition": lambda t: t.get("velocity_1h", 0) > 5 and t.get("new_merchant", False),
                "score": 0.85,
                "explanation": "Возможная компрометация аккаунта (много операций + новые получатели)"
            },
            # 7. Дробление сумм (признак отмывания)
            {
                "name": "structuring",
                "condition": lambda t: 8000 < t["amount"] < 15000,
                "score": 0.60,
                "explanation": "Дробление суммы (признак отмывания денежных средств)"
            },
        ]

    def _is_unusual_time(self, t: Dict) -> bool:
        """Проверка на ночное время"""
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
            except Exception:
                continue

        risk_score = min(total_score, 1.0)

        return {
            "risk_score": round(risk_score, 2),
            "is_fraud": risk_score >= 0.65,
            "triggered_rules": triggered_rules,
            "needs_ml_check": risk_score > 0.4
        }