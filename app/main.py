from fastapi import FastAPI

app = FastAPI(title="Система выявления мошенничества")

@app.get("/")
def home():
    return {
        "message": "Система работает! Это backend твоего диплома.",
        "version": "0.1"
    }


@app.get("/health")
def health():
    return {"status": "ok"}