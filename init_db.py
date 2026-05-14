from database.session import engine
from app.models import Base

print("Создаём таблицы в базе данных...")

# Создаёт все таблицы из моделей
Base.metadata.create_all(bind=engine)

print("Таблицы успешно созданы!")