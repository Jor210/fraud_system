from database.session import engine
from app.models import Base
from sqlalchemy import text

print("🔄 Полная очистка базы данных...")

# Удаляем все таблицы
Base.metadata.drop_all(bind=engine)
print("✅ Все таблицы удалены")

# Создаём таблицы заново
Base.metadata.create_all(bind=engine)
print("✅ Таблицы успешно пересозданы!")

print("База данных полностью очищена и готова к использованию.")