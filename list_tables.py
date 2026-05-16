from database.session import engine
from sqlalchemy import inspect, text
import pandas as pd


def list_all_tables():
    print("🔍 Получение списка всех таблиц в базе данных...\n")

    inspector = inspect(engine)

    # Получаем список всех таблиц
    tables = inspector.get_table_names()

    print(f"📊 Найдено таблиц: {len(tables)}\n")

    for i, table_name in enumerate(tables, 1):
        print(f"{i:2d}. {table_name}")

    # Более подробная информация
    print("\n" + "=" * 60)
    print("ДЕТАЛЬНАЯ ИНФОРМАЦИЯ ПО ТАБЛИЦАМ")
    print("=" * 60)

    for table_name in tables:
        columns = inspector.get_columns(table_name)
        print(f"\n📋 Таблица: {table_name} ({len(columns)} колонок)")
        print("-" * 50)

        for col in columns:
            print(
                f"   • {col['name']:20} | {str(col['type']):15} | {'NOT NULL' if not col.get('nullable', True) else 'nullable'}")

    # Показать содержимое таблицы transactions (если она есть)
    if "transactions" in tables:
        print("\n" + "=" * 60)
        print("ПЕРВЫЕ 10 ЗАПИСЕЙ ИЗ ТАБЛИЦЫ transactions")
        print("=" * 60)

        try:
            df = pd.read_sql("SELECT * FROM transactions ORDER BY id DESC LIMIT 10", engine)
            print(df.to_string(index=False))
        except Exception as e:
            print(f"Ошибка при чтении данных: {e}")


if __name__ == "__main__":
    list_all_tables()