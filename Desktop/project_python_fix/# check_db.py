import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "market_data.db")

if os.path.exists(DB_PATH):
    size = os.path.getsize(DB_PATH)
    print(f"Файл найден!")
    print(f"Путь: {DB_PATH}")
    print(f"Размер: {size} байт")
else:
    print(f"Файл не найден по пути: {DB_PATH}")
    print("Запусти сначала download_sber.py чтобы создать базу")