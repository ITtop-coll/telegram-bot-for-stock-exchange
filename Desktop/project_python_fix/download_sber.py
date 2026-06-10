import requests
import sqlite3
import pandas as pd
from datetime import datetime
import time
import schedule
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "market_data.db")

os.makedirs(BASE_DIR, exist_ok=True)

TICKERS = ['SBER', 'GAZP', 'LKOH', 'ROSN', 'VTBR', 'TATN']
COLLECT_INTERVAL = 60

last_values_cache = {}


def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            name TEXT,
            last_price REAL,
            bid REAL,
            offer REAL,
            spread REAL,
            spread_percent REAL,
            volume REAL,
            volume_rub REAL,
            change_percent REAL,
            timestamp TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS latest_prices (
            ticker TEXT PRIMARY KEY,
            name TEXT,
            last_price REAL,
            bid REAL,
            offer REAL,
            spread REAL,
            volume REAL,
            change_percent REAL,
            updated_at TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            min_price REAL,
            max_price REAL,
            avg_price REAL,
            volume_sum REAL,
            update_count INTEGER,
            start_price REAL,
            end_price REAL,
            UNIQUE(ticker, date)
        )
    ''')

    conn.commit()
    conn.close()
    print(f"База данных инициализирована: {DB_PATH}")


def get_stock_data(ticker):
    try:
        url = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.json"
        response = requests.get(url, timeout=10)
        data = response.json()

        market_data = data['marketdata']['data']
        columns = data['marketdata']['columns']

        last_idx = columns.index('LAST')
        bid_idx = columns.index('BID')
        offer_idx = columns.index('OFFER')
        val_idx = columns.index('VALUE')
        change_idx = columns.index('CHANGE')

        current = market_data[0]

        last_price = current[last_idx]
        bid = current[bid_idx] if current[bid_idx] else last_price
        offer = current[offer_idx] if current[offer_idx] else last_price
        volume = current[val_idx] / 1000000 if current[val_idx] else 0
        change = current[change_idx] if current[change_idx] else 0

        spread = offer - bid
        spread_pct = (spread / bid * 100) if bid and bid > 0 else 0

        now = datetime.now()

        cache_key = f"{ticker}_last"
        last_data = last_values_cache.get(cache_key, {})

        if (last_data.get('last_price') == last_price and
                last_data.get('bid') == bid and
                last_data.get('offer') == offer):
            print(f"{ticker}: {last_price} руб (без изменений)")
            return None

        last_values_cache[cache_key] = {
            'last_price': last_price,
            'bid': bid,
            'offer': offer,
            'volume': volume,
            'timestamp': now
        }

        return {
            'ticker': ticker,
            'name': get_stock_name(ticker),
            'last_price': round(last_price, 2),
            'bid': round(bid, 2),
            'offer': round(offer, 2),
            'spread': round(spread, 2),
            'spread_percent': round(spread_pct, 2),
            'volume': round(volume, 2),
            'volume_rub': round(volume * last_price, 2),
            'change_percent': round(change, 2),
            'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
            'date': now.strftime('%Y-%m-%d'),
            'time': now.strftime('%H:%M:%S')
        }
    except Exception as e:
        print(f"Ошибка получения {ticker}: {e}")
        return None


def get_stock_name(ticker):
    names = {
        'SBER': 'Сбербанк',
        'GAZP': 'Газпром',
        'LKOH': 'Лукойл',
        'ROSN': 'Роснефть',
        'VTBR': 'ВТБ',
        'TATN': 'Татнефть',
        'YNDX': 'Яндекс'
    }
    return names.get(ticker, ticker)


def save_to_database(data):
    if not data:
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO stock_prices (
            ticker, name, last_price, bid, offer,
            spread, spread_percent, volume, volume_rub, change_percent,
            timestamp, date, time
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data['ticker'], data['name'], data['last_price'],
        data['bid'], data['offer'], data['spread'],
        data['spread_percent'], data['volume'], data['volume_rub'],
        data['change_percent'], data['timestamp'],
        data['date'], data['time']
    ))

    cursor.execute('''
        INSERT OR REPLACE INTO latest_prices (
            ticker, name, last_price, bid, offer, spread, volume, change_percent, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data['ticker'], data['name'], data['last_price'],
        data['bid'], data['offer'], data['spread'],
        data['volume'], data['change_percent'], data['timestamp']
    ))

    conn.commit()
    conn.close()
    print(f"{data['timestamp']} - {data['ticker']}: {data['last_price']} руб (изм: {data['change_percent']:+.2f}%)")


def update_all_stocks():
    print(f"\n{datetime.now().strftime('%H:%M:%S')} - Обновление данных...")

    updated_count = 0
    for ticker in TICKERS:
        data = get_stock_data(ticker)
        if data:
            save_to_database(data)
            updated_count += 1

    if updated_count > 0:
        update_daily_stats()

    print(f"Обновление завершено. Добавлено {updated_count} записей из {len(TICKERS)}")


def update_daily_stats():
    conn = sqlite3.connect(DB_PATH)

    for ticker in TICKERS:
        today = datetime.now().strftime('%Y-%m-%d')

        query = '''
            SELECT
                MIN(last_price) as min_price,
                MAX(last_price) as max_price,
                AVG(last_price) as avg_price,
                SUM(volume) as volume_sum,
                COUNT(*) as update_count,
                (SELECT last_price FROM stock_prices
                 WHERE ticker = ? AND date = ?
                 ORDER BY timestamp ASC LIMIT 1) as start_price,
                (SELECT last_price FROM stock_prices
                 WHERE ticker = ? AND date = ?
                 ORDER BY timestamp DESC LIMIT 1) as end_price
            FROM stock_prices
            WHERE ticker = ? AND date = ?
        '''

        df = pd.read_sql_query(query, conn, params=(ticker, today, ticker, today, ticker, today))

        if not df.empty and df.iloc[0]['update_count'] > 0:
            row = df.iloc[0]
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO daily_stats
                (ticker, date, min_price, max_price, avg_price, volume_sum, update_count, start_price, end_price)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                ticker, today, row['min_price'], row['max_price'],
                row['avg_price'], row['volume_sum'], row['update_count'],
                row['start_price'], row['end_price']
            ))
            conn.commit()

    conn.close()


def check_data_changes():
    print("\nДИАГНОСТИКА: проверка текущих цен")

    for ticker in TICKERS:
        try:
            url = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.json"
            response = requests.get(url, timeout=10)
            data = response.json()

            market_data = data['marketdata']['data']
            columns = data['marketdata']['columns']

            last_idx = columns.index('LAST')
            last_price = market_data[0][last_idx]

            print(f"   {ticker}: {last_price} руб")
        except Exception as e:
            print(f"   {ticker}: ошибка - {e}")


def run_scheduler():
    schedule.every(COLLECT_INTERVAL).seconds.do(update_all_stocks)

    print(f"\nПАПКА ПРОЕКТА: {BASE_DIR}")
    print(f"ФАЙЛ БАЗЫ: {DB_PATH}")
    print(f"Интервал обновления: {COLLECT_INTERVAL} секунд")
    print("=" * 50)

    check_data_changes()
    print("\nЗапуск сбора данных")
    update_all_stocks()

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nСбор данных остановлен")

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM stock_prices")
        count = cursor.fetchone()[0]
        conn.close()

        print(f"\nИТОГОВАЯ СТАТИСТИКА:")
        print(f"   Всего записей в базе: {count}")


if __name__ == "__main__":
    print("=" * 50)
    print("Сборщик данных MOEX")
    print("=" * 50)

    init_database()

    if os.path.exists(DB_PATH):
        print(f"База данных готова: {DB_PATH}")
        print(f"Размер файла: {os.path.getsize(DB_PATH)} байт")

    run_scheduler()