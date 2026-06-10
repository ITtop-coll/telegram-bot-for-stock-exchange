import asyncio
import sqlite3
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from catboost import CatBoostRegressor
import joblib
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "market_data.db")
MODEL_PATH = os.path.join(BASE_DIR, "ai_stock_model.cbm")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

TOKEN = os.getenv("BOT_TOKEN")

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

TICKERS = ['SBER', 'GAZP', 'LKOH', 'ROSN', 'VTBR', 'TATN']
NAMES = {'SBER': 'Сбербанк', 'GAZP': 'Газпром', 'LKOH': 'Лукойл', 
         'ROSN': 'Роснефть', 'VTBR': 'ВТБ', 'TATN': 'Татнефть'}


def bot_db():
    conn = sqlite3.connect(os.path.join(BASE_DIR, "bot_users.db"))
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (user_id INTEGER PRIMARY KEY, subscribed INTEGER DEFAULT 0, threshold REAL DEFAULT 0.5)''')
    c.execute('''CREATE TABLE IF NOT EXISTS tracked 
                 (user_id INTEGER, ticker TEXT, PRIMARY KEY(user_id, ticker))''')
    c.execute('''CREATE TABLE IF NOT EXISTS alerts 
                 (alert_id TEXT PRIMARY KEY)''')
    conn.commit()
    return conn


class Analyzer:
    def __init__(self):
        self.model = CatBoostRegressor()
        self.model.load_model(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        with open(os.path.join(BASE_DIR, "model_meta.json"), encoding='utf-8') as f:
            self.features = json.load(f).get('features')
    
    def get_data(self, ticker):
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            'SELECT timestamp, last_price as close, volume, change_percent FROM stock_prices WHERE ticker=? ORDER BY timestamp DESC LIMIT 30',
            conn, params=(ticker,))
        conn.close()
        return df.sort_values('timestamp').reset_index(drop=True) if len(df) >= 10 else None
    
    def calc_features(self, df):
        df = df.copy()
        
        for p in [1, 2, 3, 5, 10]:
            df[f'return_{p}'] = df['close'].pct_change(p) * 100
            
        for w in [5, 10, 20]:
            df[f'sma_{w}'] = df['close'].rolling(w, min_periods=1).mean()
            
        df['price_sma5'] = (df['close'] / df['sma_5'] - 1) * 100
        df['price_sma10'] = (df['close'] / df['sma_10'] - 1) * 100
        df['volatility_5'] = df['return_1'].rolling(5, min_periods=1).std().fillna(0)
        df['volatility_10'] = df['return_1'].rolling(10, min_periods=1).std().fillna(0)
        df['volume_ratio'] = (df['volume'] / df['volume'].rolling(5, min_periods=1).mean()).fillna(1)
        df['volume_change'] = df['volume'].pct_change().fillna(0) * 100
        df['momentum'] = df['return_1'] - df['return_1'].shift(1)
        
        return df.fillna(0).replace([np.inf, -np.inf], 0)
    
    def predict(self, ticker):
        df = self.get_data(ticker)
        if df is None:
            return None
        X = self.calc_features(df).iloc[-1:][self.features]
        pred = self.model.predict(self.scaler.transform(X))[0]
        
        signal_map = {
            lambda p: p > 0.5: '🔴 ПОКУПКА',
            lambda p: p > 0.2: '🟡 СЛАБАЯ',
            lambda p: p > -0.2: '⚪ НЕТ',
            lambda p: p > -0.5: '🟠 ДЕРЖАТЬ'
        }
        signal = next((v for k, v in signal_map.items() if k(pred)), '🟢 ПРОДАЖА')
        
        return {
            'ticker': ticker,
            'name': NAMES[ticker],
            'price': round(df['close'].iloc[-1], 2),
            'pred': round(pred, 2),
            'signal': signal
        }
    
    def top(self, min_pred=0.2):
        return [p for p in [self.predict(t) for t in TICKERS] if p and p['pred'] > min_pred]


analyzer = Analyzer()


def get_user(user_id):
    conn = bot_db()
    c = conn.cursor()
    c.execute('SELECT subscribed, threshold FROM users WHERE user_id=?', (user_id,))
    row = c.fetchone()
    conn.close()
    return row if row else (0, 0.5)


def set_user(user_id, subscribed=None, threshold=None):
    conn = bot_db()
    c = conn.cursor()
    c.execute('INSERT OR IGNORE INTO users (user_id) VALUES (?)', (user_id,))
    if subscribed is not None:
        c.execute('UPDATE users SET subscribed=? WHERE user_id=?', (1 if subscribed else 0, user_id))
    if threshold is not None:
        c.execute('UPDATE users SET threshold=? WHERE user_id=?', (threshold, user_id))
    conn.commit()
    conn.close()


def get_tracked(user_id):
    conn = bot_db()
    c = conn.cursor()
    c.execute('SELECT ticker FROM tracked WHERE user_id=?', (user_id,))
    rows = c.fetchall()
    conn.close()
    return [r[0] for r in rows]


def alert_sent(alert_id):
    conn = bot_db()
    c = conn.cursor()
    c.execute('SELECT 1 FROM alerts WHERE alert_id=?', (alert_id,))
    exists = c.fetchone()
    if not exists:
        c.execute('INSERT INTO alerts VALUES (?)', (alert_id,))
    conn.commit()
    conn.close()
    return exists is not None


def format_stock(s):
    exp = s['price'] * (1 + s['pred'] / 100)
    return f"📈 *{s['name']} ({s['ticker']})*\n💰 {s['price']}₽ → {exp:.2f}₽\n📊 {s['pred']:+.2f}% | {s['signal']}"


@dp.message_handler(commands=['start'])
async def start(m):
    set_user(m.from_user.id)
    await m.answer(f"Привет Я AI бот для анализа акций MOEX\n\n"
                   f"Доступные акции: SBER, GAZP, LKOH, ROSN, VTBR, TATN\n\n"
                   f"Команды:\n/top - выгодные акции\n/analyze TICKER - анализ\n/subscribe - подписка\n/settings - порог\n/help - помощь")


@dp.message_handler(commands=['help'])
async def help_cmd(m):
    await m.answer("/top - топ акций\n/analyze SBER - анализ\n/subscribe - подписаться\n/unsubscribe - отписаться\n/track - мои акции\n/settings - настройки")


@dp.message_handler(commands=['top'])
async def top_cmd(m):
    await m.answer("Анализирую...")
    top = analyzer.top()
    if top:
        for s in top[:3]:
            await m.answer(format_stock(s), parse_mode='Markdown')
    else:
        await m.answer("Выгодных акций не найдено")


@dp.message_handler(commands=['analyze'])
async def analyze_cmd(m):
    args = m.get_args().upper()
    if not args or args not in TICKERS:
        await m.answer(f"Пример: /analyze SBER\nДоступны: {', '.join(TICKERS)}")
        return
    await m.answer("Анализирую...")
    s = analyzer.predict(args)
    if s:
        await m.answer(format_stock(s), parse_mode='Markdown')
    else:
        await m.answer("Недостаточно данных")


@dp.message_handler(commands=['subscribe'])
async def subscribe(m):
    sub, thresh = get_user(m.from_user.id)
    if sub:
        await m.answer("Вы уже подписаны")
    else:
        set_user(m.from_user.id, subscribed=True)
        await m.answer("Подписка оформлена, буду присылать уведомления.")


@dp.message_handler(commands=['unsubscribe'])
async def unsubscribe(m):
    set_user(m.from_user.id, subscribed=False)
    await m.answer("Вы отписались")


@dp.message_handler(commands=['track'])
async def track_list(m):
    tracked = get_tracked(m.from_user.id)
    if not tracked:
        await m.answer("Список пуст.")
        return
    
    results = [analyzer.predict(t) for t in tracked if analyzer.predict(t)]
    
    if results:
        msg = "*📋 Отслеживаемые:*\n\n" + "\n".join([f"• {s['ticker']}: {s['price']}₽ | {s['pred']:+.2f}%" for s in results])
        await m.answer(msg, parse_mode='Markdown')
    else:
        await m.answer("не удалось получить данные по отслеживаемым акциям.")


@dp.message_handler(commands=['settings'])
async def settings(m):
    sub, thresh = get_user(m.from_user.id)
    kb = InlineKeyboardMarkup(row_width=3)
    kb.add(*[InlineKeyboardButton(f"{p}%", callback_data=f"thresh_{p}") for p in [0.2, 0.5, 1, 2, 3, 5]])
    await m.answer(f"Текущий порог: {thresh}%\nВыберите новый:", reply_markup=kb)


@dp.callback_query_handler(lambda c: c.data.startswith('thresh_'))
async def set_thresh(cb):
    thresh = float(cb.data.split('_')[1])
    set_user(cb.from_user.id, threshold=thresh)
    await cb.answer(f"порог {thresh}%")
    await cb.message.edit_text(f"порог установлен на {thresh}%")


async def auto_notify():
    while True:
        try:
            top = analyzer.top()
            if top:
                conn = bot_db()
                c = conn.cursor()
                c.execute('SELECT user_id, threshold FROM users WHERE subscribed=1')
                users = c.fetchall()
                conn.close()
                
                for user_id, threshold in users:
                    for s in top:
                        if s['pred'] >= threshold:
                            alert_id = f"{s['ticker']}_{datetime.now().strftime('%Y%m%d%H')}"
                            if not alert_sent(alert_id):
                                await bot.send_message(user_id, format_stock(s), parse_mode='Markdown')
                                await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Ошибка в авто-уведомлениях: {e}")
        await asyncio.sleep(3600)


async def on_startup():
    asyncio.create_task(auto_notify())
    print("Бот запущен")


if __name__ == '__main__':
    async def main():
        await on_startup()
        await dp.start_polling(bot)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Бот остановлен")