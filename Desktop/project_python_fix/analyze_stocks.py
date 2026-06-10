import pandas as pd
import numpy as np
import sqlite3
import os
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "market_data.db")

TICKERS = ['SBER', 'GAZP', 'LKOH', 'ROSN', 'VTBR', 'TATN']

TICKER_NAMES = {
    'SBER': 'Сбербанк',
    'GAZP': 'Газпром',
    'LKOH': 'Лукойл',
    'ROSN': 'Роснефть',
    'VTBR': 'ВТБ',
    'TATN': 'Татнефть'
}


class StockAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = None
        self.model_loaded = self.load_model()

    def load_model(self):
        try:
            model_path = os.path.join(BASE_DIR, 'ai_stock_model.cbm')
            scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
            meta_path = os.path.join(BASE_DIR, 'model_meta.json')

            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = CatBoostRegressor()
                self.model.load_model(model_path)
                self.scaler = joblib.load(scaler_path)

                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                        self.features = meta.get('features')

                print("Модель загружена")
                return True
            else:
                print("Модель не найдена. Запустите train_model.py")
                return False
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return False

    def get_historical_stats(self, ticker, limit=100):
        conn = sqlite3.connect(DB_PATH)
        query = '''
            SELECT last_price as close, volume, timestamp
            FROM stock_prices
            WHERE ticker = ?
            ORDER BY timestamp DESC
            LIMIT ?
        '''
        df = pd.read_sql_query(query, conn, params=(ticker, limit))
        conn.close()

        if df.empty:
            return None

        df = df.sort_values('timestamp').reset_index(drop=True)

        stats = {
            'min_price': df['close'].min(),
            'max_price': df['close'].max(),
            'avg_price': df['close'].mean(),
            'current_price': df['close'].iloc[-1],
            'price_change': ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100),
            'volatility': df['close'].pct_change().std() * 100,
            'avg_volume': df['volume'].mean(),
            'records_count': len(df)
        }
        return stats

    def calculate_indicators(self, df):
        df = df.copy()

        for period in [1, 2, 3, 5, 10]:
            df[f'return_{period}'] = df['close'].pct_change(period) * 100

        for window in [5, 10, 20]:
            df[f'sma_{window}'] = df['close'].rolling(window, min_periods=1).mean()

        df['price_sma5'] = (df['close'] / df['sma_5'] - 1) * 100
        df['price_sma10'] = (df['close'] / df['sma_10'] - 1) * 100

        df['volatility_5'] = df['return_1'].rolling(5, min_periods=1).std().fillna(0)
        df['volatility_10'] = df['return_1'].rolling(10, min_periods=1).std().fillna(0)

        df['volume_sma'] = df['volume'].rolling(5, min_periods=1).mean()
        df['volume_ratio'] = (df['volume'] / df['volume_sma']).fillna(1)
        df['volume_change'] = df['volume'].pct_change().fillna(0) * 100

        df['momentum'] = df['return_1'] - df['return_1'].shift(1)

        df = df.fillna(0).replace([np.inf, -np.inf], 0)
        return df

    def predict_ticker(self, ticker):
        if not self.model_loaded:
            return None, "Модель не загружена"

        try:
            conn = sqlite3.connect(DB_PATH)
            query = '''
                SELECT timestamp, last_price as close, volume, change_percent
                FROM stock_prices
                WHERE ticker = ?
                ORDER BY timestamp DESC
                LIMIT 30
            '''
            df = pd.read_sql_query(query, conn, params=(ticker,))
            conn.close()

            if df.empty or len(df) < 10:
                return None, "Недостаточно данных"

            df = df.sort_values('timestamp').reset_index(drop=True)
            df = self.calculate_indicators(df)
            last_data = df.iloc[-1:].copy()

            if self.features is None:
                self.features = [
                    'return_1', 'return_2', 'return_3', 'return_5', 'return_10',
                    'sma_5', 'sma_10', 'sma_20', 'price_sma5', 'price_sma10',
                    'volatility_5', 'volatility_10', 'volume_ratio', 'volume_change',
                    'momentum', 'change_percent'
                ]

            available = [f for f in self.features if f in last_data.columns]
            X = last_data[available]

            X_scaled = self.scaler.transform(X)
            prediction = self.model.predict(X_scaled)[0]

            stats = self.get_historical_stats(ticker, limit=100)

            return {
                'ticker': ticker,
                'name': TICKER_NAMES.get(ticker, ticker),
                'current_price': stats['current_price'],
                'prediction': prediction,
                'min_price_100': stats['min_price'],
                'max_price_100': stats['max_price'],
                'avg_price_100': stats['avg_price'],
                'price_change': stats['price_change'],
                'volatility': stats['volatility'],
                'records': stats['records_count']
            }, None
        except Exception as e:
            return None, str(e)

    def get_signal(self, prediction):
        if prediction > 0.5:
            return "СИЛЬНЫЙ СИГНАЛ К ПОКУПКЕ", "green"
        elif prediction > 0.2:
            return "СЛАБЫЙ СИГНАЛ К ПОКУПКЕ", "yellow"
        elif prediction < -0.5:
            return "СИГНАЛ К ПРОДАЖЕ", "red"
        elif prediction < -0.2:
            return "РЕКОМЕНДУЕТСЯ ДЕРЖАТЬ", "orange"
        else:
            return "НЕТ ЯСНОГО СИГНАЛА", "gray"

    def analyze_all(self):
        results = []
        print("\n" + "=" * 100)
        print(f"АНАЛИЗ АКЦИЙ - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 100)

        for ticker in TICKERS:
            result, error = self.predict_ticker(ticker)
            if result:
                results.append(result)
        return results


def print_table(results):
    if not results:
        print("\nНет данных для отображения")
        return

    print("\n" + "=" * 120)
    print(f"{'Тикер':<8} {'Компания':<12} {'Цена':>10} {'Мин/100':>10} "
          f"{'Макс/100':>10} {'Ср.цена':>10} {'Изменение':>10} {'Прогноз':>10} {'Сигнал':<30}")
    print("=" * 120)

    results.sort(key=lambda x: x['prediction'], reverse=True)

    for r in results:
        signal, color = get_signal_color(r['prediction'])
        print(f"{r['ticker']:<8} {r['name']:<12} {r['current_price']:>10.2f} руб "
              f"{r['min_price_100']:>10.2f} руб {r['max_price_100']:>10.2f} руб "
              f"{r['avg_price_100']:>10.2f} руб {r['price_change']:>+9.2f}% "
              f"{r['prediction']:>+9.2f}%  {signal:<30}")


def get_signal_color(prediction):
    if prediction > 0.5:
        return "КУПИТЬ (сильный)", "green"
    elif prediction > 0.2:
        return "КУПИТЬ (слабый)", "yellow"
    elif prediction < -0.5:
        return "ПРОДАТЬ", "red"
    elif prediction < -0.2:
        return "ДЕРЖАТЬ", "orange"
    else:
        return "НЕТ СИГНАЛА", "gray"


def print_detailed_analysis(analyzer, ticker):
    result, error = analyzer.predict_ticker(ticker)

    print("\n" + "=" * 70)
    print(f"ДЕТАЛЬНЫЙ АНАЛИЗ: {result['name']} ({result['ticker']})")
    print("=" * 70)
    print(f"\nТЕКУЩАЯ ЦЕНА: {result['current_price']:.2f} руб")
    print(f"\nСТАТИСТИКА ЗА ПОСЛЕДНИЕ 100 ЗАПИСЕЙ:")
    print(f"   Минимальная цена:  {result['min_price_100']:.2f} руб")
    print(f"   Максимальная цена: {result['max_price_100']:.2f} руб")
    print(f"   Средняя цена:      {result['avg_price_100']:.2f} руб")
    print(f"   Отклонение от ср.: {((result['current_price'] - result['avg_price_100']) / result['avg_price_100'] * 100):+.2f}%")
    print(f"   Общее изменение:   {result['price_change']:+.2f}%")
    print(f"   Волатильность:     {result['volatility']:.2f}%")

    signal, _ = get_signal_color(result['prediction'])
    print(f"\nПРОГНОЗ МОДЕЛИ:")
    print(f"   {signal}")
    print(f"   Ожидаемая доходность через 10 минут: {result['prediction']:+.2f}%")

    print(f"\nРЕКОМЕНДАЦИИ:")
    if result['prediction'] > 0.3:
        stop_loss = result['current_price'] * 0.98
        take_profit = result['current_price'] * (1 + result['prediction'] / 100)
        print(f"Рекомендуется: ПОКУПКА")
        print(f"Стоп-лосс:   {stop_loss:.2f} руб (-2%)")
        print(f"Тейк-профит: {take_profit:.2f} руб (+{result['prediction']:.1f}%)")
    elif result['prediction'] < -0.3:
        print(f"Рекомендуется: ВОЗДЕРЖАТЬСЯ ОТ ПОКУПКИ")
        print(f"Рекомендуемый стоп-лосс: {result['current_price'] * 0.95:.2f} руб")
    else:
        print(f"Рекомендуется: НАБЛЮДЕНИЕ")
        print(f"Нет явных сигналов, рынок в боковом движении")
    print("=" * 70)


def main():
    print("=" * 60)
    print("АНАЛИЗАТОР АКЦИЙ С ПРОГНОЗОМ")
    print("=" * 60)

    analyzer = StockAnalyzer()

    if not analyzer.model_loaded:
        print("\nМодель не найдена")
        print("   Сначала запустите обучение: python train_model.py")
        input("\nEnter для выхода")
        return

    while True:
        print("\n" + "-" * 60)
        print("Выберите действие:")
        print("1 - Полный анализ всех акций")
        print("2 - Детальный анализ конкретной акции")
        print("3 - Только сигналы к покупке")
        print("4 - Обновить данные")
        print("0 - Выход")

        choice = input("\nВаш выбор (0-4): ").strip()

        if choice == "0":
            break
        elif choice == "1":
            results = analyzer.analyze_all()
            print_table(results)
            buy_signals = [r for r in results if r['prediction'] > 0.2]
            if buy_signals:
                print(f"\nЛУЧШИЕ КАНДИДАТЫ ДЛЯ ПОКУПКИ:")
                for r in buy_signals[:3]:
                    print(f"   {r['ticker']} ({r['name']}): +{r['prediction']:.1f}%")
            else:
                print("\nСейчас нет явных сигналов к покупке")
        elif choice == "2":
            print("\nДоступные тикеры:", ", ".join(TICKERS))
            ticker = input("Введите тикер: ").strip().upper()
            if ticker in TICKERS:
                print_detailed_analysis(analyzer, ticker)
            else:
                print(f"Неверный тикер. Доступны: {', '.join(TICKERS)}")
        elif choice == "3":
            results = analyzer.analyze_all()
            buy_signals = [r for r in results if r['prediction'] > 0.2]
            if buy_signals:
                print("\nСИГНАЛЫ К ПОКУПКЕ:")
                print("-" * 60)
                for r in buy_signals:
                    print(f"{r['ticker']} ({r['name']}):")
                    print(f"   Цена: {r['current_price']:.2f} руб")
                    print(f"   Ожидаемый рост: +{r['prediction']:.1f}%")
                    print(f"   Мин/Макс за 100: {r['min_price_100']:.2f} / {r['max_price_100']:.2f} руб")
                    print()
            else:
                print("\nНет сигналов к покупке")
        elif choice == "4":
            print("Обновление данных...")
            results = analyzer.analyze_all()
            print_table(results)
            print("\nДанные обновлены")
        else:
            print("Неверный выбор")


if __name__ == "__main__":
    main()