import pandas as pd
import numpy as np
import sqlite3
import os
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "market_data.db")

TICKERS = ['SBER', 'GAZP', 'LKOH', 'ROSN', 'VTBR', 'TATN']

print("=" * 60)
print("ЗАПУСК ОБУЧЕНИЯ МОДЕЛИ")
print("=" * 60)


def train_model():
    if not os.path.exists(DB_PATH):
        print(f"\nБаза данных не найдена: {DB_PATH}")
        print("Сначала запустите сборщик данных: python download_sber.py")
        return False

    print("\nЗагрузка данных из базы SQLite...")
    conn = sqlite3.connect(DB_PATH)

    all_data = []
    for ticker in TICKERS:
        query = '''
            SELECT timestamp, last_price as close, volume, change_percent
            FROM stock_prices
            WHERE ticker = ?
            ORDER BY timestamp
        '''
        df = pd.read_sql_query(query, conn, params=(ticker,))

        if len(df) > 20:
            df['ticker'] = ticker
            all_data.append(df)
            print(f"  {ticker}: {len(df)} записей")
        else:
            print(f"  {ticker}: только {len(df)} записей (нужно больше 20)")

    conn.close()

    if len(all_data) < 2:
        print("\nНедостаточно данных для обучения")
        print("   Нужно минимум по 20 записей по каждому тикеру")
        return False

    total_rows = sum(len(df) for df in all_data)
    print(f"\nВсего данных: {total_rows} строк")

    df = pd.concat(all_data, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print("\nСоздание технических индикаторов...")

    def create_features(df_group):
        df_group = df_group.sort_values('timestamp').copy()

        for period in [1, 2, 3, 5, 10]:
            df_group[f'return_{period}'] = df_group['close'].pct_change(period) * 100

        for window in [5, 10, 20]:
            df_group[f'sma_{window}'] = df_group['close'].rolling(window, min_periods=1).mean()

        df_group['price_sma5'] = (df_group['close'] / df_group['sma_5'] - 1) * 100
        df_group['price_sma10'] = (df_group['close'] / df_group['sma_10'] - 1) * 100

        df_group['volatility_5'] = df_group['return_1'].rolling(5, min_periods=1).std().fillna(0)
        df_group['volatility_10'] = df_group['return_1'].rolling(10, min_periods=1).std().fillna(0)

        df_group['volume_sma'] = df_group['volume'].rolling(5, min_periods=1).mean()
        df_group['volume_ratio'] = (df_group['volume'] / df_group['volume_sma']).fillna(1)
        df_group['volume_change'] = df_group['volume'].pct_change().fillna(0) * 100

        df_group['momentum'] = df_group['return_1'] - df_group['return_1'].shift(1)

        horizon = min(10, len(df_group) // 4)
        if horizon < 1:
            horizon = 1
        df_group['target'] = (df_group['close'].shift(-horizon) - df_group['close']) / df_group['close'] * 100

        df_group = df_group.fillna(0).replace([np.inf, -np.inf], 0)
        return df_group

    df_features = df.groupby('ticker').apply(create_features).reset_index(drop=True)
    df_features = df_features[df_features['target'] != 0].dropna(subset=['target'])

    print(f"  После очистки: {len(df_features)} строк")

    if len(df_features) < 30:
        print("\nНедостаточно данных после очистки")
        print("Нужно минимум 30 строк для обучения")
        return False

    print("\nПодготовка признаков")

    feature_cols = [
        'return_1', 'return_2', 'return_3', 'return_5', 'return_10',
        'sma_5', 'sma_10', 'sma_20', 'price_sma5', 'price_sma10',
        'volatility_5', 'volatility_10', 'volume_ratio', 'volume_change',
        'momentum', 'change_percent'
    ]

    available_features = [f for f in feature_cols if f in df_features.columns]
    print(f"Используем признаков: {len(available_features)}")

    X = df_features[available_features]
    y = df_features['target']

    print("\nНормализация данных...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    split_idx = int(len(X) * 0.8)
    if split_idx < 2:
        split_idx = 2

    X_train = X_scaled[:split_idx]
    X_test = X_scaled[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    print(f"Обучающая выборка: {len(X_train)} строк")
    print(f"Тестовая выборка:  {len(X_test)} строк")

    print("\nОбучение модели CatBoost...")

    model = CatBoostRegressor(
        iterations=200,
        learning_rate=0.03,
        depth=4,
        early_stopping_rounds=20,
        verbose=50,
        random_seed=42
    )

    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=50)

    print("\nОценка качества модели")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE (средняя ошибка): {mae:.2f}%")
    print(f"R2 Score: {r2:.3f}")

    if r2 < 0:
        print("Модель хуже простого среднего, нужно больше данных")

    print("\nВажность признаков (топ-10):")
    importance = model.get_feature_importance()
    importance_df = pd.DataFrame({
        'Признак': available_features,
        'Важность': importance
    }).sort_values('Важность', ascending=False)

    for i, row in importance_df.head(10).iterrows():
        bar = "=" * int(row['Важность'] * 50)
        print(f"   {row['Признак']:15s}: {row['Важность']:.3f} {bar}")

    print("\nСохранение модели...")
    os.makedirs(BASE_DIR, exist_ok=True)

    model.save_model(os.path.join(BASE_DIR, 'ai_stock_model.cbm'))
    joblib.dump(scaler, os.path.join(BASE_DIR, 'scaler.pkl'))

    meta = {
        'model_type': 'CatBoostRegressor',
        'features': available_features,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'mae': float(mae),
        'r2': float(r2),
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_data': total_rows
    }

    with open(os.path.join(BASE_DIR, 'model_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nмодель обучена")
    print(f"Папка: {BASE_DIR}")
    print(f"Файлы: ai_stock_model.cbm, scaler.pkl, model_meta.json")

    return True


if __name__ == "__main__":
    success = train_model()

    if success:
        print("\nОбучение завершено, нужно запускать analyze_stocks.py")
    else:
        print("\nОбучение не выполнено")
        print("   Возможные причины:")
        print("   1. Нет данных в базе (запустите download_sber.py)")
        print("   2. Недостаточно записей (нужно минимум 50)")
        print("   3. Ошибка при загрузке данных")

    input("\nНажмите Enter для выхода")