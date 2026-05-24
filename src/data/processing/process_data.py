import pandas as pd
from pathlib import Path
from configuration.config import STOCK, RAW_DATA_DIR, PROCESSED_DATA_DIR


def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    return df

def compute_ma(df: pd.DataFrame, window: int=20) -> pd.DataFrame:
    df['ma' + str(window)] = df['close'].rolling(window=window).mean()
    return df

def compute_rsi(df: pd.DataFrame, window=14) -> pd.DataFrame:
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    df['rsi' + str(window)] = rsi
    return df

def compute_atr(df: pd.DataFrame) -> pd.DataFrame:
    # compute ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.DataFrame({'high_low': high_low, 'high_close': high_close, 'low_close': low_close}).max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    return df


def compute_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = extract_date_features(df)
    df = compute_ma(df)
    df = compute_rsi(df)
    df = compute_atr(df)
    df = df.ffill()
    return df

def compute_return_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df['return_2'] = (df['close'].shift(-2) - df['close']) / df['close']
    df['return_5'] = (df['close'].shift(-5) - df['close']) / df['close']
    df['return_14'] = (df['close'].shift(-14) - df['close']) / df['close']
    return df

def split_data(df: pd.DataFrame, test_size=0.2):
    len_df = len(df)
    train_size = int(len_df * (1 - test_size))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    return train_df, test_df

def main():
    df = pd.read_parquet(RAW_DATA_DIR / f"{STOCK.lower()}.parquet")
    df = compute_pipeline(df)
    df = compute_return_ratio(df)
    df.to_parquet(PROCESSED_DATA_DIR / f"{STOCK.lower()}.parquet", index=False)


if __name__ == "__main__":
    main()