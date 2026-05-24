import yfinance as yf
import pandas as pd
from utils.logger import get_logger
import time

from configuration.config import STOCK, START_DATE, RAW_DATA_FILE

logger = get_logger(__name__)

def format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    df = df.rename(columns={
        df.columns[0]: "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })

    df["date"] = pd.to_datetime(df["date"])

    df['symbol'] = STOCK
    df = df[["date", "symbol", "open", "high", "low", "close", "volume"]]
    
    # filter data before today to avoid incomplete data for the current day
    today = pd.Timestamp.today().normalize()
    df = df[df["date"] < today]

    df = df.sort_values("date").reset_index(drop=True)
    
    return df

def fetch_data(start_date: str = START_DATE) -> pd.DataFrame:
    for i in range(5):
        df = yf.download(STOCK, start=start_date, progress=False)
        if not df.empty:
            logger.info(f"Fetched {len(df)} rows from Yahoo Finance")
            df = format_dataframe(df)
            return df
        wait = 5 * i
        logger.warning(f"Failed to fetch data. Retrying in {wait}s")
        time.sleep(wait)
    
    logger.error("Failed to fetch data after 5 attempts.")
    return pd.DataFrame()

def save_data(df: pd.DataFrame):
   
    df.to_parquet(RAW_DATA_FILE, index=False)
    logger.info(f"Saved {len(df)} rows to {RAW_DATA_FILE}")

def main():
    try:
        df_old = pd.read_parquet(RAW_DATA_FILE)

        latest_date = pd.to_datetime(df_old["date"].max())
        start_date = (latest_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        df_new = fetch_data(start_date)

        if df_new.empty:
            logger.info("No new data.")
            return df_old

        df = pd.concat([df_old, df_new])
        df = df.drop_duplicates(subset=["date"])
        df = df.sort_values("date")

        save_data(df)

        logger.info(
            f"Updated: {df_new['date'].iloc[0].date()} → {df_new['date'].iloc[-1].date()}"
        )
        logger.info(f"Total rows after update: {len(df)}")
        

        return df

    except FileNotFoundError:
        logger.warning("No existing data. Running full download.")
        df = fetch_data(START_DATE)
        if df.empty:
            logger.warning("No data fetched.")
            return pd.DataFrame()
        save_data(df)
        return df
    
if __name__ == "__main__":
    main()