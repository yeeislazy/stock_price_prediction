import yfinance as yf
import pandas as pd
import time
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import requests
import dateparser
import json

from utils.logger import get_logger
from data.ingestion.collect_news import collect_news

from configuration.config import STOCK, START_DATE, RAW_DATA_FILE, COMPANY_NEWS_FILE, INDUSTRY_NEWS_FILE, SEARCH_KEYWORDS

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

def collect_news(
    query:str = '""' + '" OR "'.join(SEARCH_KEYWORDS['company_keywords']) + '"',
    start_date: datetime = datetime.strptime(START_DATE, '%Y-%m-%d'),
    end_date: datetime = (datetime.today()- timedelta(days=1))
    ):
    # load Finnhub API key from environment variable
    load_dotenv()
    
    serper_api_key = os.getenv("SERPER_API_KEY")
    
    logger.info(f"Collecting news for {STOCK} from {start_date.date()} to {end_date.date()}")

    news_list = []

    # slide through date range in week intervals to avoid API limits
    date_slides = pd.Series(pd.date_range(start_date, end_date, freq='W'))
    
    # ensure the first and last date are included in the slides
    if date_slides.iloc[0] > start_date:
        date_slides = pd.concat([pd.Series([pd.to_datetime(start_date)]), date_slides]).reset_index(drop=True)
    if date_slides.iloc[-1] < end_date:
        date_slides = pd.concat([date_slides, pd.Series([pd.to_datetime(end_date)])]).reset_index(drop=True)

    for i in range(len(date_slides) - 1):
        start = date_slides.iloc[i]
        end = date_slides.iloc[i + 1] - timedelta(days=1)  # end date is inclusive, so subtract 1 day
        # get news
        response = requests.post(
            "https://google.serper.dev/news",
            headers={
                "X-API-KEY": serper_api_key,
                "Content-Type": "application/json"
                },
            json={
                "q": query,
                "gl": "my",
                "hl": "en",
                "tbs": f"cdr:1,cd_min:{start.strftime('%m/%d/%Y')},cd_max:{end.strftime('%m/%d/%Y')}"
            }
        )

        if response.status_code == 200:
            search_results = response.json()
            news_list.extend(search_results.get('news', []))
        else:
            logger.error(f"Search request failed with status code {response.status_code} and message: {response.text}")
        
    news_df = pd.DataFrame(news_list)
    
    def parse_serper_date(date_str):
        try:
            return pd.to_datetime(date_str)
        except Exception:

            return pd.to_datetime(dateparser.parse(date_str))

    news_df["date"] = news_df["date"].apply(parse_serper_date)
    news_df["date"] = pd.to_datetime(news_df["date"]).dt.normalize()
    news_df = news_df.drop_duplicates(subset=["link"])
    news_df = news_df.sort_values("date").reset_index(drop=True)

    return news_df

def save_schema(df: pd.DataFrame, path: str):
    schema = df.dtypes.apply(lambda x: x.name).to_dict()
    for col, dtype in schema.items():
        if dtype.startswith("datetime"):
            schema[col] = "datetime"
        elif dtype.startswith("int"):
            schema[col] = "int"
        elif dtype.startswith("float"):
            schema[col] = "float"
        elif dtype.startswith("object"):
            schema[col] = "string"
        else:
            schema[col] = "unknown"
    
    schema_path = path.with_suffix('.schema.json')
    with open(schema_path, 'w') as f:
        json.dump(schema, f)
    logger.info(f"Saved schema to {schema_path}")

def save_data(df: pd.DataFrame,path: str = RAW_DATA_FILE):
   
    df.to_parquet(path, index=False)
    logger.info(f"Saved {len(df)} rows to {path}")
    save_schema(df, path)

def download_news_data(file_path: str):
    try:
        old_news_df = pd.read_parquet(file_path)
    except Exception as e:
        logger.error(f"Could not read existing news data from {file_path}: {e}")
    
    if old_news_df is not None and not old_news_df.empty:
        start_date = pd.to_datetime(old_news_df["date"].max())
    else:
        start_date = datetime.strptime(START_DATE, '%Y-%m-%d')
    
    end_date = datetime.today() - timedelta(days=1)
    
    new_news_df = collect_news(start_date=start_date, end_date=end_date)
    
    if old_news_df is not None and not old_news_df.empty:
        news_df = pd.concat([old_news_df, new_news_df])
        news_df = news_df.drop_duplicates(subset=["date"])
        news_df = news_df.sort_values("date")            
    else:
        news_df = new_news_df

    save_data(news_df, file_path)

def main():
    # download stock price data
    try:
        stock_df_old = pd.read_parquet(RAW_DATA_FILE)

        latest_date = pd.to_datetime(stock_df_old["date"].max())
        start_date = (latest_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        stock_df_new = fetch_data(start_date)

        if stock_df_new.empty:
            logger.info("No new data.")
            return stock_df_old

        stock_df = pd.concat([stock_df_old, stock_df_new])
        stock_df = stock_df.drop_duplicates(subset=["date"])
        stock_df = stock_df.sort_values("date")

        save_data(stock_df, RAW_DATA_FILE)

        logger.info(
            f"Updated: {stock_df_new['date'].iloc[0].date()} → {stock_df_new['date'].iloc[-1].date()}"
        )
        logger.info(f"Total rows after update: {len(stock_df)}")
        
    except FileNotFoundError:
        logger.warning("No existing data. Running full download.")
        stock_df = fetch_data(START_DATE)
        if stock_df.empty:
            logger.warning("No data fetched.")
            return pd.DataFrame()
        save_data(stock_df, RAW_DATA_FILE)
    
    # download company news data
    download_news_data(COMPANY_NEWS_FILE)
    
    # download industry news data
    download_news_data(INDUSTRY_NEWS_FILE)    
    
    
if __name__ == "__main__":
    main()