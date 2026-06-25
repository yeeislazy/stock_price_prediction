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

from configuration.config import STOCK, START_DATE, RAW_DATA_FILE, COMPANY_NEWS_FILE, INDUSTRY_NEWS_FILE, SEARCH_KEYWORDS, NEWS_ANALYSIS_PERIOD

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

def fetch_stock_price_data(start_date: str = START_DATE) -> pd.DataFrame:
    for i in range(5):
        df = yf.download(STOCK, start=start_date, progress=False)
        if not df.empty:
            logger.info(f"Fetched {len(df)} rows from Yahoo Finance")
            df = format_dataframe(df)
            return df
        wait = 5 * i
        logger.warning(f"Failed to fetch stock price data. Retrying in {wait}s")
        time.sleep(wait)
    
    logger.error("Failed to fetch stock price data after 5 attempts.")
    return pd.DataFrame()

def fetch_dividend_data(start_date: str = START_DATE) -> pd.DataFrame:
    for i in range(5):
        divid_series = yf.Ticker(STOCK).dividends
        if not divid_series.empty:
            logger.info(f"Fetched {len(divid_series)} rows of dividend data from Yahoo Finance")
            df = pd.DataFrame(divid_series).reset_index()
            df = df.rename(columns={"Date": "date", "Dividends": "dividends"})
            df["date"] = df["date"].dt.tz_localize(None)
            df['date'] = pd.to_datetime(df['date'].dt.date)
            df = df[df["date"] >= pd.to_datetime(start_date)]
            return df
        wait = 5 * i
        logger.warning(f"Failed to fetch dividend data. Retrying in {wait}s")
        time.sleep(wait)
    
    logger.error("Failed to fetch dividend data after 5 attempts.")
    return pd.DataFrame()

def collect_stock_price_dividend(start_date: str = START_DATE) -> pd.DataFrame:
    stock_df = fetch_stock_price_data(start_date)
    dividend_df = fetch_dividend_data(start_date)

    if not dividend_df.empty:
        df = pd.merge(stock_df, dividend_df, on="date", how="left")
    else:
        df = stock_df.copy()
        df["dividend"] = None

    return df

def download_stock_data(raw_data_file: str = RAW_DATA_FILE, start_date: str = START_DATE):
    try:
        old_df = pd.read_parquet(raw_data_file)
    except Exception as e:
        logger.error(f"Could not read existing data from {raw_data_file}: {e}")
        old_df = None
    
    if old_df is not None and not old_df.empty:
        latest_date = pd.to_datetime(old_df["date"].max())
        start_date = (latest_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    
    new_df = collect_stock_price_dividend(start_date)
    
    if new_df.empty:
        logger.info("No new data.")
        return old_df if old_df is not None else pd.DataFrame()
    
    if old_df is not None and not old_df.empty:
        df = pd.concat([old_df, new_df])
        df = df.drop_duplicates(subset=["date"])
        df = df.sort_values("date").reset_index(drop=True)            
    else:
        df = new_df

    save_data(df, raw_data_file)
    
    logger.info(
        f"Updated: {new_df['date'].iloc[0].date()} → {new_df['date'].iloc[-1].date()}"
    )
    logger.info(f"Total rows after update: {len(df)}")
    
    return df

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
    if len(date_slides) == 0:
        date_slides = pd.Series([start_date, end_date])
    else:
        # ensure the first and last date are included in the slides
        if date_slides.iloc[0] > start_date:
            date_slides = pd.concat([pd.Series([pd.to_datetime(start_date)]), date_slides]).reset_index(drop=True)
        if date_slides.iloc[-1] < end_date:
            date_slides = pd.concat([date_slides, pd.Series([pd.to_datetime(end_date)])]).reset_index(drop=True)

    for i in range(len(date_slides) - 1):
        start = date_slides.iloc[i]
        end = date_slides.iloc[i + 1] - timedelta(days=1)  # end date is inclusive, so subtract 1 day
        try_interval = 6
        # get news
        for attempt in range(try_interval):
            try:
                logger.info(f"Collecting news from {start.date()} to {end.date()}, query: {query}, attempt {attempt + 1}/{try_interval}")
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
                    news = search_results.get('news', [])
                    logger.info(f"Collected {len(news)} news articles from {start.date()} to {end.date()}")
                    news_list.extend(news)
                    break
                else:
                    logger.error(f"Search request failed with status code {response.status_code} and message: {response.text}")
                    time.sleep(5 * (attempt + 1))
            except Exception as e:
                logger.error(f"Error during news collection: {e}")
                time.sleep(5 * (attempt + 1))
        else:
            logger.error(f"Failed to collect news from {start.date()} to {end.date()} after {try_interval} attempts.")
            break   # break the loop to avoid missing news of the fail time range

    if not news_list:
        logger.warning("No news articles collected.")
        return pd.DataFrame(columns=["title", "link", "snippet", "date", "source","imageUrl"])
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
        elif dtype.startswith("object") or dtype.startswith("string")or dtype.startswith("str"):
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

def download_news_data(file_path: str,query:str = '""' + '" OR "'.join(SEARCH_KEYWORDS['company_keywords']) + '"'):
    try:
        old_news_df = pd.read_parquet(file_path)
    except Exception as e:
        logger.error(f"Could not read existing news data from {file_path}: {e}")
        old_news_df = None
    
    if old_news_df is not None and not old_news_df.empty:
        start_date = pd.to_datetime(old_news_df["date"].max())
    else:
        start_date = datetime.strptime(START_DATE, '%Y-%m-%d') - timedelta(days=NEWS_ANALYSIS_PERIOD)
    
    end_date = datetime.today() - timedelta(days=1)
    
    new_news_df = collect_news(query=query, start_date=start_date, end_date=end_date)
    
    if old_news_df is not None and not old_news_df.empty:
        news_df = pd.concat([old_news_df, new_news_df])
        news_df = news_df.drop_duplicates(subset=["date"])
        news_df = news_df.sort_values("date").reset_index(drop=True)            
    else:
        news_df = new_news_df

    save_data(news_df, file_path)

def main():
    # download stock price data
    download_stock_data(RAW_DATA_FILE, START_DATE)
    
    # download company news data
    download_news_data(COMPANY_NEWS_FILE, query='"' + '" OR "'.join(SEARCH_KEYWORDS['company_keywords']) + '"')
    # download industry news data
    download_news_data(INDUSTRY_NEWS_FILE, query='"' + '" OR "'.join(SEARCH_KEYWORDS['industry_keywords']) + '"')
    
    
if __name__ == "__main__":
    main()