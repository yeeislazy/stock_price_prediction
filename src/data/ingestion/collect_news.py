from urllib import response

import finnhub
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
import os
import requests

from configuration.config import SEARCH_KEYWORDS, START_DATE, STOCK

def request_serper_news(query, start, end, serper_api_key, news_list):
    # get company news
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
            print(f"Search request failed with status code {response.status_code} and message: {response.text}")

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
    date_slides = pd.date_range(start_date, end_date, freq='W')
    
    # ensure the first and last date are included in the slides
    if date_slides[0] > start_date:
        date_slides = pd.concat([pd.Series([pd.to_datetime(start_date)]), pd.Series(date_slides)]).reset_index(drop=True)
    if date_slides[-1] < end_date:
        date_slides = pd.concat([pd.Series(date_slides), pd.Series([pd.to_datetime(end_date)])]).reset_index(drop=True)

    for i in range(len(date_slides) - 1):
        start = date_slides[i]
        end = date_slides[i + 1] - timedelta(days=1)  # end date is inclusive, so subtract 1 day
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
            print(f"Search request failed with status code {response.status_code} and message: {response.text}")
        
    news_df = pd.DataFrame(news_list)

    return news_df