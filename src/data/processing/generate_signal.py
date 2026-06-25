import datetime
import json
from typing import Union
import pandas as pd
from google import genai
from dotenv import load_dotenv
import os
import time
import re
from pydantic import BaseModel, Field, field_validator

from configuration.config import SIGNAL_DATA_FILE, STOCK, COMPANY_NEWS_FILE, INDUSTRY_NEWS_FILE, START_DATE
from utils.logger import get_logger
from data.ingestion.download_data import save_data

logger = get_logger(__name__)

class SignalResult(BaseModel):
    stock: str = Field(..., description="Stock ticker symbol")
    date: Union[datetime.date, str] = Field(..., description="Date of the signal")
    return_2_signal: Union[int,float,str] = Field(...,  description="Signal for 2-day return")  
    return_5_signal: Union[int,float,str] = Field(..., description="Signal for 5-day return")
    return_14_signal: Union[int,float,str] = Field(..., description="Signal for 14-day return")
    confidence: Union[int,float,str] = Field(..., description="Confidence level of the signal")
    
    @field_validator('date', mode='before')
    def validate_date(cls, v):
        if isinstance(v, str):
            try:
                v = datetime.datetime.strptime(v, '%Y-%m-%d').date()
            except (ValueError, TypeError):
                raise ValueError(f"Date must be in 'YYYY-MM-DD' format, got {v}")
        return v
    
    @field_validator('return_2_signal', 'return_5_signal', 'return_14_signal', 'confidence', mode='before')
    def validate_returns(cls, v):
        if isinstance(v, str):
            try:
                v = float(v)
            except (ValueError, TypeError):
                raise ValueError(f"Return signals must be a float or a string that can be converted to float, got {v}")
        if isinstance(v, int):
            v = float(v)
        if not -1 <= v <= 1:
            raise ValueError("Return signals must be between -1 and 1")
        return v

def get_news(target_date: datetime.date, start_date: datetime.date = None) -> str:
    company_news_df = pd.read_parquet(COMPANY_NEWS_FILE)
    industry_news_df = pd.read_parquet(INDUSTRY_NEWS_FILE)

    # filter news for the past week
    if start_date is None:
        start_date = target_date - datetime.timedelta(days=7)

    # filter news for the given date
    if not company_news_df.empty:
        company_news = company_news_df[(company_news_df['date'].dt.date >= start_date) & (company_news_df['date'].dt.date <= target_date)]
    else:
        company_news = pd.DataFrame(columns=['title', 'date', 'source'])
    
    if not industry_news_df.empty:
        industry_news = industry_news_df[(industry_news_df['date'].dt.date >= start_date) & (industry_news_df['date'].dt.date <= target_date)]
    else:
        industry_news = pd.DataFrame(columns=['title', 'date', 'source'])
    
    # convert to string
    company_news_list = list()
    industry_news_list = list()
    for row in company_news.itertuples(index=False):
        company_news_list.append({"title": row.title, "date": row.date.strftime('%Y-%m-%d'),"source": row.source})
    for row in industry_news.itertuples(index=False):
        industry_news_list.append({"title": row.title, "date": row.date.strftime('%Y-%m-%d'),"source": row.source})
    
    company_news_json = json.dumps(company_news_list, indent=4)
    industry_news_json = json.dumps(industry_news_list, indent=4)

    return company_news_json, industry_news_json

# Signal generating prompt
def generate_signal_prompt(stock: str = STOCK, target_date: datetime.date = datetime.datetime.today().date(), range: int = 7) -> str:
    target_date_str = target_date.strftime('%Y-%m-%d')
    start_date = target_date - datetime.timedelta(days=range)
    
    company_news_json, industry_news_json = get_news(target_date, start_date)
    
    sentiment_prompt = f"""
    You are a professional equity research analyst.

    Use the following company and industry news  to estimate the expected impact on stock returns over the next:
    
    - 2 days
    - 5 days
    - 14 days

    
    Scoring Rules:

    - 1.0 = Extremely bullish, highly likely to positively impact price
    - 0.5 = Moderately bullish
    - 0.2 = Slightly bullish
    - 0.0 = Neutral or mixed signals
    - -0.2 = Slightly bearish
    - -0.5 = Moderately bearish
    - -1.0 = Extremely bearish

    Be conservative.
    Use 0 when signals are mixed or inconclusive.
    Do not exaggerate sentiment.


    Company News  (from {start_date.strftime('%Y-%m-%d')} to {target_date_str}):
    {company_news_json}
    
    Industry News Headlines (from {start_date.strftime('%Y-%m-%d')} to {target_date_str}):
    {industry_news_json}


    Return ONLY valid JSON.

    {{
        "stock": "{stock.upper()}",
        "date": "{target_date_str}",
        "return_2_signal": <float between -1 and 1>,
        "return_5_signal": <float between -1 and 1>,
        "return_14_signal": <float between -1 and 1>,
        "confidence": <float between 0 and 1>
    }}
    """

    return sentiment_prompt

# generate signal using LLM
def generate_signal(client, stock: str = STOCK, target_date: datetime.date = datetime.datetime.today().date(), range: int = 7) -> dict:
    
    prompt = generate_signal_prompt(stock, target_date, range)
    response = client.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt)

    json_match = re.search(
        r'\{[\s\S]*\}',
        response.text
    )
    text_response = json_match.group(0) if json_match else response.text
    
    try:
        result = json.loads(text_response)
        result['date'] = target_date
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing signal result: {e}, text: {text_response}")
        raise e

    result = SignalResult(**result).model_dump()  # Validate and convert to dict
    return result


def main() :
    try:
        df = pd.read_parquet(SIGNAL_DATA_FILE)
    except FileNotFoundError:
        logger.warning(f"Signal data file not found: {SIGNAL_DATA_FILE}, creating a new one.")
        df = pd.DataFrame(columns=["stock", "date", "return_2_signal", "return_5_signal", "return_14_signal", "confidence"])
    
    
    if not df.empty:
        # Ensure 'date' column is in datetime format
        df['date'] = pd.to_datetime(df['date'])
        # filter Missing Dates and empty rows
        nan_date = pd.Series(df['date'][df['date'].isna() | (df['return_2_signal'].isna()) | (df['return_5_signal'].isna()) | (df['return_14_signal'].isna()) | (df['confidence'].isna())])
        missing_dates = pd.Series(pd.date_range(start=df['date'].min(), end=df['date'].max()).difference(df['date']))
        combined_missing_dates = pd.concat([missing_dates, nan_date]).drop_duplicates().sort_values()
        
        start_date = df['date'].max() + pd.Timedelta(days=1)
        
    else:
        combined_missing_dates = pd.Series([])
        df['date'] = pd.to_datetime(df['date'])
        
        start_date = datetime.datetime.strptime(START_DATE, '%Y-%m-%d').date()

    
    end_date = datetime.datetime.today().date() - datetime.timedelta(days=1)
    
    dates_to_process = pd.concat([combined_missing_dates, pd.Series(pd.date_range(start=start_date, end=end_date))]).drop_duplicates().sort_values()
    
    load_dotenv()
    
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    retry_attempts = 10
    retry_delay = 1  # seconds
    
    for single_date in dates_to_process:
        success = False
        for attempt in range(retry_attempts):
            try:
                result = generate_signal(client, target_date=single_date.date())
                logger.info(f"Generated signal for {single_date.date()}: {result}")
                success = True
                break  # break the retry loop if successful
            except Exception as e:
                if hasattr(e, 'code') and e.code == 429:  # rate limit error
                    logger.warning(f"Rate limit reached for {single_date.date()}: {e}, breaking.")
                    break  # break the retry loop if rate limit error
                    
                else:
                    logger.error(f"Error generating signal for {single_date.date()}: {e}, waiting {retry_delay * (attempt + 1)} seconds before retrying.")
                    time.sleep(retry_delay * (attempt + 1) )  # wait before retrying
        if not success:
            logger.error(f"Failed to generate signal for {single_date.date()} after {retry_attempts} attempts.")
            break  # break the main loop if failed after retries
        if result:
            new_df = pd.DataFrame([result])
            if not df.empty:
                df = pd.concat([df, new_df], ignore_index=True)
            else:
                df = new_df
            
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(by='date').reset_index(drop=True)

            save_data(df, SIGNAL_DATA_FILE)
            logger.info(f"Signal data updated and saved to {SIGNAL_DATA_FILE}")
        
if __name__ == "__main__":
    main()   
