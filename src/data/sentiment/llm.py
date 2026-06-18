import datetime

from configuration.config import STOCK
from google import genai

# Sentimental score generating prompt
def generate_sentiment_prompt(stock: str = STOCK, target_date: datetime = datetime.date.today()) -> str:
    target_date_str = target_date.strftime("%Y-%m-%d")
    start_date_str = (target_date - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
    
    sentiment_prompt = f"""
    You are a professional equity research analyst.

    Today is {target_date_str}. Analyze the overall short-term sentiment for {stock.upper()} using information from the last 7 days.

    Search for information from {start_date_str} to {target_date_str}.
    
    Do NOT use any information published after {target_date_str}.

    Search and consider:

    1. Company-specific information
    - Earnings reports
    - SEC filings
    - Press releases
    - Product announcements
    - Executive changes
    - Guidance updates

    2. Industry and macroeconomic information
    - Industry trends
    - Regulatory developments
    - Competitor news
    - Interest rates
    - Economic indicators

    3. Market sentiment
    - Analyst upgrades/downgrades
    - Investor discussions
    - Social sentiment
    - Institutional activity
    
    Scoring Rules:

    - 1.0 = Extremely bullish, highly likely to positively impact price
    - 0.5 = Moderately bullish
    - 0.2 = Slightly bullish
    - 0.0 = Neutral or mixed signals
    - -0.2 = Slightly bearish
    - -0.5 = Moderately bearish
    - -1.0 = Extremely bearish

    The score should represent expected short-term impact on stock price over the next 1-2 weeks.

    Be conservative.
    Use 0 when signals are mixed or inconclusive.
    Do not exaggerate sentiment.

    Return ONLY valid JSON.

    {{
        "stock": "{stock.upper()}",
        "date": "{target_date_str}",
        "sentiment_index": <float between -1 and 1>,
        "confidence": <float between 0 and 1>,
        "reasoning": "<short reasoning>"
    }}
    """

    return sentiment_prompt

# 
