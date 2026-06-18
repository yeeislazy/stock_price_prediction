from datetime import date, datetime
from pathlib import Path

# stock info
STOCK = "0166.kl"
START_DATE = "2023-01-01"
TIME_INTERVAL = "1d"

# time range
DEFAULT_TRAIN_YEARS = 2
DEFAULT_TEST_SIZE = 14  # days

# mlflow uri
MLFLOW_TRACKING_URI = "https://dagshub.com/yeeislazy/stock_price_prediction.mlflow"

# data paths
DATA_ROOT_DIR = Path(__file__).parent.parent.parent / "data" / STOCK.lower()
RAW_DATA_DIR = DATA_ROOT_DIR / "raw"
PROCESSED_DATA_DIR = DATA_ROOT_DIR / "processed"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

RAW_DATA_FILE = RAW_DATA_DIR / f"{STOCK.lower()}.parquet"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / f"{STOCK.lower()}.parquet"

NEWS_DATA_DIR = DATA_ROOT_DIR / "news"
NEWS_DATA_DIR.mkdir(parents=True, exist_ok=True)

COMPANY_NEWS_DIR = NEWS_DATA_DIR / "company"
COMPANY_NEWS_DIR.mkdir(parents=True, exist_ok=True)
INDUSTRY_NEWS_DIR = NEWS_DATA_DIR / "industry"
INDUSTRY_NEWS_DIR.mkdir(parents=True, exist_ok=True)

COMPANY_NEWS_FILE = COMPANY_NEWS_DIR / f"{STOCK.lower()}_company_news.parquet"
INDUSTRY_NEWS_FILE = INDUSTRY_NEWS_DIR / f"{STOCK.lower()}_industry_news.parquet"


# artifacts paths
ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "artifacts" / STOCK.lower()
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_SCALER_PATH = ARTIFACTS_DIR / "feature_scaler.pkl"
TARGET_SCALER_PATH = ARTIFACTS_DIR / "target_scaler.pkl"


# candidate model parameters
CANDIDATE_PARAMS = {
    "features":{
        "base_features": ["open", "high", "low", "close", "volume", "year", "month", "day"],
        "extra_features": [ "ma20", "rsi14", "ATR" ]
    },
    "targets": {
        "columns": ["return_2", "return_5", "return_14"],
        "weight_decay": [0.7, 0.8, 0.9]
    },
    "model": {
            "name": "LSTMModel",
            "input_size": 8,
            "output_size": 3,
            "projection_size": [32, 64, 128],
            "hidden_size": [64, 128, 256],
            "num_layers": [2, 3, 4],
            "seq_len": [30, 60],
            "num_epochs": 100,
            "lr": [1e-4 , 1e-3]
    }
}

# search keyword
STOCKS_KEYWORDS = {
    "QQQ": {
        "company_keywords":
            ["Invesco QQQ", "QQQ", "NVIDIA", "Microsoft", "Apple", "Amazon"]
    },
    "AAPL": {
        "company_keywords":
            ["Apple", "AAPL"]
    },
    "NVDA": {
        "company_keywords":
            ["NVIDIA", "NVDA"]
    },
    "0166.KL": {
        "company_keywords":
            ["Inari Amertron", "Inari Amertron Berhad", "0166.KL"],
        "industry_keywords":
            ["semiconductor", "OSAT", "photonics", "optical transceiver", "Broadcom"]
    }
}

SEARCH_KEYWORDS = STOCKS_KEYWORDS[STOCK.upper()]