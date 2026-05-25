from pathlib import Path

# stock info
STOCK = "qqq"
START_DATE = "2020-01-01"

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