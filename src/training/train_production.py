import os
import mlflow
import pandas as pd
from dotenv import load_dotenv

from data.ingestion.download_data import STOCK
from utils.logger import get_logger
from models.others import train_scaler
from training.train_candidate import train_lstm
from models.request_model import request_model
import argparse

from configuration.config import PROCESSED_DATA_FILE, DEFAULT_TRAIN_YEARS ,DEFAULT_TEST_SIZE
from utils.split_train_test import split_train_test
from utils.logger import get_logger

def main():
    load_dotenv()
    
    logger = get_logger(__name__)
    
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(tracking_uri)
    model_name = 'stock-price-prediction-model'  
    
    # argparse for command line arguments
    parser = argparse.ArgumentParser(description='Retrain the stock price prediction model.')
    parser.add_argument('--model_name', type=str, default=model_name, help='Name of the model to retrain')
    parser.add_argument('--model_tag', type=str, default='Production', help='Tag of the model to retrain')
    parser.add_argument('--train_years', type=int, default=DEFAULT_TRAIN_YEARS, help='Number of years of data to use for training')
    parser.add_argument('--test_size', type=int, default=DEFAULT_TEST_SIZE, help='Number of days to use for testing')
    
    args = parser.parse_args()
    
    # request model from MLflow
    tag = args.model_tag
    _, _, model_params, old_metrics = request_model(model_name=model_name, tags = tag)
    
    #load training data
    train_df = pd.read_parquet(PROCESSED_DATA_FILE)
    
    # filter data for retraining
    start_date = pd.to_datetime(train_df['date'].max()) - pd.DateOffset(years=DEFAULT_TRAIN_YEARS)
    train_df = train_df[train_df['date'] >= start_date].reset_index(drop=True)
    
    # drop incomplete rows
    train_df = train_df.dropna().reset_index(drop=True)
    
    # get dataset time range for experiment tracking
    start_date = train_df['date'].min().strftime("%Y%m%d")
    end_date = train_df['date'].max().strftime("%Y%m%d")
    
    model_params['data_period'] = f"{start_date}_{end_date}"

    target_columns = model_params["target_columns"]
    feature_columns = model_params["feature_columns"]

    features_scaler, targets_scaler = train_scaler(train_df, feature_columns, target_columns=target_columns)
    model_params["scaler"] = "standard_scaler"
    best_metrics = train_lstm(
        parameters=model_params,
        train_df=train_df,
        feature_columns=feature_columns,
        target_columns=target_columns,
        features_scaler=features_scaler,
        targets_scaler=targets_scaler,
        experiment_name=f'{STOCK.lower()}_{tag.lower()}_production',
        run_name=f'production_{STOCK.lower()}_{model_params["model"]}_{end_date}'
    )
    
    # compare metrics with old model
    if old_metrics['final_return_2_test_loss'] < best_metrics['final_return_2_test_loss']:
        logger.info(f"New model outperforms the old model. Old return_2 test loss: {old_metrics['final_return_2_test_loss']}, New return_2 test loss: {best_metrics['final_return_2_test_loss']}")
    else:
        logger.info(f"New model does not outperform the old model. Old return_2 test loss: {old_metrics['final_return_2_test_loss']}, New return_2 test loss: {best_metrics['final_return_2_test_loss']}")

if __name__ == "__main__":
    main()