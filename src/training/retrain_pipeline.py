# This script is used to retrain and evaluate the model with new data and compare the performance with the old model. It can be run periodically (e.g., monthly) to ensure that the model is up-to-date and performs well with the latest data.

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
from utils.test_model import test_model

def main():
    load_dotenv()
    
    logger = get_logger(__name__)
    
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(tracking_uri)
    model_name = 'stock-price-prediction-model'  
    
    # argparse for command line arguments
    parser = argparse.ArgumentParser(description='Retrain and evaluate the stock price prediction model.')
    parser.add_argument('--model_name', type=str, default=model_name, help='Name of the model to retrain')
    parser.add_argument('--model_alias', type=str, default='Production', help='alias of the model to retrain')
    parser.add_argument('--train_years', type=int, default=DEFAULT_TRAIN_YEARS, help='Number of years of data to use for training')
    parser.add_argument('--test_size', type=int, default=DEFAULT_TEST_SIZE, help='Number of days to use for testing')
    
    args = parser.parse_args()
    
    # request model from MLflow
    alias = args.model_alias
    old_model, old_model_version, model_params, _ = request_model(model_name=model_name, alias=alias)
    
    #load training data
    df = pd.read_parquet(PROCESSED_DATA_FILE)
    
    # filter data for retraining
    start_date = pd.to_datetime(df['date'].max()) - pd.DateOffset(years=args.train_years)
    df = df[df['date'] >= start_date].reset_index(drop=True)
    
    # split train test
    train_df, test_df = split_train_test(df, test_mode='period', test_size=args.test_size)

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
    model_info, best_metrics = train_lstm(
        parameters=model_params,
        train_df=train_df,
        test_df=test_df,
        feature_columns=feature_columns,
        target_columns=target_columns,
        features_scaler=features_scaler,
        targets_scaler=targets_scaler,
        experiment_name=f'{STOCK.lower()}_{alias.lower()}_retraining',
        run_name=f'retraining_{STOCK.lower()}_{model_params["model"]}_{end_date}'
    )
    
    # test old model on the new data test set
    old_metrics = test_model(old_model, test_df, device='cpu', parameters=model_params, targets_scaler=targets_scaler)
    evaluate_target = target_columns[0]
    
    # log the old model's metrics to MLflow for comparison
    exp = mlflow.set_experiment(f'{STOCK.lower()}_{alias.lower()}_retraining')
    with mlflow.start_run(run_name=f'current_{alias.lower()}_evaluation_{STOCK.lower()}_{end_date}', experiment_id=exp.experiment_id) as eval_run:
        mlflow.set_tags({
            "evaluated_model": model_name,
            "evaluated_version": old_model_version,
            "evaluation_type": "post_deployment_test",
            "data_version": f"{start_date}_{end_date}"
        })
        mlflow.log_metrics(old_metrics)
        
        client = mlflow.MlflowClient()
        client.set_model_version_tag(
            model_name, 
            int(old_model_version), 
            "last_evaluated_run", 
            eval_run.info.run_id
        )
    
    # compare the new model's performance with the old model and register the new model if it outperforms the old model
    if best_metrics.get(f'final_{evaluate_target}_test_loss', float('inf')) < old_metrics.get(f'{evaluate_target}_test_loss', float('inf')):
        logger.info(f"New model outperforms the old model on {evaluate_target} test loss. Consider promoting the new model to production.")
        
        # register the new model in MLflow Model Registry with the same name and a new alias "Challenger"
        register_info = mlflow.register_model(model_info.model_uri, name=model_name)
        version = register_info.version
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(model_name, "Challenger", version)
        logger.info(f"Registered model version {version} of {model_name} with alias 'Challenger'")

if __name__ == "__main__":
    main()