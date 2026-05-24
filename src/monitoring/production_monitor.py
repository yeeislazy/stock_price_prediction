from dotenv import load_dotenv
import json
import mlflow
import torch
from torch.utils.data import DataLoader
from mlflow import MlflowClient, pyfunc
import os
from pathlib import Path
import pandas as pd
from datetime import datetime
from statistics import mean

from utils import logger
from data.ingestion.download_data import STOCK
from models.others import StockPriceDataset
from training.retrain_pipeline import request_model

def evaluate_model(model, metrics, agg_metrics, seq, target,feature_columns, target_columns) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        pred = model.predict(pd.DataFrame(seq.cpu().numpy(), columns=feature_columns))[0] # shape: (output_size), type: list
        actual = target.tolist()

    for i, col in enumerate(target_columns):
        if actual[i] is not None and actual[i] != 0:
            metrics[f"actual_{col}"] = actual[i]
            metrics[f"predicted_{col}"] = pred[i]
            metrics[f"error_{col}"] = abs(pred[i] - actual[i])
            metrics[f"error_pct_{col}"] = abs(pred[i] - actual[i]) / abs(actual[i])
            metrics[f'direction_correct_{col}'] = float(pred[i] * actual[i] > 0)
            agg_metrics[f"mae_{col}"].append(abs(pred[i] - actual[i]))
            agg_metrics[f"direction_accuracy_{col}"].append(float(pred[i] * actual[i] > 0))

    return metrics

def main():
    load_dotenv()
    
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(tracking_uri)
    
    model_name = 'stock-price-prediction-model'
    
    model, model_version, model_params = request_model(model_name=model_name,tags = 'Production')
        
    # get model info
    end_date = pd.to_datetime(model_params["end_date"])
    seq_len = model_params["seq_len"]
    feature_columns = model_params["feature_columns"] # ["open", "high", "low", "close", "volume", "year", "month", "day"]
    target_columns = model_params["target_columns"]
    
    # parameters for mlflow logging
    parameters = {
        "model_name": model_name,
        "model_version": model_version,
        "end_date": end_date.strftime("%Y-%m-%d"),
        "seq_len": seq_len,
        "feature_columns": feature_columns,
        "target_columns": target_columns
    }
        
    # load dataset
    dataset_path = Path(__file__).parent.parent.parent / "data" / STOCK.lower() / "training" / f"{STOCK.lower()}_test.parquet"
    test_df = pd.read_parquet(dataset_path)
    logger.info(f"Loaded test dataset with {len(test_df)} rows from {dataset_path}")
    
    # filter test data to only include rows after the model's training end date
    idx = test_df[test_df["date"] == end_date].index[0]
    test_df = test_df.iloc[idx + 1 - seq_len:]
    
    # create test dataset and dataloader
    test_dataset = StockPriceDataset(test_df, feature_columns, target_columns, seq_len=seq_len)
    
    agg_metrics = dict()
    for col in target_columns:
        agg_metrics[f"mae_{col}"] = []
        agg_metrics[f"direction_accuracy_{col}"] = []
    
    exp = mlflow.set_experiment('production_evaluation')
    
    now_date = datetime.now()
    run_name = f"evaluation_{STOCK.lower()}_{now_date.strftime('%Y-%m-%d')}"
    with mlflow.start_run(run_name=run_name, experiment_id=exp.experiment_id) as run:
        for i in range(test_dataset.__len__()):
            metrics = dict()
            logger.info(f"Test dataset sample {i}: features={test_dataset[i][0]}, target={test_dataset[i][1]}")
            seq, target = test_dataset[i]
            
            year = seq[-1][feature_columns.index("year")] if "year" in feature_columns else None
            month = seq[-1][feature_columns.index("month")] if "month" in feature_columns else None
            day = seq[-1][feature_columns.index("day")] if "day" in feature_columns else None
            
            date = datetime(year=int(year), month=int(month), day=int(day)) if year and month and day else None
            
            # evaluate the model
            metrics = evaluate_model(model, metrics, agg_metrics, seq, target, feature_columns, target_columns)
            
            # log the metrics to MLflow
            mlflow.log_metrics(metrics, step= int(date.timestamp()) if date else i)
            
        # log aggregate metrics
        for col in target_columns:
            agg_metrics[f"mae_{col}"] = mean(agg_metrics[f"mae_{col}"])
            agg_metrics[f"direction_accuracy_{col}"] = mean(agg_metrics[f"direction_accuracy_{col}"])
        
        mlflow.log_metrics(agg_metrics)