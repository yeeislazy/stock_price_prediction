import itertools
import os
import mlflow
import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import pandas as pd
import tempfile
import joblib
from pathlib import Path
import json
import copy
from mlflow.models.signature import infer_signature
from mlflow import pyfunc
from dotenv import load_dotenv
from itertools import combinations

from models.lstm import LSTMModel
from models.wrappers import LSTMWithScalerWrapper
from models.others import StockPriceDataset, EarlyStopping, train_scaler, get_schema
from utils.logger import get_logger
from utils.split_train_test import split_train_test
from utils.test_model import test_model
from configuration.config import STOCK, CANDIDATE_PARAMS, PROCESSED_DATA_FILE, MLFLOW_TRACKING_URI, DEFAULT_TRAIN_YEARS



def train_lstm(parameters,train_df,test_df,feature_columns,target_columns,features_scaler=None,targets_scaler=None, run_name=None, experiment_name='qqq_stock_price_prediction',model_name=None,alias=None):
    logger = get_logger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    exp = mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name, experiment_id=exp.experiment_id) as run:  
        train_dataset = StockPriceDataset(
            df=train_df, 
            feature_columns=feature_columns, 
            target_columns=target_columns,
            features_scaler=features_scaler, 
            seq_len=parameters["seq_len"],
            targets_scaler=targets_scaler
        )
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        if test_df is not None:
            test_dataset = StockPriceDataset(
                df=test_df,
                feature_columns=feature_columns,
                target_columns=target_columns,
                features_scaler=features_scaler,
                targets_scaler=targets_scaler,
                seq_len=parameters["seq_len"]
            )
            test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        model = LSTMModel(
            input_size=parameters["input_size"],
            output_size=parameters["output_size"],
            projection_size=parameters["projection_size"],
            hidden_size=parameters["hidden_size"],
            num_layers=parameters["num_layers"]
        )
        
        parameters["start_date"] = train_df['date'].min().strftime("%Y%m%d")
        parameters["end_date"] = train_df['date'].max().strftime("%Y%m%d")
        logger.info(f"Training data from {parameters['start_date']} to {parameters['end_date']}")
        
        model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=parameters["lr"])
        
        early_stopping = EarlyStopping(patience=parameters["num_epochs"] // 10, min_delta=0.001)
        
        # remaining weight allocation
        weight_decay = parameters.get("weight_decay", 0.7)
        num_targets = len(parameters["target_columns"])
        weights = []
        remaining_weight = 1.0
        for i in range(num_targets-1):
            current_weight = remaining_weight * weight_decay
            weights.append(current_weight)
            remaining_weight -= current_weight
        weights.append(remaining_weight) #eg: for 3 targets with weight_decay=0.7, weights would be [0.7, 0.21, 0.09]

        weights = torch.tensor(weights, device=device)
        
        mlflow.log_params(parameters)
        best_model = None
        best_metrics = None
        for epoch in range(parameters["num_epochs"]):
            model.train()
            train_loss = 0
            train_targets_loss = {target: 0 for target in parameters["target_columns"]}
            
            
            for X_batch, Y_batch in train_dataloader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                
                
                loss_per_target = (outputs - Y_batch) ** 2
                weighted_loss = loss_per_target * weights
                loss = weighted_loss.sum(dim=1).mean()
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                for target in parameters["target_columns"]:
                    target_idx = parameters["target_columns"].index(target)
                    train_targets_loss[target] += loss_per_target[:, target_idx].mean().item()
                
            avg_train_loss = train_loss / len(train_dataloader)
            avg_train_targets_loss = {target+"_train_loss": train_targets_loss[target] / len(train_dataloader) for target in parameters["target_columns"]}
            
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch+1)
            mlflow.log_metrics(avg_train_targets_loss, step=epoch+1)

                
            if test_df is not None:
                test_metrics = test_model(model, test_dataloader, device, parameters, targets_scaler=targets_scaler)
                mlflow.log_metrics(test_metrics, step=epoch+1)
            else:
                test_metrics = {}

            val_loss_str = f"{test_metrics.get('val_loss', 0):.4f}" if test_metrics.get('val_loss', 0) is not None else "N/A"
            logger.info(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss_str}")
            
            if test_df is not None:
                if best_model is None or (test_metrics.get('val_loss', 1) > best_metrics.get("final_val_loss", 2)):
                    best_model = copy.deepcopy(model.state_dict())
                    best_metrics = {
                        "final_train_loss": avg_train_loss,
                        "final_val_loss": test_metrics.get('val_loss', None)
                    }
                    for key, value in test_metrics.items():
                        best_metrics['final_'+key] = value
            else:
                if best_model is None or avg_train_loss < best_metrics.get("final_train_loss", float('inf')):
                    best_model = copy.deepcopy(model.state_dict())
                    best_metrics = {
                        "final_train_loss": avg_train_loss
                    }

            
            if test_df is not None:
                if early_stopping(test_metrics.get('val_loss', float('inf'))):
                    logger.info("Early stopping triggered")
                    break
            else:
                if early_stopping(avg_train_loss):
                    logger.info("Early stopping triggered")
                    break                
            
        # log the scaler and model as artifacts in MLflow
        with tempfile.TemporaryDirectory() as tmp_dir:
            # log the scaler as an artifact in MLflow
            features_scaler_path = Path(tmp_dir) / f"features_stdscaler_{parameters['start_date']}_{parameters['end_date']}.pkl"
            targets_scaler_path = Path(tmp_dir) / f"targets_stdscaler_{parameters['start_date']}_{parameters['end_date']}.pkl"
            joblib.dump(features_scaler, features_scaler_path)
            joblib.dump(targets_scaler, targets_scaler_path)
            mlflow.log_artifact(str(features_scaler_path), artifact_path="preprocessor")
            mlflow.log_artifact(str(targets_scaler_path), artifact_path="preprocessor")

            ### log model to mlflow
            # save the best model state dict to a temporary file
            model_path = Path(tmp_dir) / "lstm_model.pth"
            torch.save(best_model, model_path)
            
            artifacts = {
                "model_state": str(model_path),
                "features_scaler": str(features_scaler_path),
                "targets_scaler": str(targets_scaler_path)
            }
            print(X_batch.cpu()[0,:,:].unsqueeze(0).numpy().shape)
            print(Y_batch.cpu()[0,:].unsqueeze(0).numpy().shape)
            signature = infer_signature(
                model_input=train_df[feature_columns].iloc[-parameters["seq_len"]:],
                model_output=train_df[parameters["target_columns"]].iloc[-1:].values
            )
            
            try:
                model_info = pyfunc.log_model(
                    artifact_path="lstm_model",
                    python_model=LSTMWithScalerWrapper(
                        features_scaler= features_scaler,feature_columns=feature_columns,
                        output_size=parameters["output_size"], projection_size=parameters["projection_size"], hidden_size=parameters["hidden_size"], num_layers=parameters["num_layers"], 
                        seq_len=parameters["seq_len"],
                        targets_scaler=targets_scaler
                        ),
                    artifacts=artifacts,
                    signature=signature,
                    )
            except Exception as e:
                logger.warning(f"Failed to log PyFunc model: {str(e)}. Falling back to PyTorch model logging.")
                model.load_state_dict(best_model)
                model_info = mlflow.pytorch.log_model(model, artifact_path="lstm_model")
            
        # log the best model's metrics to MLflow
        mlflow.log_metrics(best_metrics, step=0)
            
        #log schema
        schema = get_schema(train_df, feature_columns, parameters["target_columns"])
        mlflow.log_dict(schema, "schema.json")
        
        #log configurations
        mlflow.log_dict(parameters, "config.json")
        
        #log the training dataset as an MLflow artifact
        train_ds = mlflow.data.from_pandas(
            train_df[feature_columns + parameters["target_columns"]],
            name=f"{STOCK.lower()}_train_dataset_{parameters['start_date']}_{parameters['end_date']}"
        )
        mlflow.log_input(train_ds, context="training")
        
        # register the model in MLflow Model Registry if model_name provided
        if model_name is not None:
            register_info = mlflow.register_model(model_info.model_uri, name=model_name)
            if alias is not None:
                version = register_info.version
                client = mlflow.tracking.MlflowClient()
                client.set_registered_model_alias(model_name, alias, version)
                logger.info(f"Registered model version {version} of {model_name} with alias '{alias}'")
    return model_info, best_metrics

def main():
    load_dotenv()
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    df = pd.read_parquet(PROCESSED_DATA_FILE)
    
    start_date = pd.to_datetime(df['date'].max()) - pd.DateOffset(years=DEFAULT_TRAIN_YEARS)
    df = df[df['date'] >= start_date].reset_index(drop=True)
    
    train_df, test_df = split_train_test(df, test_mode='period', test_size=14)
    
    end_date = train_df['date'].max().strftime("%Y%m%d")

    try_parameters = CANDIDATE_PARAMS
    
    extra_features = try_parameters["features"]["extra_features"]
    extra_features_combinations = [[f] for f in extra_features]
    for l in range(2, len(extra_features) + 1):
        extra_features_combinations.extend(list(combinations(extra_features, l)))

    base_features = try_parameters["features"]["base_features"]
    
    target_columns = try_parameters["targets"]["columns"]
    
    weight_decay_options = try_parameters["targets"]["weight_decay"]
    projection_size_options = try_parameters["model"]["projection_size"]
    hidden_size_options = try_parameters["model"]["hidden_size"]
    num_layers_options = try_parameters["model"]["num_layers"]
    seq_len_options = try_parameters["model"]["seq_len"]
    lr_options = try_parameters["model"]["lr"]
    
    

    # try different combinations of parameters
    for feature_combination, weight_decay, projection_size, hidden_size, num_layers, seq_len, lr in itertools.product(
        extra_features_combinations, weight_decay_options, projection_size_options, hidden_size_options, num_layers_options, seq_len_options, lr_options
    ):
        feature_columns = base_features + list(feature_combination)
        parameters = {
            "model": "lstm_model",
            "input_size": len(feature_columns),
            "feature_columns": feature_columns,
            "target_columns": target_columns,
            "weight_decay": weight_decay,
            "output_size": len(target_columns),
            "projection_size": projection_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "seq_len": seq_len,
            "num_epochs": 100,
            "lr": lr,
            "data_period": f"{start_date}-{end_date}",
            "scaler": None,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "cpu",
            "pytorch_version": torch.__version__,   
            "mlflow_version": mlflow.__version__
        }
        
        features_scaler, targets_scaler = train_scaler(train_df, feature_columns,target_columns=target_columns)
        parameters["scaler"] = "standard_scaler"
        train_lstm(
            parameters=parameters,
            train_df=train_df,
            feature_columns=feature_columns,
            target_columns=target_columns,
            features_scaler=features_scaler,
            targets_scaler=targets_scaler,
            test_df=test_df,
            experiment_name=f'{STOCK.lower()}_params_experiment'
        )

if __name__ == "__main__":
    main()