import os

import mlflow
import mlflow.pyfunc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import time
import tempfile
from pathlib import Path
from mlflow.models.signature import infer_signature
from mlflow.models import ModelSignature
from mlflow.types import Schema, ColSpec
from torch import nn
from scripts.logger import get_logger
import boto3
import json
from itertools import combinations
from scripts.download_data import STOCK

from dotenv import load_dotenv


class LSTMModel(nn.Module):
    def __init__(self, input_size,output_size, projection_size=32, hidden_size=128, num_layers=2):
        super().__init__()
        
        # input projection layer
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, projection_size),
            nn.BatchNorm1d(projection_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=projection_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.shape
        
        # input projection
        x = x.view(batch_size * seq_len, -1)
        x = self.input_proj(x)
        x = x.view(batch_size, seq_len, -1)
        
        # LSTM
        out, _ = self.lstm(x)
        
        # get the last time step output
        out = out[:, -1, :]
        
        # output layer
        out = self.output_layer(out)
        
        return out

class testmodel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        # Take the last time step and apply linear layer
        x = x[:, -1, :]  # (batch_size, input_size)
        return self.linear(x)  # (batch_size, output_size)

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop
    
class StockPriceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_columns, target_columns,features_scaler, seq_len=30,targets_scaler=None):
        self.df = df
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.features_scaler = features_scaler
        self.targets_scaler = targets_scaler
        self.seq_len = seq_len
        
        # scale the features using the provided scaler
        self.scaled_features = self.features_scaler.transform(self.df[self.feature_columns])
        self.scaled_targets = self.targets_scaler.transform(self.df[self.target_columns]) if self.targets_scaler is not None else self.df[self.target_columns].values

    def __len__(self):
        return len(self.df) - self.seq_len + 1

    def __getitem__(self, idx):
        # Get the sequence of features
        seq = self.scaled_features[idx:idx + self.seq_len]
        # Get the target value
        target = self.scaled_targets[idx + self.seq_len - 1]
        
        # Convert to tensors (make copies to avoid non-writable array warning)
        seq = torch.FloatTensor(seq.copy())
        target = torch.FloatTensor(target.copy())
        return seq, target


# package the LSTM model and the scaler into a single MLflow pyfunc model
class LSTMWithScalerWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, features_scaler, feature_columns, output_size, projection_size=32, hidden_size=128, num_layers=2, seq_len=30, targets_scaler=None):
        self.features_scaler = features_scaler
        self.targets_scaler = targets_scaler
        self.feature_columns = feature_columns
        self.seq_len = seq_len
        self.model = None
        self.input_size = len(feature_columns)
        self.output_size = output_size
        self.projection_size = projection_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_context(self, context):
        # load the model state dict from the artifact path
        model_state_dict = mlflow.pytorch.load_model(context.artifacts["model_state"])
        
        # initialize the LSTM model and load the state dict
        self.model = LSTMModel(input_size=self.input_size, output_size=self.output_size, projection_size=self.projection_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, context: mlflow.pyfunc.PythonModelContext , model_input: pd.DataFrame) -> list:        
        # input validation: check if all required feature columns are present
        if not all(col in model_input.columns for col in self.feature_columns):
            raise ValueError(f"Missing columns. Expected: {self.feature_columns}")
        
        # extract the feature columns
        model_input = model_input[self.feature_columns].values
        
        # extract the last seq_len rows for each feature to form the input for LSTM
        if len(model_input) < self.seq_len:
            raise ValueError(f"Input length {len(model_input)} < seq_len {self.seq_len}")
        model_input = model_input[-self.seq_len:]
        
        scaled_input = self.features_scaler.transform(model_input)
        
        lstm_input_tensor = torch.FloatTensor(scaled_input).unsqueeze(0) # shape: (1, seq_len, input_size)
        
        # predict with the LSTM model
        with torch.no_grad():
            predictions = self.model(lstm_input_tensor.to(self.device)).cpu().numpy()
        
        # unscale the predictions if targets_scaler is available
        if self.targets_scaler is not None:
            predictions = self.targets_scaler.inverse_transform(predictions)
            
        return predictions.tolist()


def train_scaler(train_df, feature_columns, target_columns=None):
    features_scaler = StandardScaler()
    features_scaler.fit(train_df[feature_columns])
    
    features_scaler_path = Path(__file__).parent.parent.parent / "artifacts" /  STOCK.lower()/ "feature_scaler.pkl"
    features_scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(features_scaler, features_scaler_path)
    
    if target_columns is not None:
        targets_scaler = StandardScaler()
        targets_scaler.fit(train_df[target_columns])
        targets_scaler_path = Path(__file__).parent.parent.parent / "artifacts" / STOCK.lower()/ "target_scaler.pkl"
        targets_scaler_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(targets_scaler, targets_scaler_path)
        return features_scaler, targets_scaler

    return features_scaler

def get_schema(df: pd.DataFrame, feature_columns, target_columns):
    schema_dict = {
        "inputs": [{ "name": col, "type": df[col].dtype } for col in feature_columns],
        "outputs": [{ "name": col, "type": df[col].dtype } for col in target_columns]
    }
    return schema_dict

def get_signature(X_train, Y_train, model, dataset_name, version):
    sign = infer_signature(X_train,model.predict(X_train))
    
    if sign.outputs.inputs[0].type == 'object':
        new_output = Schema([ColSpec("string")])

        signature = ModelSignature(
            inputs=sign.inputs,
            outputs=new_output
        )
    else:
        signature = sign
    train_ds = mlflow.data.from_pandas(
        X_train.assign(Churn=Y_train),
        name=dataset_name + ':' + version
    )
    return signature, train_ds

def train_lstm(parameters,train_df,feature_columns,target_columns,features_scaler=None,targets_scaler=None,test_df=None, run_name=None, experiment_name='apple_stock_price_prediction'):  
    exp = mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name, experiment_id=exp.experiment_id) as run:  
        logger = get_logger(__name__)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        
        model.to(device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=parameters["lr"])
        
        early_stopping = EarlyStopping(patience=parameters["num_epochs"] // 10, min_delta=0.001)
        
        mlflow.log_params(parameters)
        
        for epoch in range(parameters["num_epochs"]):
            model.train()
            train_loss = 0
            train_targets_loss = {target: 0 for target in parameters["target_columns"]}
            
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
                val_loss = 0
                test_targets_loss = {target+"_test_loss": 0 for target in parameters["target_columns"]}
                model.eval()
                test_targets_unscaled = {target+"_test_unscaled_mae": 0 for target in parameters["target_columns"]}
                test_targets_direction_acc = {target+"_test_direction_acc": 0 for target in parameters["target_columns"]}

                target_to_idx = {
                    target: idx
                    for idx, target in enumerate(parameters["target_columns"])
                }
                
                for X_batch, Y_batch in test_dataloader:
                    X_batch = X_batch.to(device)
                    Y_batch = Y_batch.to(device)

                    with torch.no_grad():
                        outputs = model(X_batch)
                        loss_per_target = (outputs - Y_batch) ** 2
                        weighted_loss = loss_per_target * weights
                        loss = weighted_loss.sum(dim=1).mean()
                        val_loss += loss.item()
                        
                        for target in parameters["target_columns"]:
                            target_idx = target_to_idx[target]
                            test_targets_loss[target+"_test_loss"] += ((outputs[:, target_idx] - Y_batch[:, target_idx]) ** 2).mean().item()
                            
                        
                        # unscale the outputs and targets to evaluate actual difference and direction accuracy
                        if targets_scaler is not None:
                            outputs_unscaled = targets_scaler.inverse_transform(outputs.cpu().numpy())
                            Y_batch_unscaled = targets_scaler.inverse_transform(Y_batch.cpu().numpy())
                            
                            for target in parameters["target_columns"]:
                                target_idx = target_to_idx[target]
                                test_targets_unscaled[target+"_test_unscaled_mae"] += (abs(outputs_unscaled[:, target_idx] - Y_batch_unscaled[:, target_idx])).mean().item()
                                
                                test_targets_direction_acc[target+"_test_direction_acc"] += (
                                    ((outputs_unscaled[:, target_idx] > 0) == (Y_batch_unscaled[:, target_idx] > 0))
                                    .mean()
                                )
                                
                            
                    
                avg_val_loss = val_loss / len(test_dataloader)
                avg_test_targets_loss = {target+"_test_loss": test_targets_loss[target+"_test_loss"] / len(test_dataloader) for target in parameters["target_columns"]}
                avg_test_targets_unscaled = {target+"_test_unscaled_mae": test_targets_unscaled[target+"_test_unscaled_mae"] / len(test_dataloader) for target in parameters["target_columns"]}
                avg_test_targets_direction_acc = {target+"_test_direction_acc": test_targets_direction_acc[target+"_test_direction_acc"] / len(test_dataloader) for target in parameters["target_columns"]}
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch+1)
                mlflow.log_metrics(avg_test_targets_loss, step=epoch+1)
                mlflow.log_metrics(avg_test_targets_unscaled, step=epoch+1)
                mlflow.log_metrics(avg_test_targets_direction_acc, step=epoch+1)

            val_loss_str = f"{avg_val_loss:.4f}" if avg_val_loss is not None else "N/A"
            logger.info(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss_str}")
            
            if test_df is not None:
                if early_stopping(avg_val_loss):
                    logger.info("Early stopping triggered")
                    break
            else:
                if early_stopping(avg_train_loss):
                    logger.info("Early stopping triggered")
                    break                
            
        # log the scaler and model as artifacts in MLflow
        with tempfile.TemporaryDirectory() as tmp_dir:
            # log the scaler as an artifact in MLflow
            start_date = train_df['date'].min().strftime("%Y%m%d")
            end_date = train_df['date'].max().strftime("%Y%m%d")
            features_scaler_path = Path(tmp_dir) / f"features_stdscaler_{start_date}_{end_date}.pkl"
            targets_scaler_path = Path(tmp_dir) / f"targets_stdscaler_{start_date}_{end_date}.pkl"
            joblib.dump(features_scaler, features_scaler_path)
            joblib.dump(targets_scaler, targets_scaler_path)
            mlflow.log_artifact(str(features_scaler_path), artifact_path="preprocessor")
            mlflow.log_artifact(str(targets_scaler_path), artifact_path="preprocessor")

            ### log model to mlflow
            # save the model state dict to a temporary file
            model_path = Path(tmp_dir) / "lstm_model.pth"
            torch.save(model.state_dict(), model_path)
            
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
                model_info = mlflow.pyfunc.log_model(
                    artifact_path="lstm_model",
                    python_model=LSTMWithScalerWrapper(
                        features_scaler= features_scaler,feature_columns=feature_columns,
                        output_size=parameters["output_size"], projection_size=parameters["projection_size"], hidden_size=parameters["hidden_size"], num_layers=parameters["num_layers"], 
                        seq_len=parameters["seq_len"],
                        targets_scaler=targets_scaler
                        ),
                    artifacts=artifacts,
                    signature=signature)
            except Exception as e:
                logger.warning(f"Failed to log PyFunc model: {str(e)}. Falling back to PyTorch model logging.")
                mlflow.pytorch.log_model(model, artifact_path="lstm_model")
            
            
        #log schema
        schema = get_schema(train_df, feature_columns, parameters["target_columns"])
        mlflow.log_dict(schema, "schema.json")
        
        #log the training dataset as an MLflow artifact
        train_ds = mlflow.data.from_pandas(
            train_df[feature_columns + parameters["target_columns"]],
            name=f"aapl_train_dataset_{start_date}_{end_date}"
        )
        mlflow.log_input(train_ds, context="training")

def main():
    load_dotenv()
    
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(tracking_uri)
    
    
    train_df = pd.read_parquet(Path(__file__).parent.parent.parent / "data" / STOCK.lower() / "training" / f"{STOCK.lower()}_train.parquet")
    test_df = pd.read_parquet(Path(__file__).parent.parent.parent / "data" / STOCK.lower() / "training" / f"{STOCK.lower()}_test.parquet")

    start_date = train_df['date'].min().strftime("%Y%m%d")
    end_date = train_df['date'].max().strftime("%Y%m%d")

    try_parameters = json.load(open(Path(__file__).parent.parent / "training" / "parameters.json"))
    
    extra_features = try_parameters["features"]["extra_features"]
    extra_features_combinations = [[f] for f in extra_features]
    for l in range(2, len(extra_features) + 1):
        extra_features_combinations.extend(list(combinations(extra_features, l)))

    target_columns = try_parameters["targets"]['columns']

    ### try different combinations of parameters
    ## try different combinations of extra features
    for feature_combination in extra_features_combinations:
        feature_columns = ["open", "high", "low", "close", "volume"] + list(feature_combination)
        
        
        # try different weight decay for multi-task loss
        for weight_decay in try_parameters["targets"]["weight_decay"]:
            # try different projection size
            for projection_size in try_parameters["model"]["projection_size"]:
                #try different hidden size
                for hidden_size in try_parameters["model"]["hidden_size"]:
                    # try different number of layers
                    for num_layers in try_parameters["model"]["num_layers"]:
                        # sequence length
                        for seq_len in try_parameters["model"]["seq_len"]:        
                            for lr in [1e-4,1e-3]:
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
                                    experiment_name=f'{STOCK.lower()}_experiment'
                                )

if __name__ == "__main__":
    main()