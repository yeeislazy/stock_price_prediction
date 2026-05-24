import mlflow
from mlflow.models import ModelSignature, ModelSignature, infer_signature
from mlflow.types import ColSpec, Schema, Schema
import pandas as pd
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib

from configuration.config import FEATURE_SCALER_PATH, TARGET_SCALER_PATH


class StockPriceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_columns, target_columns,features_scaler = None, seq_len=30,targets_scaler=None):
        self.df = df
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.features_scaler = features_scaler
        self.targets_scaler = targets_scaler
        self.seq_len = seq_len
        

    def __len__(self):
        return len(self.df) - self.seq_len + 1

    def __getitem__(self, idx):
        # Get the sequence of features
        seq = self.df[self.feature_columns].iloc[idx:idx+self.seq_len].values
        target = self.df[self.target_columns].iloc[idx+self.seq_len-1].values
        
        if self.features_scaler is not None:
            seq = self.features_scaler.transform(seq)
        if self.targets_scaler is not None:
            target = self.targets_scaler.transform(target.reshape(1, -1)).flatten()
        
        # Convert to tensors (make copies to avoid non-writable array warning) 
        seq = torch.FloatTensor(seq.copy())
        target = torch.FloatTensor(target.copy())
        
        return seq, target
    
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
    
def train_scaler(train_df, feature_columns, target_columns=None):
    features_scaler = StandardScaler()
    features_scaler.fit(train_df[feature_columns])
    
    joblib.dump(features_scaler, FEATURE_SCALER_PATH)
        
    
    if target_columns is not None:
        targets_scaler = StandardScaler()
        targets_scaler.fit(train_df[target_columns])
        joblib.dump(targets_scaler, TARGET_SCALER_PATH)
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