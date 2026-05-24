import mlflow
import mlflow.pyfunc
import torch
import pandas as pd

from models.lstm import LSTMModel

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

