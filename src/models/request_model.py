import json
from mlflow import MlflowClient, pyfunc
import tempfile
import ast

from utils.logger import get_logger

def recover_params(params):
    recovered = {}

    recovered["model"] = params["model"]

    recovered["input_size"] = int(params["input_size"])
    recovered["output_size"] = int(params["output_size"])
    recovered["projection_size"] = int(params["projection_size"])
    recovered["hidden_size"] = int(params["hidden_size"])
    recovered["num_layers"] = int(params["num_layers"])
    recovered["seq_len"] = int(params["seq_len"])
    recovered["num_epochs"] = int(params["num_epochs"])

    recovered["lr"] = float(params["lr"])
    recovered["weight_decay"] = float(params["weight_decay"])

    recovered["feature_columns"] = ast.literal_eval(
        params["feature_columns"]
    )

    recovered["target_columns"] = ast.literal_eval(
        params["target_columns"]
    )

    recovered["data_period"] = params["data_period"]
    recovered["scaler"] = params["scaler"]

    return recovered

def request_model( model_name='stock-price-prediction-model', alias='Production'):
    logger = get_logger(__name__)

    model_uri = f"models:/{model_name}"
    
    client = MlflowClient()
    try:
        model_uri = f"models:/{model_name}@{alias.lower()}"
        model = pyfunc.load_model(
            model_uri=model_uri
        )
                                  
        model_version = client.get_model_version_by_alias(
            model_name,
            alias.lower()
        )
        latest_version = model_version.version
        logger.info(f"Successfully loaded model from {model_uri}")
    except Exception as e:
        logger.error(f"No {alias} model found at {model_uri}. Trying latest version.\n Error: {e}")
        model = None
    
    if model is None:
        try:
            latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
            model_uri = f"models:/{model_name}/{latest_version}"
            model = pyfunc.load_model(model_uri)
            logger.info(f"Successfully loaded latest model from {model_uri}/{latest_version}")
        except Exception as e:
            logger.error(f"Failed to load any version of the model from {model_uri}: {e}")
            return
        
    # get metrics from MLflow
    run = client.get_run(run_id)
    metrics = run.data.metrics
    
    # get model parameters from MLflow
    model_info = client.get_model_version(model_name, latest_version)
    run_id = model_info.run_id
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = client.download_artifacts(run_id, "config.json", dst_path=tmpdir)
            with open(config_path) as f:
                model_params = json.load(f)
    except Exception as e:
        logger.error(f"Failed to download model parameters from {run_id}: {e}")
        model_params = run.data.params
        if model_params is not None:
            model_params = recover_params(model_params)
            
    
    
            
    return model, latest_version, model_params , metrics