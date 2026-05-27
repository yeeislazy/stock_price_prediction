#   Stock Price Forecasting MLOps Platform

End-to-end MLOps platform for multi-horizon stock return forecasting using PyTorch LSTM, Apache Airflow, MLflow, and DAGsHub.

The system automates:

* financial data ingestion
* feature engineering
* model training
* challenger retraining
* production monitoring
* model lifecycle management

---

# Features

## Multi-Horizon Forecasting

Predicts future stock returns for:

* `return_2`
* `return_5`
* `return_14`

using weighted multi-target learning.

---

## PyTorch LSTM Forecasting Model

* Sequence-based time-series forecasting
* Multi-target regression
* Weighted loss optimization
* Early stopping
* Automatic best-model checkpointing

---

## Automated Retraining Pipeline

Apache Airflow DAG automatically:

1. Downloads latest stock data
2. Processes features
3. Retrains challenger model
4. Evaluates against production model
5. Registers challenger if performance improves

---

## Production Monitoring Pipeline

Daily monitoring workflow:

* loads production/challenger models
* evaluates latest data
* tracks prediction metrics
* retrains on full dataset
* logs monitoring results

---

## MLflow + DAGsHub Integration

Remote MLflow Tracking Server hosted on [DAGsHub](https://dagshub.com?utm_source=chatgpt.com)

Supports:

* experiment tracking
* artifact logging
* model registry
* model versioning
* alias-based lifecycle management

Aliases:

* `production`
* `challenger`

---

## Dockerized Infrastructure

* Dockerized Airflow environment
* UV dependency management
* Reproducible pipelines
* Volume-mounted artifacts and datasets

---

# System Architecture

```text
Yahoo Finance Data
        │
        ▼
Data Ingestion Pipeline
        │
        ▼
Feature Engineering
(RSI, ATR, MA, Returns)
        │
        ▼
PyTorch LSTM Training
        │
        ▼
MLflow Tracking Server
(Hosted on DAGsHub)
        │
        ▼
Model Registry
(production / challenger)
        │
        ├────────────────────┐
        ▼                    ▼
Retraining DAG         Monitoring DAG
(Every 3 Days)         (Daily)
        │                    │
        ▼                    ▼
Compare Against       Evaluate Production
Production Model      Performance
        │                    │
        ▼                    ▼
Register Challenger   Retrain on Full Data
```

---

# Tech Stack

| Category              | Technology     |
| --------------------- | -------------- |
| ML Framework          | PyTorch        |
| Experiment Tracking   | MLflow         |
| Registry Hosting      | DAGsHub        |
| Orchestration         | Apache Airflow |
| Data Processing       | Pandas         |
| Model Serving         | MLflow PyFunc  |
| Containerization      | Docker         |
| Dependency Management | UV             |
| Data Source           | Yahoo Finance  |

---

# Project Structure

```text
src/
├── configuration/
├── data/
│   ├── ingestion/
│   └── processing/
├── training/
├── monitoring/
├── orchestration/
│   └── airflow/
│       └── dags/
├── models/
├── utils/
└── app/
```

---

# Airflow Workflows

## DAG 1 — Retraining Pipeline (Every 3 Days)

```text
download-data
    ↓
process-data
    ↓
retrain-pipeline
```

Workflow:

* loads current production model parameters
* trains challenger model on latest data
* evaluates against production metrics
* registers better model as challenger

---

## DAG 2 — Production Monitoring (Daily)

```text
monitor-production
```

Workflow:

* evaluates deployed model performance
* retrains using full available dataset
* logs monitoring metrics to MLflow

---

# Feature Engineering

Implemented technical indicators:

* Moving Average (MA)
* Relative Strength Index (RSI)
* Average True Range (ATR)

Additional features:

* date features
* return generation
* normalization/scaling
* sequence window generation

---

# Model Lifecycle

```text
production  → current deployed model
challenger  → newly retrained candidate
```

Workflow:

1. retrain challenger
2. compare metrics
3. promote if better

---

# Running with Docker

## Build

```bash
docker compose build
```

## Start Airflow

```bash
docker compose up
```

Airflow UI:

```text
http://localhost:8080
```

---

# Example Training Commands

```bash
uv run download-update-data
uv run process-data
uv run train-candidates
uv run train-production
```

---

# Example Metrics Logged

* train_loss
* val_loss
* MAE
* MSE
* direction_accuracy
* unscaled prediction error

---

# Future Improvements

* FastAPI inference service
* online inference API
* GPU training workers
* distributed retraining
* hyperparameter optimization
* Kubernetes deployment
* automated champion promotion

---

# License

MIT License
