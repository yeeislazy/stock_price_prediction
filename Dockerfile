FROM apache/airflow:3.0.6-python3.11

USER root

RUN apt-get update && apt-get install -y \
    curl \
    build-essential

USER airflow

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/home/airflow/.local/bin:${PATH}"
ENV PYTHONPATH="/opt/airflow/project/src"

WORKDIR /opt/airflow/project

COPY pyproject.toml uv.lock ./
COPY src ./src

RUN uv sync --torch-backend auto

COPY . .