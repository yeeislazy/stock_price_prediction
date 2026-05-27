FROM apache/airflow:3.0.6-python3.11

USER root

RUN apt-get update && apt-get install -y \
    curl \
    build-essential

USER airflow

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/home/airflow/.local/bin:${PATH}"
ENV PYTHONPATH="/opt/airflow/project/src"
ENV PATH="/opt/airflow/project/.venv/bin:${PATH}"

WORKDIR /opt/airflow/project

COPY --chown=airflow:root pyproject.toml uv.lock ./
COPY --chown=airflow:root src ./src

RUN uv sync

COPY --chown=airflow:root . .