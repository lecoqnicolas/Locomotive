"""
Methods for mlflow interaction : retrieve/save/create runs and experiments from our training configuration and models.
"""
import logging

import mlflow
from mlflow import MlflowClient
from dataclasses import fields, is_dataclass
from ..schema import LlmConfiguration
from ._git import get_git_commit


# mlflow client
CLIENT = MlflowClient()


def fetch_mlflow_data(run_id: str, client=CLIENT) -> tuple:
    """
    Fetch data of a given mlflow run

    Args:
        run_id: id of the run
        client: optional, to fetch a distance client instead of the local one.

    Returns:
        parameters, metrics, tags and artifacts of the run run_id.
    """
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return data.params, data.metrics, tags, artifacts


def log_dataclass(config: LlmConfiguration, prefix: str = "") -> None:
    """
    Log any dataclass (our experiment configuration) as mlflow parameters. Supports nested dataclass.

    Args:
        config: configuration dataclass
        prefix: internal, to differentiate when nested dataclasses have the same parameters as its parent
    """
    # add the commit number
    config.git_commit = get_git_commit()
    for field in fields(config):
        field_name = field.name
        field_value = getattr(config, field_name)
        if is_dataclass(field_value):
            log_dataclass(field_value, prefix=f"{prefix}{field_name}_")
        logging.debug(f"{prefix}{field_name}: {field_value}")
        mlflow.log_param(f"{prefix}{field_name}", field_value)