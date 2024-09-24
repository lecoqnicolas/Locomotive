from ..schema import LlmConfiguration
from pathlib import Path
import yaml


def load_config(config_path: str | Path, reverse=False) -> LlmConfiguration:
    try:
        with open(str(config_path), "r") as f:
            config_dict = yaml.safe_load(f)
        config = LlmConfiguration(**config_dict)
        if reverse:
            config.src_code, config.tgt_code = config.tgt_code, config.src_code
            config.src_name, config.tgt_name = config.tgt_name, config.src_name
        return config
    except Exception as e:
        raise Exception(f"Cannot open config file") from e
