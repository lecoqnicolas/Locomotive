import logging

import torch

from ._fake_pipeline import FakePipeline
from ._madlad_pipeline import MadladPipeline
from ._tower_instruct_pipeline_langchain import TowerInstructPipelineLangChain
from ..schema import LlmConfiguration


def get_device(device):
    if isinstance(device, str):
        if device.lower() == "cpu":
            device = -1
        elif device.startswith("cuda"):
            device = torch.cuda.current_device()
        else:
            raise ValueError(f"Unsupported device: {device}")
    else:
        device = int(device)
    return device


def get_pipeline(config: LlmConfiguration):
    # load model
    if config.pipeline == "fake":
        logging.info("Using fake pipeline")
        pipeline_class = FakePipeline
    elif config.pipeline == "madlad400":
        logging.info("Using madlad pipeline")
        pipeline_class = MadladPipeline
    else:
        pipeline_class = TowerInstructPipelineLangChain
    return pipeline_class
