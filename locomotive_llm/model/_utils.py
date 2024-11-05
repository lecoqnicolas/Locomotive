from ..schema import LlmConfiguration
from ._tower_instruct_pipeline_langchain import TowerInstructPipelineLangChain
from ._madlad_pipeline import MadladPipeline
from ._fake_pipeline import FakePipeline
import logging


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
