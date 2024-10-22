from ..schema import LlmConfiguration
from ._tower_instruct_pipeline_langchain import TowerInstructPipelineLangChain
from ._fake_pipeline import FakePipeline
import logging


def get_pipeline(config: LlmConfiguration):
    # load model
    if config.pipeline == "fake":
        logging.info("Using fake pipeline")
        pipeline_class = FakePipeline
    else:
        pipeline_class = TowerInstructPipelineLangChain
    return pipeline_class
