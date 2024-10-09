from ..schema import LlmConfiguration
from ._tower_instruct_pipeline_langchain import TowerInstructPipelineLangChain

def get_pipeline(config: LlmConfiguration):
    # load model
    if config.use_towerinstruct and config.use_langchain:
        pipeline_class = TowerInstructPipelineLangChain
    return pipeline_class
