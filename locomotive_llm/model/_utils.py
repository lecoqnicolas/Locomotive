from ..schema import LlmConfiguration
from ._tower_instruct_pipeline_langchain import TowerInstructPipelineLangChain


def get_pipeline(config: LlmConfiguration):
    # load model
    if config.use_towerinstruct and config.use_langchain:
        pipeline_class = TowerInstructPipelineLangChain
    else:
        raise NotImplementedError("Not pipeline found for config")
    return pipeline_class
