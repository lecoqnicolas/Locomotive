from ..schema import LlmConfiguration
from ._tower_lmm_instruct_pipeline import TowerLlmInstructPipeline
from ._tower_lmm_pipeline import TowerLlmPipeline
#from ._tower_instruct_pipeline_langchain import TowerInstructPipelineLangChain
from ._tower_instruct_pipeline_langchain_batch import TowerInstructPipelineLangChain
from ._tower_llm_pipeline_langchain import TowerLlmPipelineLangChain


def get_pipeline(config: LlmConfiguration):
    # load model
    if config.use_towerinstruct and config.use_langchain:
        pipeline_class = TowerInstructPipelineLangChain
    elif config.use_langchain:
        pipeline_class = TowerLlmPipelineLangChain
    elif config.use_towerinstruct:
        pipeline_class = TowerLlmInstructPipeline
    else:
        pipeline_class = TowerLlmPipeline
    return pipeline_class
