from dataclasses import dataclass
from typing import List


@dataclass
class LlmConfiguration:
    experiment_name: str = "default_experiment"
    run_name: str = "default_run"
    git_commit: str = None
    mlflow_repository: str = "mlflow_repository"
    prompt: str = "config/prompts/example_prompt.yml"
    pipeline: str = "tower_instruct"
    model_type: str = "tower"
    use_context: bool = False
    separateur_context: str = ' '
    context_window: int = 2
    preserve_formatting: bool= True
    langchain_parsing: bool= False
    version: int = 1
    llm_model: str = "Unbabel/TowerInstruct-Mistral-7B-v0.2"
    batch_size: int = 1024
    max_token: int = 512
    response_parsing_method: str = "keep_first_line"
    device: str = "cuda"
    src_code: str = "en_Latn"
    src_name: str = "Enlish"
    tgt_code: str = "fra_Latn"
    tgt_name: str = "French"
    ignore_prompt: List[str] = None
