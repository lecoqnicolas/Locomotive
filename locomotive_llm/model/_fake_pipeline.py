import logging

import torch
from langchain.prompts import load_prompt
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from ..postprocess import get_output_parsing_method


class FakePipeline:
    def __init__(self, model_id="Unbabel/TowerInstruct-Mistral-7B-v0.2", device="cuda", max_tokens=512,
                 prompt_file=None, prompt_ignore: list = None, batch_size: int = 50, output_parser: str = "json",
                 use_context: bool = False, separateur_context: str = ' ', context_window=0):
        logging.info("RUNNING FAKE LLM PIPELINE FOR TESTING PURPOSE")

    def transform(self, texts: list[str], src_lang, tgt_lang, prev_contexts: list[list[str]] = None):
        logging.debug(f"source langue {src_lang}")
        logging.debug(f"target langue {tgt_lang}")
        logging.debug(f"Input texts : {texts}")
        logging.debug(f"Input context : {prev_contexts}")
        return ["FAKE"+text[4:] for text in texts]
