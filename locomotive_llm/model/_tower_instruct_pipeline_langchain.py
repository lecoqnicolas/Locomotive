import torch
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from ..postprocess import BasicPostProcessor
from ..preprocess import ContextPreprocessor, BasicPreprocessor
from ._utils import get_device


class TowerInstructPipelineLangChain:
    def __init__(self, model_id="Unbabel/TowerInstruct-Mistral-7B-v0.2", device="cuda", max_tokens=512,
                 prompt_file=None, prompt_ignore: list = None, batch_size: int = 50, output_parser: str = "json",
                 use_context: bool = False, separateur_context: str = ' ', context_window=0):
        self._id = model_id
        self._device = get_device(device)
        self._max_tokens = max_tokens

        # init langchain hugginface pipeline
        self._tokenizer = AutoTokenizer.from_pretrained(self._id)
        torch_dtype = torch.float32 if self._device == -1 else torch.float16
        self._model = AutoModelForCausalLM.from_pretrained(self._id, torch_dtype=torch_dtype)
        self._hf_pipeline = pipeline("text-generation", model=self._model, tokenizer=self._tokenizer,
                                         device=self._device, batch_size=batch_size)
        self.llm = HuggingFacePipeline(pipeline=self._hf_pipeline)

        # init custom preprocessing
        if use_context:
            self._preprocessor = BasicPreprocessor(prompt_file, prompt_ignore)
        else:
            self._preprocessor = ContextPreprocessor(prompt_file,
                                                     prompt_ignore,
                                                     context_window=context_window,
                                                     context_sep=separateur_context)

        # init postprocessing
        self._output_parser = BasicPostProcessor(output_parser)

    def transform(self, texts: list[str], src_lang: list | str, tgt_lang: list | str):
        prompts, valid_masks = self._preprocessor.transform(texts, src_lang, tgt_lang)
        outputs = self._hf_pipeline(prompts, max_new_tokens=self._max_tokens, do_sample=False) if prompts else []
        return self._output_parser.transform(valid_masks, prompts, outputs)
