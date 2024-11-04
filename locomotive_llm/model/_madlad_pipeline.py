import logging
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import load_prompt
from ..postprocess import get_output_parsing_method
language_names = {
    "ar": "Arabic",
    "fr": "French",
    "en": "English",
    "zh": "Chinese",
    "pt_br": "Brazilian Portuguese",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ru": "Russian"
}
def match_code_language(src_name: str):
    for code, name in language_names.items():
        if name.lower() == src_name.lower():
            return code
    return "Language not found"

class MadladPipeline:
    def __init__(self, model_id="google/madlad400-10b-mt", device="cuda", max_tokens=50, prompt_file=None,
                 prompt_ignore: list = None, batch_size: int = 50, output_parser: str = "json",
                 use_context: bool = False, separateur_context: str = ' ', context_window=0):
        self._id = model_id
        if isinstance(device, str):
            if device.lower() == "cpu":
                self._device = -1 
            else:
                self._device = int(device)
        else:
            self._device = device 
        self._max_tokens = max_tokens

        logging.info(f"Loading model {model_id} on device: {device}")
        
        self._tokenizer = T5Tokenizer.from_pretrained(self._id)
        self._model = T5ForConditionalGeneration.from_pretrained(
            self._id,
            device_map="auto",
            torch_dtype=torch.float16
        )

        self._prompt_ignore = set(prompt_ignore) if prompt_ignore else set()

    def _is_text_valid(self, text: str):
        return text not in self._prompt_ignore

    def _prepare_inputs(self, texts, tgt_lang):
        code = match_code_language(tgt_lang)
        tag_prefix = f"<2{code}>"
        return [f"{tag_prefix} {text}" for text in texts]

    def transform(self, texts: list[str], src_lang: str, tgt_lang: str):
        valid_texts = [text for text in texts if self._is_text_valid(text)]
        prepared_texts = self._prepare_inputs(valid_texts, tgt_lang)
        input_ids = self._tokenizer(prepared_texts, return_tensors="pt", padding = True, truncation=True).input_ids.to(self._model.device)
        outputs = self._model.generate(input_ids=input_ids, max_new_tokens=self._max_tokens)
        translations = [self._tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]
        return translations