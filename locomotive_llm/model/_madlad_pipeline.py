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
def match_code_language(src_name):
    if isinstance(src_name, list):
        src_name = src_name[0]
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
            elif device.startswith("cuda"):
                self._device = torch.cuda.current_device()  # Automatically select the current GPU
            else:
                raise ValueError(f"Unsupported device: {device}")
        else:

            self._device = int(device)
        self._max_tokens = max_tokens
        self._batch_size = batch_size
        logging.info(f"Loading model {model_id} on device: {device}")
        
        self._tokenizer = T5Tokenizer.from_pretrained(self._id)
        self._model = T5ForConditionalGeneration.from_pretrained(
            self._id,
            device_map= "cpu" if self._device == -1 else "auto",
            torch_dtype=torch.float32 if self._device == -1 else torch.float16
        )

        self._prompt_ignore = set(str(item) for item in (prompt_ignore or []))
        self._use_context = use_context
        self._separateur_context = separateur_context
        self._context_window = context_window

    def _is_text_valid(self, text: str):
        return isinstance(text, str) and text not in self._prompt_ignore

    def _prepare_inputs(self, texts, tgt_lang, context=None):
        code = match_code_language(tgt_lang)
        tag_prefix = f"<2{code}>"
        if context:
            texts = [f"{context} {text}" for text in texts]

        return [f"{tag_prefix} {text}" for text in texts]

    def _process_batch(self, batch):
        token_lengths = [len(self._tokenizer(text, return_tensors="pt").input_ids[0]) for text in batch]
        max_eln = max(token_lengths)
        input_ids = self._tokenizer(batch, return_tensors="pt", padding=True, truncation=False).input_ids.to(self._model.device)
        outputs = self._model.generate(input_ids=input_ids, max_length=max_eln)
        
        return [self._tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]


    def transform(self, texts: list[str], src_name: str, tgt_lang: str):
        valid_texts = [text for text in texts if isinstance(text, str) and self._is_text_valid(text)]
        translations = []
        context = None

        # Process each batch and include context if needed
        for i in range(0, len(valid_texts), self._batch_size):
            batch = valid_texts[i:i + self._batch_size]

            # If use_context is True, add the context
            if self._use_context:
                context_window_start = max(0, i - self._context_window)
                context = self._separateur_context.join(valid_texts[context_window_start:i])

            prepared_texts = self._prepare_inputs(batch, tgt_lang, context)
            batch_translations = self._process_batch(prepared_texts)
            translations.extend(batch_translations)

        results = []
        valid_idx = 0
        for text in texts:
            if self._is_text_valid(text):
                results.append(translations[valid_idx])
                valid_idx += 1
            else:
                results.append("")
        return results