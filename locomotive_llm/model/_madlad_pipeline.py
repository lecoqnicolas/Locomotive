import logging

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from ._utils import get_device
from ..preprocess import BasicPreprocessor
from ..postprocess import BasicPostProcessor, LlmResponseParser


class MadladPipeline:
    def __init__(self, model_id="google/madlad400-10b-mt", device="cuda", max_tokens=50, prompt_file=None,
                 prompt_ignore: list = None, batch_size: int = 50, output_parser: str = "json",
                 use_context: bool = False, separateur_context: str = ' ', context_window=0):
        self._id = model_id
        self._device = get_device(device)
        self._max_tokens = max_tokens
        self._batch_size = batch_size

        # init model and tokenizers
        logging.info(f"Loading model {model_id} on device: {device}")
        self._tokenizer = T5Tokenizer.from_pretrained(self._id)
        self._model = T5ForConditionalGeneration.from_pretrained(
            self._id,
            device_map="cpu" if self._device == -1 else "auto",
            torch_dtype=torch.float32 if self._device == -1 else torch.float16
        )

        self._preprocessor = BasicPreprocessor(prompt_file, prompt_ignore, use_lang_code=True)
        # madlad output is already parsed : postprocessing parser should be identity
        self._output_parser = BasicPostProcessor(LlmResponseParser.identity)

    def transform(self, texts: list[str], src_lang: str | list[str], tgt_lang: str | list[str]):
        translations = []
        valid_texts, valid_masks = self._preprocessor.transform(texts, src_lang, tgt_lang)

        # Batch processing
        for i in range(0, len(valid_texts), self._batch_size):
            batch = valid_texts[i:i + self._batch_size]

            input_ids = self._tokenizer(batch, return_tensors="pt", padding=True, truncation=False).input_ids.to(
                self._model.device)
            outputs = self._model.generate(input_ids=input_ids, max_length=self._max_tokens)
            translations = [self._tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]
            translations.extend(translations)

        return self._output_parser.transform(valid_masks, valid_texts, translations)
