import torch
from langchain.prompts import load_prompt
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from ..postprocess import get_output_parsing_method


class TowerInstructPipelineLangChain:
    def __init__(self, model_id="Unbabel/TowerInstruct-Mistral-7B-v0.2", device="cuda", max_tokens=512,
                 prompt_file=None, prompt_ignore: list = None, batch_size: int = 50, output_parser: str = "json",
                 use_context: bool = False, separateur_context: str = ' ', context_window=0):
        self._id = model_id
        self._device = device
        self._max_tokens = max_tokens
        self._tokenizer = AutoTokenizer.from_pretrained(self._id)

        self._model = AutoModelForCausalLM.from_pretrained(self._id, torch_dtype=torch.float16).to(self._device)

        self._hf_pipeline = pipeline("text-generation", model=self._model, tokenizer=self._tokenizer,
                                     device=0 if self._device == "cuda" else -1, batch_size=batch_size)
        self.llm = HuggingFacePipeline(pipeline=self._hf_pipeline)
        self._prompt = load_prompt(prompt_file)
        self._prompt_ignore = set(prompt_ignore) if prompt_ignore is not None else {}
        self._output_parser = get_output_parsing_method(output_parser)
        self._use_context = use_context
        self._separateur_context = separateur_context
        self._context_window = context_window

    def _create_prompt_with_contexte(self, texts, src_lang, tgt_lang, prev_contexts=None):
        prompts = []
        for text, context in zip(texts, prev_contexts):
            prompt = self._prompt.format(text=text, src_lang=src_lang, tgt_lang=tgt_lang, context=context)
            prompts.append(prompt)
        return prompts

    def _create_prompt(self, texts, src_lang, tgt_lang):
        return [self._prompt.format(text=text, src_lang=src_lang, tgt_lang=tgt_lang) for text in texts]

    def _is_text_valid(self, text: str):
        return text not in self._prompt_ignore

    def _process_pipeline(self, prompts):
        if prompts:
            return self._hf_pipeline(prompts, max_new_tokens=self._max_tokens, do_sample=False)
        return []

    def _get_results(self, valid_mask, prompts, outputs):
        results = []
        output_idx = 0
        for is_valid in valid_mask:
            if is_valid:
                if output_idx < len(outputs) and "generated_text" in outputs[output_idx][0]:
                    cleaned_output = self._output_parser(outputs[output_idx][0]["generated_text"], prompts[output_idx])
                else:
                    cleaned_output = ""
                output_idx += 1
            else:
                cleaned_output = ""
            results.append(cleaned_output)
        return results

    def transform(self, texts: list[str], src_lang, tgt_lang, prev_contexts: list[str] = None):
        valid_mask = [self._is_text_valid(text) for text in texts]
        valid_texts = [text for text in texts if self._is_text_valid(text)]

        if self._use_context:
            # context-assisted translation
            if prev_contexts is None:
                # if no context was provided, create one based on previous sentences received
                prev_contexts = [texts[i-self._context_window:i] for i in range(len(texts))]
                # merge the sentences over context_window for each context
                prev_contexts = [self._separateur_context.join(context) if len(context) else ""
                                 for context in prev_contexts]
            valid_prev_contexts = [prev_context for idx, prev_context in enumerate(prev_contexts) if valid_mask[idx]]
            prompts = self._create_prompt_with_contexte(valid_texts, src_lang, tgt_lang, valid_prev_contexts)
        else:
            # basic translation
            prompts = self._create_prompt(valid_texts, src_lang, tgt_lang)

        outputs = self._process_pipeline(prompts)
        return self._get_results(valid_mask, prompts, outputs)