import torch
from langchain.prompts import load_prompt
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from ..postprocess import get_output_parsing_method


class TowerInstructPipelineLangChain:
    def __init__(self, model_id="Unbabel/TowerInstruct-Mistral-7B-v0.2", device="cuda", max_tokens=512, prompt_file=None,
                 prompt_ignore: list = None, batch_size: int = 50, output_parser: str = "keep_first_line", use_context: bool = False):
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
        self.use_context = use_context

    def _create_prompt(self, texts, prev_contexts, src_lang, tgt_lang):
        prompts = []
        for text, context in zip(texts, prev_contexts):

            full_text = " ".join(context + [text])
            prompt = self._prompt.format(text=full_text, src_lang=src_lang, tgt_lang=tgt_lang)
            prompts.append(prompt)
        return prompts

    def _is_text_valid(self, text: str):
        return text not in self._prompt_ignore

    def transform(self, texts: list[str], src_lang, tgt_lang, prev_contexts: list[list[str]] = None):
        if self.use_context:
            if prev_contexts is None:
                prev_contexts = [[] for _ in texts]
            
            valid_mask = [self._is_text_valid(text) for text in texts]
            valid_texts = [text for text in texts if self._is_text_valid(text)]
            valid_prev_contexts = [prev_context for idx, prev_context in enumerate(prev_contexts) if valid_mask[idx]]
            if valid_texts:
                prompts = self._create_prompt(valid_texts, valid_prev_contexts, src_lang, tgt_lang)
                outputs = self._hf_pipeline(prompts, max_new_tokens=self._max_tokens, do_sample=False)
            else:
                outputs = []
                prompts = []
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
        else:
            valid_mask = [self._is_text_valid(text) for text in texts]
            valid_texts = [text for text in texts if self._is_text_valid(text)]
            if valid_texts:
                prompts = self._create_prompt(valid_texts, [[] for _ in valid_texts], src_lang, tgt_lang)  # Pas de contexte
                outputs = self._hf_pipeline(prompts, max_new_tokens=self._max_tokens, do_sample=False)
            else:
                outputs = []
                prompts = []

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
