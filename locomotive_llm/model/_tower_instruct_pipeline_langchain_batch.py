import torch
from langchain.prompts import load_prompt
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class TowerInstructPipelineLangChain:
    def __init__(self, model_id="Unbabel/TowerInstruct-Mistral-7B-v0.2", device="cuda", max_tokens=512, prompt_file=None,
                 prompt_ignore=None, batch_size=50):
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

    def _create_prompt(self, texts, src_lang, tgt_lang):
        return [self._prompt.format(text=text, src_lang=src_lang, tgt_lang=tgt_lang) for text in texts]

    def _clean_output(self, output, prompt):
        cleaned_output = output.replace(prompt, "").strip()
        if "\n" in cleaned_output:
            cleaned_output = cleaned_output.split("\n")[0].strip()
        return cleaned_output

    def _is_text_valid(self, text):
        return text not in self._prompt_ignore

    def transform(self, texts, src_lang, tgt_lang):
        valid_mask = [self._is_text_valid(text) for text in texts]
        valid_texts = [text for text in texts if self._is_text_valid(text)]
        # avoid asking for inference if no valid texts
        if len(valid_texts) > 0:
            prompts = self._create_prompt(valid_texts, src_lang, tgt_lang)
            outputs = self._hf_pipeline(prompts, max_new_tokens=self._max_tokens, do_sample=False)
        else:
            outputs = []
            prompts = []

        results = []
        output_idx = 0
        for is_valid in valid_mask:
            if is_valid:
                if "generated_text" in outputs[output_idx][0]:
                    cleaned_output = self._clean_output(outputs[output_idx][0]["generated_text"], prompts[output_idx])
                else:
                    cleaned_output = ""
                output_idx += 1
            else:
                cleaned_output = ""
            results.append(cleaned_output)
        return results
