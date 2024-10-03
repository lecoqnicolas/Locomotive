from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
import torch
from langchain.prompts import load_prompt


class TowerInstructPipelineLangChain:
    def __init__(self, model_id="Unbabel/TowerInstruct-13B-v0.1", device="cuda", max_tokens=512, prompt_file=None,
                 batch_size=32):
        self._id = model_id
        self._device = device
        self._max_len = max_tokens
        self._tokenizer = AutoTokenizer.from_pretrained(self._id)

        self._model = AutoModelForCausalLM.from_pretrained(self._id, torch_dtype=torch.float16).to(self._device)

        self._hf_pipeline = pipeline("text-generation", model=self._model, tokenizer=self._tokenizer,
                                     device=0 if self._device == "cuda" else -1)
        self.llm = HuggingFacePipeline(pipeline=self._hf_pipeline)
        self._prompt = load_prompt(prompt_file)

    def _create_prompt(self, text, src_lang, tgt_lang):
        return self._prompt.format(text=text, src_lang=src_lang, tgt_lang=tgt_lang)

    def _clean_output(self, output, prompt):
        cleaned_output = output.replace(prompt, "").strip()
        if "\n" in cleaned_output:
            cleaned_output = cleaned_output.split("\n")[0].strip()
        return cleaned_output

    def transform(self, texts, src_lang, tgt_lang):
        results = []
        for text in texts:
            prompt = self._create_prompt(text, src_lang, tgt_lang)
            outputs = self._hf_pipeline(prompt, max_new_tokens=100, do_sample=False)
            raw_response = outputs[0]["generated_text"]
            translation = self._clean_output(raw_response, prompt)
            results.append(translation)
        return results
