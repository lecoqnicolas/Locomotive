from langchain.prompts import load_prompt
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

torch.cuda.empty_cache()
class TowerLlmPipelineLangChain:
    
    def __init__(self, model_id="Unbabel/TowerInstruct-13B-v0.1", device="cpu", max_len=512,  prompt_file=None, batch_size=16):
        self._id = model_id
        self._device = device
        self._max_len = max_len
        self._tokenizer = AutoTokenizer.from_pretrained(self._id,  use_auth_token=True)
        self._model = AutoModelForCausalLM.from_pretrained(self._id).to(self._device)
        self._hf_pipeline = pipeline("text-generation", model=self._model, torch_dtype=torch.bfloat16,tokenizer=self._tokenizer,
                                      device=0 if self._device == "cuda" else -1)
        self.llm = HuggingFacePipeline(pipeline=self._hf_pipeline)
        self._prompt = load_prompt(prompt_file) if prompt_file else None

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
            outputs = self._hf_pipeline(prompt, max_new_tokens=10, do_sample=False)
            raw_response = outputs[0]["generated_text"]
            translation = self._clean_output(raw_response, prompt)
            results.append(translation)
        return results
