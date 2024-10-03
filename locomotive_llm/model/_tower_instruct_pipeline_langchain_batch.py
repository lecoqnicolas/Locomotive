from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
import torch
from langchain.prompts import load_prompt

class TowerInstructPipelineLangChain:
    def __init__(self, model_id="Unbabel/TowerInstruct-Mistral-7B-v0.2", device="cuda", max_len=512, prompt_file=None,
                 prompt_ignore=None):
        self._id = model_id
        self._device = device
        self._max_len = max_len
        self._tokenizer = AutoTokenizer.from_pretrained(self._id)
       
        self._model = AutoModelForCausalLM.from_pretrained(self._id, torch_dtype=torch.float16).to(self._device)
        
        self._hf_pipeline = pipeline("text-generation", model=self._model, tokenizer=self._tokenizer,
                                     device=0 if self._device == "cuda" else -1)
        self.llm = HuggingFacePipeline(pipeline=self._hf_pipeline)
        self._prompt = load_prompt(prompt_file)
        self._prompt_ignore = prompt_ignore if prompt_ignore else {}

    def _create_prompt(self, texts, src_lang, tgt_lang):
        return [self._prompt.format(text=text, src_lang=src_lang, tgt_lang=tgt_lang) for text in texts]

    def _clean_output(self, output, prompt):
        cleaned_output = output.replace(prompt, "").strip()
        if "\n" in cleaned_output:
           cleaned_output = cleaned_output.split("\n")[0].strip()
        return cleaned_output

    def _filter_texts(self, text):
        return text in self._prompt_ignore
        
    def transform(self, texts, src_lang, tgt_lang):
        prompts = self._create_prompt(texts, src_lang, tgt_lang)
        outputs = self._hf_pipeline(prompts, max_new_tokens=100, do_sample=False)
        results = []
        for i, output in enumerate(outputs):
            try:
                if "generated_text" in output:
                    cleaned_output = self._clean_output(output["generated_text"], prompts[i])
                else:
                    cleaned_output = "" 
            except Exception as e:
                print(f"An error occurred while processing text {i}: {e}")
                cleaned_output = ""
            results.append(cleaned_output)
        
        return results
