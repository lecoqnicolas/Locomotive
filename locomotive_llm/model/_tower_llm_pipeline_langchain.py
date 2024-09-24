from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# Initialisation de TowerLLM avec LangChain
class TowerLlmPipelineLangChain:
    
    def __init__(self, model_id="Unbabel/TowerInstruct-13B-v0.1", device="cpu", max_len=512):

        self._id = model_id
        self._device = device
        self._max_len = max_len
        self._tokenizer = AutoTokenizer.from_pretrained(self._id)
        self._model = AutoModelForCausalLM.from_pretrained(self._id).to(self._device)
        self._hf_pipeline = pipeline("text-generation", model=self._model, tokenizer=self._tokenizer, device=0 if self._device == "cuda" else -1)

        self.llm = HuggingFacePipeline(pipeline=self._hf_pipeline)

    def _create_prompt(self, text, src_lang, tgt_lang):
        return f"{src_lang}: {text}\n{tgt_lang}:"
    def _clean_output(self, output, prompt):
      
        cleaned_output = output.replace(prompt, "").strip()
        if "\n" in cleaned_output:
            cleaned_output = cleaned_output.split("\n")[0].strip()

        return cleaned_output

    def transform(self, texts, src_lang, tgt_lang):

        results = []
        
        for text in texts:
          
            prompt = self._create_prompt(text, src_lang, tgt_lang)
     
            response = self.llm.invoke(prompt, max_new_tokens=100)
            print("Raw response:", response)
            translation = self._clean_output(response, prompt)
            results.append(translation)
        
        return results
