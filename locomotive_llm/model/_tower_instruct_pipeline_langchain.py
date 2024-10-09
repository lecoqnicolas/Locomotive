import torch
from langchain.prompts import load_prompt
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
import re

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
    def _clean_translation_output(self, output):
        # Find translations in the output
        matches = re.findall(r'"translated_text":\s*"([^"]*)"', output)

        # Extract translations
        translations = [match for match in matches]

        # Create a valid JSON response
        if translations:
            # Take the last translation and remove any quotes
            cleaned_translation = translations[1].replace('"', '').strip()  # Remove quotes
            return cleaned_translation  # Return cleaned translation

        return None
    def _clean_output_json(self, output, prompt):
        json_output = ""
        
        # Find the first and last braces to extract potential JSON
        output = output.replace("'", '"')
        matches = re.findall(r'\{.*?\}', output, re.DOTALL)
        if matches:
            json_output = matches[-1].strip()  # Take the last matched JSON structure

        if json_output:
            try:
                parsed_output = json.loads(json_output)

                return parsed_output.get("translated_text", "").strip().replace(prompt, "")
            except json.JSONDecodeError as e:
                print(f"Invalid JSON output: {e}, output was: {json_output}")
                return ""
        
        print(f"No valid JSON found in output: {output}")
        return ""


    def _is_text_valid(self, text):
        return text not in self._prompt_ignore

    def transform(self, texts, src_lang, tgt_lang):
        valid_mask = [self._is_text_valid(text) for text in texts]
        valid_texts = [text for text in texts if self._is_text_valid(text)]
        # avoid asking for inference if no valid texts
        if valid_texts:

            prompts = self._create_prompt(valid_texts, src_lang, tgt_lang)
            outputs = self._hf_pipeline(prompts, max_new_tokens=self._max_tokens, do_sample=False)
        else:
            outputs = []
            prompts = []

        results = []
        output_idx = 0
        for is_valid in valid_mask:
            if is_valid:
                if output_idx < len(outputs) and "generated_text" in outputs[output_idx][0]:
                    cleaned_output = self._clean_output(outputs[output_idx][0]["generated_text"], prompts[output_idx])
                    #cleaned_output = self._clean_translation_output(outputs[output_idx][0]["generated_text"])
                else:
                    cleaned_output = ""
                output_idx += 1
            else:
                cleaned_output = "" 
            results.append(cleaned_output)
        return results
