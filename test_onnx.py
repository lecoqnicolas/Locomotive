from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
import torch
import time
def _clean_output(output, prompt):
    cleaned_output = output.replace(prompt[0], "").strip()
    if "\n" in cleaned_output:
        cleaned_output = cleaned_output.split("\n")[0].strip()

    return cleaned_output
def _create_prompt(texts, prompt, src_langs, tgt_langs):
    return [prompt.format(text=text, src_lang=src, tgt_lang=tgt)
            for text, src, tgt in zip(texts, src_langs, tgt_langs)]

def translate_texts(texts, src_lang, tgt_lang, model, tokenizer, device):
    prompt_template = "Only answer with the traduction. Never explain or detail your answers. Translate the following text from {src_lang} to {tgt_lang} :\n{src_lang}: {text}\n{tgt_lang}:"
    prompts = _create_prompt(texts, prompt_template, [src_lang] * len(texts), [tgt_lang] * len(texts))
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    gen_tokens = model.generate(**inputs, max_new_tokens=50)
    translations = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    outputs = [_clean_output(trans, prompts) for trans in translations]
    return outputs

tokenizer = AutoTokenizer.from_pretrained("./tower_onnx/")
model = ORTModelForCausalLM.from_pretrained("./tower_onnx/", use_cache=False, use_io_binding=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
total_start_time = time.time()
# Example
src_lang = "English"
tgt_lang = "Frdeench"
text = ["My name is Arthur and I live in Paris"]

translations = translate_texts(text, src_lang, tgt_lang, model, tokenizer, device)
print(time.time() - total_start_time)
print(translations)
