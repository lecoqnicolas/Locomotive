from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
import torch
import time

from locomotive_llm.preprocess import BasicPreprocessor
from locomotive_llm.postprocess import BasicPostProcessor, LlmResponseParser


def translate_texts(texts, src_lang, tgt_lang, model, tokenizer, device):
    #prompt_template = "Only answer with the traduction. Never explain or detail your answers. Translate the following text from {src_lang} to {tgt_lang} :\n{src_lang}: {text}\n{tgt_lang}:"
    prompt_template = "./config/prompts/prompt_v1.yml"
    prepro = BasicPreprocessor(prompt_file=prompt_template, ignore_prompts=[" ", "\n", " \n", "  "])
    prompts, valid_mask = prepro.transform(texts, src_lang, tgt_lang)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    gen_tokens = model.generate(**inputs, max_new_tokens=50)
    translations = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    postpro = BasicPostProcessor(output_parsing_method=LlmResponseParser.keep_first_line)
    return postpro.transform(valid_mask=valid_mask, input_prompts=prompts, outputs=translations)


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
