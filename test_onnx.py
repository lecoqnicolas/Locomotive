from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
import torch
import time
from torch.cuda.amp import autocast
import onnxruntime as ort
from locomotive_llm.preprocess import BasicPreprocessor
from locomotive_llm.postprocess import BasicPostProcessor, LlmResponseParser
import torch
print(torch.version.cuda)
import onnxruntime as ort
print("Available providers:", ort.get_available_providers())
def translate_texts(texts, src_lang, tgt_lang, model, tokenizer, device, batch_size=10):
    prompt_template = "./config/prompts/prompt_v1.yml"
    prepro = BasicPreprocessor(prompt_file=prompt_template, ignore_prompts=[" ", "\n", " \n", "  "])
    prompts, valid_mask = prepro.transform(texts, src_lang, tgt_lang)
    print(prompts)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    print(inputs)
    gen_tokens = model.generate(**inputs, max_new_tokens=50)
    translations = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    print(translations)
    postpro = BasicPostProcessor(output_parsing_method=LlmResponseParser.keep_first_line, output_field=None)
    return postpro.transform(valid_mask=valid_mask, input_prompts=prompts, outputs=translations)

providers = ["CUDAExecutionProvider"]
model_path = "tower_onnx/"
model = ORTModelForCausalLM.from_pretrained(model_path, use_cache=False, providers=providers, use_io_binding=False)
#model = ort.InferenceSession(model_path, providers=providers)
tokenizer = AutoTokenizer.from_pretrained("tower_onnx/")
print(torch.cuda.is_available() )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
total_start_time = time.time()
# Example
src_lang = "English"
tgt_lang = "French"
text = ["My name is Arthur and I live in Paris"]

translations = translate_texts(text, src_lang, tgt_lang, model, tokenizer, device)
print(time.time() - total_start_time)
print(translations)
