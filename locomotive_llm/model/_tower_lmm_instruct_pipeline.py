import torch
from transformers import pipeline


class TowerLlmInstructPipeline:

    def __init__(self, model_id="Unbabel/TowerBase-7B-v0.1", device="cpu", max_len=512, batch_size=1000,  prompt_file=None):
        self._id = model_id
        self._device = device
        self._model = pipeline("text-generation", model="Unbabel/TowerInstruct-Mistral-7B-v0.2", torch_dtype=torch.bfloat16, device_map="auto")
        self._max_len = max_len
        self._batch_size = batch_size
        self._prompt = load_prompt(prompt_file)

    def _towerinstruct_prompt(self, text, src, tgt):
        return self._prompt.format(text=text, src_lang=src, tgt_lang=tgt)

    def transform(self, texts: list[str], src: str = None, tgt: str = None):
        results = []
        if src is not None and tgt is not None:
            prompts = [self._towerinstruct_prompt(text, src, tgt) for text in texts]
            outputs = self._model(prompts, max_new_tokens=256, do_sample=False)
            results = [output[0]["generated_text"] for output in outputs]
        return results
