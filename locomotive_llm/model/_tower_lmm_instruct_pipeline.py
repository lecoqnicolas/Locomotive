import torch
from transformers import pipeline


class TowerLlmInstructPipeline:

    def __init__(self, model_id="Unbabel/TowerBase-7B-v0.1", device="cpu", max_len=512, batch_size=1000):
        self._id = model_id
        self._device = device
        self._model = pipeline("text-generation", model="Unbabel/TowerInstruct-Mistral-7B-v0.2", torch_dtype=torch.bfloat16, device_map="auto")
        self._max_len = max_len
        self._batch_size = batch_size

    def _towerinstruct_prompt(self, str, src, tgt):
        messages = [
            {"role": "user",
             "content": f"Translate the following text from {src} into {tgt}. Only answer with the translation.  "
                        f"Keep the translation as close to the original in tone and style as you can.\n{src}: {str}\n{tgt}:"},
        ]
        return self._model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def remove_prompt(self, answer, input):
        return answer.split(input)[1:]

    def transform(self, texts:list[str], src: str=None, tgt: str=None):
        """
        Translate input text using the TowerLLM model.

        :param texts: list of texts to translate.
        :return: The translated texts.
        """
        res = []
        prompts = [self._towerinstruct_prompt(text, src, tgt) for text in texts]
        for prompt in prompts:
            outputs = self._model(prompt, max_new_tokens=256, do_sample=False)
            res.append(outputs[0]["generated_text"])
        return res
