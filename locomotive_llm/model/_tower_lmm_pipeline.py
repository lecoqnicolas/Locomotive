from transformers import AutoTokenizer, AutoModelForCausalLM


class TowerLlmPipeline:

    def __init__(self, model_id="Unbabel/TowerBase-7B-v0.1", device="cpu", max_len=512, batch_size=1000):
        self._id = model_id
        self._device = device
        self._model = AutoModelForCausalLM.from_pretrained(self._id).to(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self._id)
        self._model.eval()
        self._max_len = max_len
        self._batch_size= batch_size

    def _towerbase_prompt(self, str, src, tgt):
        return f"{src} : {str}\n{tgt} :"

    def transform(self, texts, src=None, tgt=None):
        """
        Translate input text using the TowerLLM model.

        :param texts: list of texts to translate.
        :return: The translated texts.
        """
        if src is not None and tgt is not None:
            texts = [self._towerbase_prompt(text, src, tgt) for text in texts]
        inputs = self._tokenizer(texts, return_tensors="pt").input_ids
        inputs = inputs.to(self._device)
        outputs = self._model.generate(inputs, max_length=self._max_len)
        return self._tokenizer.decode(outputs, skip_special_tokens=True)
