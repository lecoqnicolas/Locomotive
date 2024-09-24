from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline



class TowerLlmPipeline:

    def __init__(self, model_id="Unbabel/TowerBase-7B-v0.1", device="cpu", max_len=512, batch_size=1000):
        self._id = model_id
        self._device = device
        self._model = AutoModelForCausalLM.from_pretrained(self._id).to(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self._id)
        self._model.eval()
        self._max_len = max_len
        self._batch_size = batch_size

    def _towerbase_prompt(self, str, src, tgt):
        return f"{src} : {str}\n{tgt} :"

    def remove_prompt(self, answer, input):
        return answer.split(input)[1:]
    

    def transform(self, texts, src=None, tgt=None):
        """
        Translate input text using the TowerLLM model.

        :param texts: list of texts to translate.
        :return: The translated texts.
        """
        res = []
        if src is not None and tgt is not None:
            texts = [self._towerbase_prompt(text, src, tgt) for text in texts]
        for text in texts:
            inputs = self._tokenizer(text, return_tensors="pt")
            outputs = self._model.generate(inputs.input_ids.to(self._device),
                                           max_length=self._max_len,
                                           attention_mask=inputs["attention_mask"].to(self._device),
                                           pad_token_id=self._tokenizer.eos_token_id)
            res.append(self.remove_prompt(self._tokenizer.decode(outputs[0], skip_special_tokens=True),
                                          input=text))
        return res
