from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate, load_prompt


class TowerLlmPipeline:

    def __init__(self, model_id="Unbabel/TowerBase-7B-v0.1", device="cpu", max_len=512, batch_size=1000,  prompt_file=None):
        self._id = model_id
        self._device = device
        self._model = AutoModelForCausalLM.from_pretrained(self._id).to(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self._id)
        self._model.eval()
        self._max_len = max_len
        self._batch_size = batch_size
        self._prompt = load_prompt(prompt_file)
    def _create_prompt(self, text, src_lang, tgt_lang):
        return self._prompt.format(text=text, src_lang=src_lang, tgt_lang=tgt_lang)
    def _clean_output(self, output, prompt):
        cleaned_output = output.replace(prompt, "").strip()
        if "\n" in cleaned_output:
           cleaned_output = cleaned_output.split("\n")[0].strip()
        return cleaned_output
    def _remove_prompt(self, answer, input):
        return answer.split(input)[1:]

    def transform(self, texts, src=None, tgt=None):
        """
        Translate input text using the TowerLLM model.

        :param texts: list of texts to translate.
        :return: The translated texts.
        """
        res = []
        if src is not None and tgt is not None:
            texts = [self._create_prompt(text, src, tgt) for text in texts]
        for text in texts:
            inputs = self._tokenizer(text, return_tensors="pt")
            outputs = self._model.generate(inputs.input_ids.to(self._device),
                                           max_length=self._max_len,
                                           attention_mask=inputs["attention_mask"].to(self._device),
                                           pad_token_id=self._tokenizer.eos_token_id)
            res.append(self._remove_prompt(self._tokenizer.decode(outputs[0], skip_special_tokens=True),
                                          input=text))
        return res
