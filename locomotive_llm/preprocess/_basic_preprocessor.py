from langchain.prompts import load_prompt

from ._utils import clean_inputs


class BasicPreprocessor:
    def __init__(self, prompt_file: str, ignore_prompts: list, use_lang_code: bool = False) -> None:
        self._prompt = load_prompt(prompt_file)
        self._ignore_prompts = set(str(item) for item in ignore_prompts) if len(ignore_prompts) > 0 else {}
        self._use_lang_code = use_lang_code

    def transform(self, texts: list[str], src_lang: str | list[str], tgt_lang: str | list[str]) \
            -> tuple[list[str], list[bool]]:
        valid_mask, valid_texts, valid_src_lang, valid_tgt_lang = clean_inputs(texts, src_lang, tgt_lang,
                                                                               self._ignore_prompts,
                                                                               self._use_lang_code)
        prompts = [self._prompt.format(text=text, src_lang=src, tgt_lang=tgt)
                   for text, src, tgt in zip(valid_texts, valid_src_lang, valid_tgt_lang)]
        return prompts, valid_mask
