from langchain.prompts import load_prompt

from ._utils import clean_inputs


class ContextPreprocessor:
    def __init__(self, prompt_file: str, ignore_prompts: list, use_lang_code: bool = False,
                 context_window: int = 1, context_sep: str = "\n"):
        self._ignore_prompts = set(str(item) for item in ignore_prompts) if len(ignore_prompts) > 0 else {}
        self._prompt = load_prompt(prompt_file)
        self._context_window = context_window
        self._context_sep = context_sep
        self._use_lang_code = use_lang_code

    def _create_prompt_with_context(self, texts: list[str], src_langs: list[str], tgt_langs: list[str],
                                    prev_contexts: list[str] = None) -> list[str]:
        return [self._prompt.format(text=text, src_lang=src_lang, tgt_lang=tgt_lang, context=context)
                    for text, context, src_lang, tgt_lang in zip(texts, prev_contexts, src_langs, tgt_langs)]

    def transform(self, texts: list[str], src_lang: str | list[str], tgt_lang: str | list[str]) \
            -> tuple[list[str], list[bool]]:
        valid_mask, valid_texts, valid_src_lang, valid_tgt_lang = clean_inputs(texts, src_lang, tgt_lang,
                                                                               self._ignore_prompts,
                                                                               self._use_lang_code)

        # if no context was provided, create one based on previous sentences received
        prev_contexts = [texts[i - self._context_window:i] for i in range(len(texts))]
        # merge the sentences over context_window for each context
        prev_contexts = [self._context_sep.join(map(str, context)) if len(context) else "" for context in
                         prev_contexts]
        valid_prev_contexts = [prev_context for idx, prev_context in enumerate(prev_contexts) if valid_mask[idx]]
        prompts = self._create_prompt_with_context(valid_texts, valid_src_lang, valid_tgt_lang, valid_prev_contexts)
        return prompts, valid_mask
