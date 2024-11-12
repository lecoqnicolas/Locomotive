import langcodes


def is_text_valid(text: str, unvalid_prompts: set) -> bool:
    return isinstance(text, str) and text not in unvalid_prompts


def filter_clean_langs(lang_list:  str | list[str], valid_mask: list[bool], output_lang_code: bool)\
        -> list[str]:
    if isinstance(lang_list, list):
        valid_src_lang = [lang for i, lang, text in enumerate(lang_list) if valid_mask[i]]
    else:
        valid_src_lang = [lang_list for idx in range(len(lang_list)) if valid_mask[idx]]
    # convert lang names to code if asked
    if output_lang_code:
        valid_src_lang = [langcodes.find(lang).language for lang in valid_src_lang]
    return valid_src_lang


def clean_inputs(texts: list[str], src_lang: str | list[str], tgt_lang: str | list[str], ignore_prompts: set[str],
                 output_lang_code: bool) -> tuple[list[bool], list[str], list[str], list[str]]:
    valid_mask = [is_text_valid(text, ignore_prompts) for text in texts]
    valid_texts = [text for i, text in enumerate(texts) if valid_mask[i]]

    valid_src_lang = filter_clean_langs(valid_src_lang)

    if isinstance(src_lang, list):
        valid_src_lang = [lang for i, lang, text in enumerate(src_lang) if valid_mask[i]]
    else:
        valid_src_lang = [src_lang for idx in range(len(texts)) if valid_mask[idx]]
    if use_lang_code:
        valid_src_lang = [langcodes.find(lang).language for lang in valid_src_lang]

    if isinstance(tgt_lang, list):
        valid_tgt_lang = [lang for i, lang, text in enumerate(tgt_lang) if valid_mask[i]]
    else:
        valid_tgt_lang = [tgt_lang for idx in range(len(texts)) if valid_mask[idx]]
    return valid_mask, valid_texts, valid_src_lang, valid_tgt_lang


