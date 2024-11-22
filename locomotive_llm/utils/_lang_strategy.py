import langcodes

TOWER_LANGS = ["English", "German", "French", "Spanish", "Chinese", "Portuguese", "Italian", "Russian", "Korean", "Dutch"]
TOWER_MODEL = "sentence_trad_tower"
TOWER_MODEL_DOCUMENT = "sentence_trad_tower_docs"
MADLAD_MODEL = "madlad"


def get_seq2seq_model(src_code:str, dest_code: str) -> str:
    return f"{src_code}_{dest_code}_seq2seq"


def get_model_prio(model_list: list[str], src_lang: str, tgt_lang: str, document_mode: bool = False) -> list[str]:
    priority_list = []
    src_code = langcodes.find(src_lang).language
    tgt_code = langcodes.find(tgt_lang).language
    seq2seq_model = get_seq2seq_model(src_code, tgt_code)

    # always prioritize the seq2seq, as it is less costly and faster
    if seq2seq_model in model_list:
        priority_list.append(seq2seq_model)

    # document_mode
    if document_mode:
        if tgt_lang in TOWER_LANGS and src_lang in TOWER_LANGS:
            priority_list.append(TOWER_MODEL_DOCUMENT)
    priority_list.append(MADLAD_MODEL)
    if not document_mode:
        priority_list.append(TOWER_MODEL)
    return priority_list
