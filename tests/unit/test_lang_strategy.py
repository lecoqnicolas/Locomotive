import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from locomotive_llm.utils._lang_strategy import get_model_prio


def test_lang_strat():
    src = "French"
    dest = "English"
    prio = get_model_prio(["en_fr_seq2seq"], src, dest)
    assert len(prio) == 0
    prio = get_model_prio(["fr_en_seq2seq", "en_fr_seq2seq", "sentence_trad_tower", "madlad","madlad2"], src,
                          dest)
    assert len(prio) == 3
    assert prio[1] == "fr_en_seq2seq"
    assert prio[2] == "madlad"
    assert prio[2] == "sentence_trad_tower"
    prio = get_model_prio(["fr_en_seq2seq", "en_fr_seq2seq", "sentence_trad_tower", "madlad",
                           "sentence_trad_tower_docs"], src,
                          dest, document_mode=True)
    assert len(prio) == 3
    assert prio[1] == "fr_en_seq2seq"
    assert prio[2] == "sentence_trad_tower_docs"
    assert prio[2] == "madlad"
