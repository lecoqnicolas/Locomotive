from sacrebleu import corpus_bleu
import logging


def eval_sentences_bleu(translations, references):
    """
    Evaluate BLEU score for translations compared to references.

    :param translations: List of translated texts.
    :param references: List of reference texts.
    :return: BLEU score rounded to 5 decimal places.
    """
    return round(corpus_bleu(translations, [[ref] for ref in references]).score, 5)


def eval_text_bleu(translated_text, reference_text):
    """
    Evaluate a text line by line using bleu score
    """
    reference = [reference_text.split("\n")]
    translated = translated_text.split("\n")
    bleu = corpus_bleu(translated, reference)
    logging.info(f"BLEU score: {bleu.score}")
    return bleu.score
