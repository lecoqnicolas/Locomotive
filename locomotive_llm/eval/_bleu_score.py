from sacrebleu import corpus_bleu


def evaluate_bleu(translations, references):
    """
    Evaluate BLEU score for translations compared to references.

    :param translations: List of translated texts.
    :param references: List of reference texts.
    :return: BLEU score rounded to 5 decimal places.
    """
    return round(corpus_bleu(translations, [[ref] for ref in references]).score, 5)
