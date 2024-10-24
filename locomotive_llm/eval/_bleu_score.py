from sacrebleu.metrics import BLEU
import logging

def eval_sentences_bleu(translations, references):
    """
    Evaluate BLEU score for translations compared to references.

    :param translations: List of translated texts.
    :param references: List of reference texts (multiple sets).
    :return: BLEU score rounded to 5 decimal places.
    """
    bleu = BLEU()
    score = bleu.corpus_score(translations, [references])
    logging.info(f"BLEU score: {score}")
    return round(score.score, 5)


def eval_text_bleu(translated_text, reference_text):
    """
    Evaluate a text line by line using BLEU score.
    """
    reference_lines = reference_text.split("\n")
    translated_lines = translated_text.split("\n")
    # Perform BLEU score calculation
    bleu = BLEU()
    score = bleu.corpus_score(translated_lines, [reference_lines])
    logging.info(f"BLEU score: {score}")
    
    return score.score
