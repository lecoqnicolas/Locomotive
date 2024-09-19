import os


def load_flores(src_lang, tgt_lang, dataset="dev"):
    """
    Load source and target texts from FLORES dataset.

    :param src_lang: Source language code (e.g., "deu_Latn").
    :param tgt_lang: Target language code (e.g., "fra_Latn").
    :param dataset: Dataset partition to use (e.g., "dev", "devtest").
    :return: Tuple of (source texts, target texts).
    """
    src_texts = get_flores(src_lang, dataset)
    tgt_texts = get_flores(tgt_lang, dataset)
    return src_texts, tgt_texts


def get_flores(lang_code, dataset="dev"):
    """
    Load FLORES corpus sentences for a given language and dataset.

    :param lang_code: The language code (e.g., "deu_Latn" or "fra_Latn").
    :param dataset: The dataset type (e.g., "dev", "devtest").
    :return: List of sentences in the specified language.
    """
    file_path = get_flores_file_path(lang_code, dataset)

    # Read sentences from file
    with open(file_path, "r", encoding="utf-8") as file:
        sentences = file.readlines()

    # Strip leading/trailing whitespace and newline characters
    return [sentence.strip() for sentence in sentences]


def get_flores_file_path(lang_code, dataset="dev"):
    """
    Get the full file path of the FLORES dataset for a given language and dataset.

    :param lang_code: The language code (e.g., "deu_Latn" or "fra_Latn").
    :param dataset: The dataset type (e.g., "dev", "devtest").
    :return: Full path to the FLORES dataset file.
    """
    file_path = os.path.join("cache/flores200_dataset", dataset, f"{lang_code}.{dataset}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")

    return file_path
