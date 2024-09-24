import os
import json
import argparse
import subprocess
from sacrebleu import corpus_bleu
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# Functions for loading and handling FLORES dataset
def get_flores(lang_code, dataset="dev"):
    """
    Load FLORES corpus sentences for a given language and dataset.

    :param lang_code: The language code (e.g., "deu_Latn" or "fra_Latn").
    :param dataset: The dataset type (e.g., "dev", "devtest").
    :return: List of sentences in the specified language.
    """
    file_path = os.path.join("cache/flores200_dataset", dataset, f"{lang_code}.{dataset}")

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File for {lang_code} in dataset {dataset} not found.")

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


def load_towerllm_model():
    """
    Load the TowerLLM model and tokenizer from Huggingface.
    """
    tokenizer = AutoTokenizer.from_pretrained("Unbabel/TowerBase-7B-v0.1")
    model = AutoModelForCausalLM.from_pretrained("Unbabel/TowerBase-7B-v0.1")
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model = model.to(device)
    model.eval()
    return model, tokenizer


def translate_text(text, model, tokenizer):
    """
    Translate input text using the TowerLLM model.

    :param text: The text to translate.
    :param model: The TowerLLM model.
    :param tokenizer: The TowerLLM tokenizer.
    :return: The translated text.
    """
    inputs = tokenizer(text, return_tensors="pt").input_ids
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    outputs = model.generate(inputs, max_length=512)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text


# Argument parser setup
parser = argparse.ArgumentParser(description='Evaluate TowerLLM model')
parser.add_argument('--config',
    type=str,
    default="config_llm.json",
    help='Path to model-config.json. Default: %(default)s')
parser.add_argument('--reverse',
    action='store_true',
    help='Reverse the source and target languages in the configuration and data sources. Default: %(default)s')
parser.add_argument('--bleu',
    action="store_true",
    help='Evaluate BLEU score. Default: %(default)s')
parser.add_argument('--flores-id',
    type=int,
    default=None,
    help='Evaluate this FLORES sentence ID. Default: %(default)s')
parser.add_argument('--tokens',
    action="store_true",
    help='Display tokens rather than words. Default: %(default)s')
parser.add_argument('--flores_dataset',
    type=str,
    default="dev",
    help='Defines the FLORES200 dataset to translate. Default: %(default)s')
parser.add_argument('--translate_flores',
    action="store_true",
    help='Translate the FLORES200 corpus into a text file with .evl extension. Default: %(default)s')
parser.add_argument('--comet',
    action="store_true",
    help='Run COMET score command on the translated FLORES text. Default: %(default)s')
parser.add_argument('--cpu',
    action="store_true",
    help='Force CPU use. Default: %(default)s')
parser.add_argument('--max-batch-size',
    type=int,
    default=32,
    help='Max batch size for translation. Default: %(default)s')

args = parser.parse_args()

# Load model configuration
try:
    with open(args.config) as f:
        config = json.loads(f.read())
    if args.reverse:
        config["from"], config["to"] = config["to"], config["from"]
except Exception as e:
    print(f"Cannot open config file: {e}")
    exit(1)

current_dir = os.path.dirname(__file__)
cache_dir = os.path.join(current_dir, "cache")
model, tokenizer = load_towerllm_model()
model_dirname = f"{config['from']['code']}_{config['to']['code']}-{config['version']}"
run_dir = os.path.join(current_dir, "run", model_dirname)

def translator(src_texts):
    """
    Translate a list of source texts using the TowerLLM model.

    :param src_texts: List of source texts to translate.
    :return: List of translated texts.
    """
    return [translate_text(text, model, tokenizer) for text in src_texts]


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


def evaluate_bleu(translations, references):
    """
    Evaluate BLEU score for translations compared to references.

    :param translations: List of translated texts.
    :param references: List of reference texts.
    :return: BLEU score rounded to 5 decimal places.
    """
    return round(corpus_bleu(translations, [[ref] for ref in references]).score, 5)


if args.bleu or args.flores_id or args.translate_flores or args.comet is not None:
    src_texts, tgt_texts = load_flores(config["from"]["code"], config["to"]["code"], args.flores_dataset)

    if args.flores_id is not None:
        src_texts = [src_texts[args.flores_id]]
        tgt_texts = [tgt_texts[args.flores_id]]

    translated_texts = translator(src_texts)

    if args.bleu:
        bleu_score = evaluate_bleu(translated_texts, tgt_texts)
        print(f"BLEU score: {bleu_score}")

    if args.comet:
        src_f = get_flores_file_path(config["from"]["code"], args.flores_dataset)
        ref_f = get_flores_file_path(config["to"]["code"], args.flores_dataset)
        tra_f = os.path.join(run_dir, f"flores200_{args.flores_dataset}-{model_dirname}.evl")

        with open(tra_f, "w", encoding="utf8") as translation_file:
            for t in translated_texts:
                translation_file.write(t + "\n")

        subprocess.run([
            "comet-score",
            "--sources", src_f,
            "--translations", tra_f,
            "--references", ref_f,
            "--quiet", "--only_system"])

else:
    print("Starting interactive mode")
    while True:
        try:
            text = input(f"({config['from']['code']})> ")
        except KeyboardInterrupt:
            print("")
            exit(0)

        translated_text = translate_text(text, model, tokenizer)
        print(f"({config['to']['code']})> {translated_text}")
