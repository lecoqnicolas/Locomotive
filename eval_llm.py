import argparse
import os
import subprocess
from pathlib import Path

import torch

from locomotive_llm.model import TowerLlmPipeline
from locomotive_llm.load import load_flores, get_flores_file_path, load_config
from locomotive_llm.eval import evaluate_bleu

torch.cuda.empty_cache()


def main(params: argparse.Namespace) -> None:
    # Do nothing if no evaluation was selected
    if not params.bleu and not params.comet:
        print("Please select an evaluation method, bleu or comet")
        exit(1)

    # Load model configuration
    config = load_config(params.config, params.reverse)
    # load model
    model = TowerLlmPipeline(device="cuda" if torch.cuda.is_available() and not args.cpu else "cpu",
                             batch_size=args.batch_size)

    # output directories
    model_dirname = f"{config['from']['code']}_{config['to']['code']}-{config['version']}"
    run_dir = Path("run") / model_dirname

    # load flores dataset
    src_texts, tgt_texts = load_flores(config["from"]["code"], config["to"]["code"], params.flores_dataset)

    # optionnal, eval only specific sentences
    if params.flores_id is not None:
        src_texts = [src_texts[params.flores_id]]
        tgt_texts = [tgt_texts[params.flores_id]]

    # translate the texts
    translated_texts = model.transform(src_texts, src=config["from"]["name"], tgt=config["to"]["name"])

    # run the evaluations
    if params.bleu:
        bleu_score = evaluate_bleu(translated_texts, tgt_texts)
        print(f"BLEU score: {bleu_score}")

    if params.comet:
        src_f = get_flores_file_path(config["from"]["code"], params.flores_dataset)
        ref_f = get_flores_file_path(config["to"]["code"], params.flores_dataset)
        tra_f = os.path.join(run_dir, f"flores200_{params.flores_dataset}-{model_dirname}.evl")

        with open(tra_f, "w", encoding="utf8") as translation_file:
            for t in translated_texts:
                translation_file.write(t + "\n")

        subprocess.run([
            "comet-score",
            "--sources", src_f,
            "--translations", tra_f,
            "--references", ref_f,
            "--quiet", "--only_system"])


if __name__ == "__main__":
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
    parser.add_argument('--batch_size',
        type=int,
        default=32,
        help='Max batch size for translation. Default: %(default)s')

    args = parser.parse_args()

    main(args)
