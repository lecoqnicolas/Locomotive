import argparse
import logging
import os
from pathlib import Path

import mlflow
import torch

from locomotive_llm.eval import eval_sentences_bleu
from locomotive_llm.load import load_flores, get_flores_file_path, load_config
from locomotive_llm.model import get_pipeline
from locomotive_llm.utils import log_dataclass, comet_eval, CometConfig


def batch_texts(texts, batch_size):
    """Divise les textes en batchs de taille batch_size"""
    for i in range(0, len(texts), batch_size):
        yield texts[i:i + batch_size]


def main(params: argparse.Namespace) -> None:
    # init logging
    logging.basicConfig(level="DEBUG")

    # Do nothing if no evaluation was selected
    if not params.bleu and not params.comet:
        logging.error("Please select an evaluation method, bleu or comet")
        exit(1)

    try:
        config = load_config(params.config, params.reverse)
        pipeline_class = get_pipeline(config)
        pipeline = pipeline_class(model_id=config.llm_model,
                                  device="cuda" if torch.cuda.is_available() and not params.cpu else "cpu",
                                  prompt_file=config.prompt,
                                  prompt_ignore=config.ignore_prompt,
                                  batch_size=config.batch_size,
                                  output_parser=config.response_parsing_method, use_context=config.use_context)

        model_dirname = f"{config.src_code}_{config.tgt_code}-{config.version}"
        run_dir = Path("run") / model_dirname
        os.makedirs(run_dir, exist_ok=True)

        # init mlflow
        mlflow.set_tracking_uri(config.mlflow_repository)
        mlflow.set_experiment(config.experiment_name)

        with mlflow.start_run(run_name=config.run_name) as run:
            log_dataclass(config)
            mlflow.log_param("eval_dataset", params.flores_dataset)
            src_texts, tgt_texts = load_flores(config.src_code, config.tgt_code, params.flores_dataset)
            if params.flores_id is not None:
                src_texts = [src_texts[params.flores_id]]
                tgt_texts = [tgt_texts[params.flores_id]]

            logging.info("Translating input texts")

            translated_texts = pipeline.transform(src_texts, config.src_name, config.tgt_name)

            torch.cuda.empty_cache()

            # Run the evaluations
            valid_translations = [text for text in translated_texts if text]
            if params.bleu:
                logging.info("Starting BLEU evaluation")
                bleu_score = eval_sentences_bleu(valid_translations, tgt_texts)
                logging.info(f"BLEU score: {bleu_score}")
                mlflow.log_metric("bleu_score", bleu_score)

            if params.comet:
                src_f = get_flores_file_path(config.src_code, params.flores_dataset)
                ref_f = get_flores_file_path(config.tgt_code, params.flores_dataset)
                tra_f = os.path.join(run_dir, f"flores200_{params.flores_dataset}-{model_dirname}.evl")

                with open(tra_f, "w", encoding="utf8") as translation_file:
                    for t in translated_texts:
                        translation_file.write(t + "\n")
                logging.info("Starting COMET evaluation")
                comet_conf = CometConfig(sources=src_f, translations=[tra_f], references=ref_f, quiet=True,
                                         only_system=True)
                sys_scores = comet_eval(comet_conf)
                mlflow.log_metric("comet_score", sys_scores[0])

    except Exception as e:
        logging.error(f"An error occurred during the evaluation: {str(e)}")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description='Evaluate TowerLLM model')
    parser.add_argument('--config', type=str, default="config/config_en_fr.yml", help='Path to model-config.json.')
    parser.add_argument('--reverse', action='store_true', help='Reverse the source and target languages.')
    parser.add_argument('--bleu', action="store_true", help='Evaluate BLEU score.')
    parser.add_argument('--flores-id', type=int, default=None, help='Evaluate this FLORES sentence ID.')
    parser.add_argument('--flores_dataset', type=str, default="dev", help='Defines the FLORES200 dataset to translate.')
    parser.add_argument('--translate_flores', action="store_true",
                        help='Translate the FLORES200 corpus into a text file.')
    parser.add_argument('--comet', action="store_true", help='Run COMET score command on the translated FLORES text.')
    parser.add_argument('--cpu', action="store_true", help='Force CPU use.')
    args = parser.parse_args()

    main(args)