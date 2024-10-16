import argparse
import logging

import mlflow

from locomotive_llm.load import read_doc
from locomotive_llm.eval import eval_text_bleu, eval_text_comet
from locomotive_llm.schema import LlmConfiguration


def main(params: argparse.Namespace) -> None:
    # ground and source must be provided
    if not params.ground_truth or not params.source:
        logging.error("To run evaluation, --ground_truth and --source must be provided.")
        return

    config = LlmConfiguration()
    # init mlflow
    mlflow.set_tracking_uri(config.mlflow_repository)
    mlflow.set_experiment("eval_document")

    with mlflow.start_run(run_name=f"{params.input_file}_{params.ground_truth}_{params.source}") as run:
        #log_dataclass(config)
        mlflow.log_param("eval_dataset", params.flores_dataset)
        # Read the source, translated and reference texts
        translated_text = read_doc(params.input_file)
        reference_text = read_doc(params.ground_truth)
        src_text = read_doc(params.source)

        # Calculate BLEU score
        bleu_score = eval_text_bleu(translated_text, reference_text)

        # Calculate COMET score
        comet_score = eval_text_comet(translated_text, reference_text, src_text)['system_score']

        logging.info(f"Evaluation completed. BLEU: {bleu_score}, COMET: {comet_score}")

        # Log the metrics
        mlflow.log_metric("bleu_score", bleu_score)
        mlflow.log_metric("comet_score", comet_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a document translated using TowerLLM model')
    # Add arguments for evaluation
    parser.add_argument('--input_file',
                        type=str,
                        required=True,
                        help='Path to the input document file (.docx or .txt).')
    parser.add_argument('--ground_truth',
                        type=str,
                        help='Path to the ground truth document for evaluation.')
    parser.add_argument('--source',
                        type=str,
                        help='Path to the source document (required for COMET evaluation).')
    args = parser.parse_args()

    main(args)
