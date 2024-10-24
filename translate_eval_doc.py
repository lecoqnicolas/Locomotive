import argparse
import logging
import time

import mlflow
import torch

from locomotive_llm.eval import eval_text_bleu, eval_text_comet
from locomotive_llm.load import load_config, read_doc, DocumentTemplate, PDFDocumentTemplate
from locomotive_llm.model import get_pipeline
from locomotive_llm.save import write_doc
from locomotive_llm.utils import log_dataclass
from pathlib import Path

def main(params: argparse.Namespace) -> None:
    logging.basicConfig(level="DEBUG")
    config = load_config(params.config, params.reverse)
    pipeline_class = get_pipeline(config)
    pipeline = pipeline_class(model_id=config.llm_model,
                              device="cuda" if torch.cuda.is_available() and not params.cpu else "cpu",
                              prompt_file=config.prompt,
                              batch_size=config.batch_size,
                              output_parser=config.response_parsing_method, prompt_ignore=config.ignore_prompt,
                              use_context=config.use_context, separateur_context=config.separateur_context,
                              context_window=config.context_window)
    file_extension = Path(params.input_file).suffix
    if config.preserve_formatting:
        if file_extension ==".docx":
            doc = DocumentTemplate(params.input_file)
            texts = doc.get_content()
            lines = [el["content"] for el in texts]
        elif file_extension =='.pdf':
            pdf_doc = PDFDocumentTemplate(params.input_file)
            elements = pdf_doc.get_content()
            lines = [el['content'] for el in elements]
    else:
        texts = read_doc(params.input_file, use_langchain_txt=config.langchain_parsing)
        lines = [line for line in texts.split("\n") if line.strip()]

    reference_text = read_doc(params.ground_truth, use_langchain_txt=config.langchain_parsing)

    # init mlflow
    mlflow.set_tracking_uri(config.mlflow_repository)
    mlflow.set_experiment(config.experiment_name)
    # ground and source must be provided

    with mlflow.start_run(run_name=config.run_name) as run:
        log_dataclass(config)
        # Translate the text
        logging.info("Starting translation of the document.")
        total_start_time = time.time()
        translated_sentences = pipeline.transform(lines, config.src_name, config.tgt_name)
        translated_text = "\n".join(translated_sentences)

        logging.info(f"Total translation time: {time.time() - total_start_time:.2f} seconds.")
        if params.output_file is not None:
            if config.preserve_formatting:
                if file_extension ==".docx":
                    doc.map_translations(translated_sentences)
                    doc.save(params.output_file)
                elif file_extension =='.pdf':
                    pdf_doc.map_translations(translated_sentences)
                    pdf_doc.save(params.output_file)
            else:
                write_doc(translated_text, params.output_file, config.preserve_formatting)

        logging.info(f"Translation completed. Output saved at {params.output_file}.")
        # Calculate BLEU score
        bleu_score = eval_text_bleu(translated_text, reference_text)

        # Calculate COMET score
        comet_score = eval_text_comet(translated_text, reference_text, lines)['system_score']

        logging.info(f"Evaluation completed. BLEU: {bleu_score}, COMET: {comet_score}")

        # Log the metrics
        mlflow.log_metric("bleu_score", bleu_score)
        mlflow.log_metric("comet_score", comet_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Translate a document using TowerLLM model')

    parser.add_argument('--config',
                        type=str,
                        default="config/config_en_fr.yml",
                        help='Path to model-config.yml. Default: %(default)s')

    parser.add_argument('--reverse',
                        action='store_true',
                        help='Reverse the source and target languages in the configuration and data sources. Default: '
                             '%(default)s')

    parser.add_argument('--input_file',
                        type=str,
                        required=True,
                        help='Path to the document to translate (.docx or .txt).')

    parser.add_argument('--ground_truth',
                        type=str,
                        required=True,
                        help='Path to the ground truth for eval (.docx or .txt).')

    parser.add_argument('--cpu',
                        action="store_true",
                        help='Force CPU use. Default: %(default)s')

    parser.add_argument('--output_file',
                        type=str,
                        default=None,
                        help='Optional, Path to the output translated document file (.docx or .txt).')

    args = parser.parse_args()

    main(args)
