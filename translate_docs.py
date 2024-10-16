import argparse
import logging
import time

import torch

from locomotive_llm.load import load_config, read_doc
from locomotive_llm.model import get_pipeline
from locomotive_llm.save import write_doc


def translate_document_with_formatting(pipeline, elements, src_lang, tgt_lang):
    translated_elements = []

    # Split paragraphs and table rows for sequential processing
    for el in elements:
        if el['type'] == 'paragraph' and el['content'].strip():
            # Translate paragraph in batch if needed
            paragraphs = [el['content']]
            for batch in paragraphs:
                translated_paragraphs = pipeline.transform(batch, src_lang, tgt_lang)
                for text in translated_paragraphs:
                    translated_elements.append({'type': 'paragraph', 'content': text})

        elif el['type'] == 'table':
            # Translate each table row one by one
            translated_table = []
            for row in el['content']:
                translated_row = pipeline.transform(row, src_lang, tgt_lang)
                translated_table.append(translated_row)
            translated_elements.append({'type': 'table', 'content': translated_table})

    return translated_elements


def main(params: argparse.Namespace) -> None:
    logging.basicConfig(level="DEBUG")
    config = load_config(params.config, params.reverse)
    pipeline_class = get_pipeline(config)
    pipeline = pipeline_class(model_id=config.llm_model,
                              device="cuda" if torch.cuda.is_available() and not params.cpu else "cpu",
                              prompt_file=config.prompt,
                              batch_size=config.batch_size,
                              output_parser=config.response_parsing_method)

    text = read_doc(params.input_file, use_langchain_txt=params.langchain_parsing,
                    preserve_formatting=params.preserve_formatting)

    # Translate the text
    logging.info("Starting translation of the document.")
    total_start_time = time.time()
    if params.preserve_formatting:
        translated_text = translate_document_with_formatting(pipeline, text, config.src_name, config.tgt_name)
    else:
        lines = [line for line in text.split("\n") if line.strip()]
        translated_sentences = pipeline.transform(lines, config.src_name, config.tgt_name)
        translated_text = "\n".join(translated_sentences)

    logging.info(f"Total translation time: {time.time() - total_start_time:.2f} seconds.")

    write_doc(translated_text, params.output_file, preserve_formatting=params.preserve_formatting)
    logging.info(f"Translation completed. Output saved at {params.output_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Translate a document using TowerLLM model')
    
    parser.add_argument('--config',
                        type=str,
                        default="config/config_en_fr.yml",
                        help='Path to model-config.yml. Default: %(default)s')
    
    parser.add_argument('--reverse',
                        action='store_true',
                        help='Reverse the source and target languages in the configuration and data sources. Default: %(default)s')
    
    parser.add_argument('--input_file',
                        type=str,
                        required=True,
                        help='Path to the input document file (.docx or .txt).')
    
    parser.add_argument('--output_file',
                        type=str,
                        required=True,
                        help='Path to the output translated document file (.docx or .txt).')

    parser.add_argument('--cpu',
                        action="store_true",
                        help='Force CPU use. Default: %(default)s')

    parser.add_argument('--langchain_parsing', action="store_true",
                        help='use langchain for document parsing')

    parser.add_argument('--preserve_formatting', action="store_true",
                        help='preserve and send to the llm the document format')
    args = parser.parse_args()

    main(args)
