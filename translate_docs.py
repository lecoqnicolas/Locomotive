import argparse
import logging
import time
from pathlib import Path
import torch

from locomotive_llm.load import load_config, read_doc, DocumentTemplate
from locomotive_llm.model import get_pipeline
from locomotive_llm.save import write_doc


def main(params: argparse.Namespace) -> None:
    logging.basicConfig(level="DEBUG")
    config = load_config(params.config, params.reverse)
    pipeline_class = get_pipeline(config)
    pipeline = pipeline_class(model_id=config.llm_model,
                              device="cuda" if torch.cuda.is_available() and not params.cpu else "cpu",
                              prompt_file=config.prompt,
                              batch_size=config.batch_size,
                              output_parser=config.response_parsing_method, prompt_ignore=config.ignore_prompt, use_context=config.use_context, separateur_context= config.separateur_context)
    context_window = 2
    file_extension = Path(params.input_file).suffix

    if config.preserve_formatting:
        doc = DocumentTemplate(params.input_file)
        texts = doc.get_content()
    else:
        texts = read_doc(params.input_file, use_langchain_txt=config.langchain_parsing)

    # Translate the text
    logging.info("Starting translation of the document.")
    total_start_time = time.time()
    if config.preserve_formatting:
        # translate content per content
        contents = [el["content"] for el in texts]
        translated_text = translate_with_context(contents, pipeline, config.src_name, config.tgt_name, context_window)
        #for i, el in enumerate(texts):
        #    el["content"] = translated_text[i]
    else:
        # translate line by line
        lines = [line for line in texts.split("\n") if line.strip()]
        translated_sentences = translate_with_context(lines, pipeline, config.src_name, config.tgt_name, context_window)
        translated_text = "\n".join(translated_sentences)

    logging.info(f"Total translation time: {time.time() - total_start_time:.2f} seconds.")
    if config.preserve_formatting:
        doc.map_translations(translated_text)
        doc.save(params.output_file)
    else:
        write_doc(translated_text, params.output_file, config.preserve_formatting)
    
    logging.info(f"Translation completed. Output saved at {params.output_file}.")

def translate_with_context(texts, pipeline, src_lang, tgt_lang, context_window):
    translated_texts = []
    context = []
    for i, text in enumerate(texts):
        prev_context = context[-context_window:] if pipeline.use_context else None
        translated = pipeline.transform([text], src_lang, tgt_lang, [prev_context])[0]
        context.append(text)  # add translated or original text to the context, to be tested ? 
        translated_texts.append(translated)
    return translated_texts

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
    args = parser.parse_args()

    main(args)
