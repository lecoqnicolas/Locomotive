import argparse
import time
from pathlib import Path

from locomotive_llm.load import DocumentTemplate, PDFDocumentTemplate, read_doc
from locomotive_llm.save import write_doc
from locomotive_llm.utils import TritonLlmClient, RequestCounter
import logging


def document_callback(counter: RequestCounter, output_path: str, doc_template: DocumentTemplate, result: object,
                      error: object):
    if error is not None:
        logging.error(f"Error received from server: {str(error)}")
        counter.neg_count += 1
    if result is not None:
        translated_text = [text.decode("UTF-8") for text in result.as_numpy("translation")]
        if doc_template:
            logging.info(f"Using template: {type(doc_template).__name__}")
            doc_template.map_translations(translated_text)
            try:
                doc_template.save(output_path)
                logging.info(f"Document saved to {output_path}")
            except Exception as e:
                logging.error(f"Error saving document: {e}")
        else:
            output_path = output_path if output_path.endswith('.txt') else output_path + ".txt"
            write_doc("\n".join(translated_text), output_path)
            logging.info(f"Text translation saved to {output_path}")
        counter.pos_count += 1


def load_document(path):
    file_extension = Path(path).suffix
    if file_extension == ".docx":
        doc = DocumentTemplate(path)
        return doc.get_content(), doc
    elif file_extension == ".pdf":
        pdf_doc = PDFDocumentTemplate(path)
        return pdf_doc.get_content(), pdf_doc
    else:
        texts = read_doc(path, use_langchain_txt=True)
        return [line for line in texts.split("\n") if line.strip()]


def main(model_name):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    client = TritonLlmClient()

    # load document
    output_file = "translated_doc1.pdf"
    doc_path = "documents/input/DE_version_doc1.pdf"
    raw_content, doc_template = load_document(doc_path)
    all_lines = [el["content"] for el in raw_content] if doc_path.endswith(('.docx', '.pdf')) else raw_content

    src_obj = ["German" for _ in all_lines]
    tgt_obj = ["French" for _ in all_lines]
    logging.info(all_lines)
    logging.info(src_obj)
    logging.info(tgt_obj)
    # Set Inputs
    counter = RequestCounter()
    client.infer(model_name=model_name, texts=all_lines, src_lang=src_obj, tgt_lang=tgt_obj,
                 callback=lambda result, error: document_callback(counter, output_file, doc_template, result, error))

    logging.info("Doing other stuff while the answer is computed")
    while counter.request_count < 1:
        time.sleep(0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton client for model inference")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use for inference")
    args = parser.parse_args()

    main(args.model_name)
