import argparse
import time
from pathlib import Path

from locomotive_llm.load import DocumentTemplate, PDFDocumentTemplate, read_doc
from locomotive_llm.save import write_doc
from locomotive_llm.utils import TritonLlmClient, RequestCounter
import logging


def merge_sentences(translated_sentences: list[str], mapping: list[int]):
    # map and merge every sentence into texts corresponding to the input texts
    previous_mapping = -1
    results = []
    for i, translated_sentence in enumerate(translated_sentences):
        # if the current sentence is not from the same text as the previous one, we create a new output element
        if mapping[i] != previous_mapping:
            results.append(translated_sentence)
        else:  # else it was from the same text, so we merge them together
            results[-1] = " ".join((results[-1], translated_sentence))
        previous_mapping = mapping[i]
    return results


def segment_sentences(texts: list[str], max_char_size = 512) -> tuple[list[str], list[int]]:
    """
    Segment texts into sentences.

    Args:
        texts: Input texts.
    Returns:
        list of sentences, list of mapping (in which input text was each sentence
    """
    sentences = []
    input_mapping = []
    for i, text in enumerate(texts):
        prev_len = 0
        for j, sentence in enumerate(text.split("\n")):
            if j > 0 and prev_len + len(sentence) < max_char_size:
                # merge the sentence with the previous one if small enough
                prev_len += len(sentence)
                sentences[-1] = " ".join([sentences[-1], sentence])
            else:
                # ad it as another one.
                sentences.append(sentence)
                input_mapping.append(i)
                prev_len = len(sentence)
    return sentences, input_mapping


def document_callback(counter: RequestCounter, texts: list, result: object,
                      error: object):
    if error is not None:
        logging.error(f"Error received from server: {str(error)}")
        counter.neg_count += 1
    if result is not None:
        for text in result.as_numpy("translation"):
            texts.append(text)
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
    output_path = "translated_doc1.pdf"
    doc_path = "documents/input/DE_version_doc1.pdf"
    raw_content, doc_template = load_document(doc_path)
    all_lines = [el["content"] for el in raw_content] if doc_path.endswith(('.docx', '.pdf')) else raw_content

    src_obj = ["German" for _ in all_lines]
    tgt_obj = ["French" for _ in all_lines]
    logging.info(all_lines)
    logging.info(src_obj)
    logging.info(tgt_obj)
    nb_requests = 0
    max_batch_size = 10

    counter = RequestCounter()
    all_lines, mapping = segment_sentences(all_lines)
    batches = []
    # Set Inputs
    for i in range(0, len(src_obj), max_batch_size):
        nb_requests += 1
        text_batch = list()
        batches.append(text_batch)
        client.infer(model_name=model_name, texts=all_lines[i:i+max_batch_size], src_lang=src_obj[i:i+max_batch_size],
                     tgt_lang=tgt_obj[i:i+max_batch_size],
                     callback=lambda result, error: document_callback(counter, text_batch, result, error))
    logging.info("Doing other stuff while the answer is computed")
    while counter.request_count < nb_requests:
        time.sleep(0.1)
    unbatched_results = [text for batch in batches for text in batch]
    translated_text = merge_sentences(unbatched_results, mapping)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton client for model inference")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use for inference")
    args = parser.parse_args()

    main(args.model_name)
