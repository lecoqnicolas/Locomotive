import sacrebleu
import os
from comet import download_model, load_from_checkpoint
import torch
import logging
import argparse
import mlflow
from locomotive_llm.utils import log_dataclass, comet_eval, CometConfig
from locomotive_llm.documents_handling import DocxLoader
from langchain_community.document_loaders import TextLoader

def read_docx_with_langchain(file_path):
    """Reads a .docx file using the custom LangChain loader."""
    loader = DocxLoader(file_path)
    documents = loader.load()
    return "\n".join([doc.page_content for doc in documents])


def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
    
def read_ground_truth(file_path):
    """Reads the ground truth document (either .docx or .txt) using appropriate loaders."""
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == ".docx":
        return read_docx_with_langchain(file_path)  # Use the custom loader for .docx
    elif file_extension == ".txt":
        return read_txt(file_path)  # Use the standard function for .txt
    else:
        logging.error("Unsupported file format for ground truth. Only .docx and .txt files are supported.")
        exit(1)

def eval_bleu(translated_text, reference_text):
    reference = [reference_text.split("\n")]
    translated = translated_text.split("\n")
    bleu = sacrebleu.corpus_bleu(translated, reference)
    logging.info(f"BLEU score: {bleu.score}")
    return bleu.score

def eval_comet(translated_text, reference_text, src_text):
    model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(model_path)
    
    data = [
        {"src": src_text, "mt": translated_text, "ref": reference_text}
    ]
    
    comet_score = comet_model.predict(data, batch_size=8, gpus=1 if torch.cuda.is_available() else 0)
    logging.info(f"COMET score: {comet_score}")
    return comet_score

def eval_docs(translated_file, reference_file, src_file):
    # Read the source, translated and reference texts
    translated_text = read_docx_with_langchain(translated_file) if translated_file.endswith(".docx") else read_txt(translated_file)
    reference_text = read_ground_truth(reference_file)
    src_text = read_ground_truth(src_file)
    
    # Calculate BLEU score
    bleu_score = eval_bleu(translated_text, reference_text)

    # Calculate COMET score
    comet_score = eval_comet(translated_text, reference_text, src_text)
    
    return bleu_score, comet_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a document translated using TowerLLM model')
    
    # Add arguments for evaluation
    parser.add_argument('--input_file',
                        type=str,
                        required=True,
                        help='Path to the input document file (.docx or .txt).')
    parser.add_argument('--eval',
                        action='store_true',
                        help='Evaluate the translated document against ground truth.')
    
    parser.add_argument('--ground_truth',
                        type=str,
                        help='Path to the ground truth document for evaluation.')
    
    parser.add_argument('--source',
                        type=str,
                        help='Path to the source document (required for COMET evaluation).')
    
    args = parser.parse_args()

    # Perform evaluation
    if args.eval and args.ground_truth and args.source:
        bleu_score, comet_score = eval_docs(args.input_file, args.ground_truth, args.source)
        comet_score_value = comet_score['system_score']
        logging.info(f"Evaluation completed. BLEU: {bleu_score}, COMET: {comet_score_value}")
        
        # Log the metrics
        mlflow.log_metric("bleu_score", bleu_score)
        mlflow.log_metric("comet_score", comet_score_value)
    else:
        logging.error("To run evaluation, --ground_truth and --source must be provided.")
