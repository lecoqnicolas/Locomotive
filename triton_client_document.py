import numpy as np
import tritonclient.grpc as tclient
from tritonclient.utils import np_to_triton_dtype
from pathlib import Path
from locomotive_llm.load import DocumentTemplate, PDFDocumentTemplate, read_doc
import time

def async_callback(result, error):
    if error is not None:
        print(f"Error received from server: {str(error)}")
    if result is not None:
        translated_text = result.as_numpy("translation")[0]
        print("Triton server answer:")
        for text in translated_text:
            print(text.decode('UTF-8'))


def load_document(path):
    file_extension = Path(path).suffix
    if file_extension == ".docx":
        doc = DocumentTemplate(path)
        return doc.get_content()
    elif file_extension == ".pdf":
        pdf_doc = PDFDocumentTemplate(path)
        return pdf_doc.get_content()
    else:
        texts = read_doc(path, use_langchain_txt=True)
        return [line for line in texts.split("\n") if line.strip()]


def main():
    client = tclient.InferenceServerClient(url="localhost:8002")

    # Inputs
    documents = [
        "documents/input/DE_version_doc1.docx", 
        "documents/input/DE_version_doc3.docx",
        "documents/input/190122_DE.pdf"
    ]

    all_lines = []
    for doc_path in documents:
        raw_content = load_document(doc_path)
        lines = [el["content"] for el in raw_content] if doc_path.endswith(('.docx', '.pdf')) else raw_content
        all_lines.extend(lines)

    text_obj = np.array([all_lines], dtype="object")

    input_tensors = [
        tclient.InferInput("text_to_translate", text_obj.shape, np_to_triton_dtype(text_obj.dtype)),
        tclient.InferInput("src_name", (1, 1), np_to_triton_dtype(np.array([["German"]], dtype="object").dtype)),
        tclient.InferInput("tgt_name", (1, 1), np_to_triton_dtype(np.array([["French"]], dtype="object").dtype)),
    ]
    
    input_tensors[0].set_data_from_numpy(text_obj)
    input_tensors[1].set_data_from_numpy(np.array([["German"]], dtype="object"))
    input_tensors[2].set_data_from_numpy(np.array([["French"]], dtype="object"))

    # Set outputs
    output = [tclient.InferRequestedOutput("translation")]

    # Asynchronous Query
    client.async_infer(
        model_name="document_trad", 
        inputs=input_tensors, 
        outputs=output, 
        callback=async_callback
    )

    print("Doing other stuff while the answer is computed")
    print(time.sleep(60))


if __name__ == "__main__":
    main()
