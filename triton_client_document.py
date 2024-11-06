import numpy as np
import tritonclient.grpc as tclient
from tritonclient.utils import np_to_triton_dtype
from pathlib import Path
from locomotive_llm.load import DocumentTemplate, PDFDocumentTemplate, read_doc
from locomotive_llm.save import write_doc
import time


def async_callback(output_path, doc_template, result, error):
    if error is not None:
        print(f"Error received from server: {str(error)}")
    if result is not None:
        translated_text = [text.decode("UTF-8") for text in result.as_numpy("translation")]
        if doc_template:
            print(f"Using template: {type(doc_template).__name__}")
            doc_template.map_translations(translated_text)
            try:
                doc_template.save(output_path)
                print(f"Document saved to {output_path}")
            except Exception as e:
                print(f"Error saving document: {e}")
        else:
            output_path = output_path if output_path.endswith('.txt') else output_path + ".txt"
            write_doc("\n".join(translated_text), output_path)
            print(f"Text translation saved to {output_path}")


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


def main():
    client = tclient.InferenceServerClient(url="localhost:8001")

    output_file = "translated_doc1.docx"

    doc_path = "documents/input/DE_version_doc1.docx"
    raw_content, doc_template = load_document(doc_path)
    all_lines = [[el["content"]] for el in raw_content] if doc_path.endswith(('.docx', '.pdf')) else raw_content

    text_obj = np.array(all_lines, dtype="object")
    src_obj = np.array([["German"] for _ in all_lines], dtype="object")
    tgt_obj = np.array([["French"] for _ in all_lines], dtype="object")
    print(text_obj)
    print(src_obj)
    print(tgt_obj)
    # Set Inputs
    input_tensors = [
        tclient.InferInput(
            "text_to_translate", text_obj.shape, np_to_triton_dtype(text_obj.dtype)
        ),
        tclient.InferInput(
            "src_name", src_obj.shape, np_to_triton_dtype(text_obj.dtype)
        ),
        tclient.InferInput(
            "tgt_name", tgt_obj.shape, np_to_triton_dtype(text_obj.dtype)
        ),
    ]
    input_tensors[0].set_data_from_numpy(text_obj)
    input_tensors[1].set_data_from_numpy(src_obj)
    input_tensors[2].set_data_from_numpy(tgt_obj)

    # Set outputs
    output = [tclient.InferRequestedOutput("translation")]

    # Asynchronous Query
    client.async_infer(
        model_name="document_trad",
        inputs=input_tensors,
        outputs=output,
        callback=lambda result, error: async_callback(output_file, doc_template, result, error)
    )

    print("Doing other stuff while the answer is computed")
    time.sleep(60)  # Wait for async task to complete


if __name__ == "__main__":
    main()
