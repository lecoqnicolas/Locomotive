import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import numpy as np
import tritonclient.grpc as tclient
from tritonclient.utils import np_to_triton_dtype
import time
from transformers import AutoTokenizer
import logging
from pathlib import Path
from locomotive_llm.utils import RequestCounter, get_callback_with_counter



def async_callback(counter:RequestCounter , result, error, tokenizer):
    if error:
        logging.error(f"Error during inference: {error}")
        counter.neg_count += 1
        return

    if result is None:
        logging.error("Received None result from inference.")
        counter.neg_count += 1
        return

    try:
        prompts = result.as_numpy("prompts")
        input_ids = result.as_numpy("input_ids")
        attention_mask = result.as_numpy("attention_mask")
        valid_mask = result.as_numpy("valid_mask")

        logging.debug(f"Prompts: {prompts}")
        logging.debug(f"Input IDs: {input_ids}")
        logging.debug(f"Attention Mask: {attention_mask}")
        logging.debug(f"Valid Mask: {valid_mask}")
        counter.pos_count += 1
    except Exception as e:
        logging.error(f"Exception processing result: {str(e)}")
        counter.neg_count += 1
def test_model(model_name="sentence_trad_prepro"):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    counter = RequestCounter()
    # grpc url should be prefered
    cert_dir = Path("./certs")
    client = tclient.InferenceServerClient(url="localhost:8001",  ssl=True,
                                                     root_certificates=cert_dir / "ca_localhost.crt",
                                                     private_key=cert_dir / "client_localhost.key",
                                                     certificate_chain=cert_dir / "client_localhost.crt")
    tokenizer = AutoTokenizer.from_pretrained("./tower_onnx_2/")
    # Inputs
    prompts = ["Hello world"]
    logging.debug(f"Sentences to translate :")
    logging.debug(f"{prompts}")
    text_obj = np.array(prompts, dtype="object")
    text_obj = text_obj.reshape([-1,1])
    src_obj = np.array(["English"], dtype="object")
    src_obj = src_obj.reshape([-1,1])
    tgt_obj = np.array(["French"], dtype="object")
    tgt_obj = tgt_obj.reshape([-1,1])
    # Set Inputs
    input_tensors = [
        tclient.InferInput(
            "text_to_translate", text_obj.shape, np_to_triton_dtype(text_obj.dtype)
        ),
        tclient.InferInput(
            "src_name", src_obj.shape, np_to_triton_dtype(src_obj.dtype)
        ),
        tclient.InferInput(
            "tgt_name", tgt_obj.shape, np_to_triton_dtype(tgt_obj.dtype)
        ),
    ]
    input_tensors[0].set_data_from_numpy(text_obj)
    input_tensors[1].set_data_from_numpy(src_obj)
    input_tensors[2].set_data_from_numpy(tgt_obj)

    # Set outputs
    output = [
        tclient.InferRequestedOutput("prompts"),
        tclient.InferRequestedOutput("input_ids"),
        tclient.InferRequestedOutput("attention_mask"),
        tclient.InferRequestedOutput("valid_mask")
    ]

    # Query
    client.async_infer(
        model_name, inputs=input_tensors, outputs=output, callback=lambda result, error: async_callback(counter, result, error, tokenizer)
    )
    start_time = time.time()
    timeout = 60
    while counter.request_count < 1:
        time.sleep(0.1)
        logging.debug(f"neg {counter.neg_count}, pos {counter.pos_count}")
        if time.time() - start_time > timeout:
            logging.error("Timeout reached. Exiting.")
            break
    assert counter.pos_count == 1
if __name__ == "__main__":
    test_model()
    
