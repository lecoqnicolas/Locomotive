
import numpy as np
import tritonclient.grpc as tclient
from tritonclient.utils import np_to_triton_dtype
import time
import argparse
from transformers import AutoTokenizer

class Counter:
    def __init__(self):
        self.count_pos = 0
        self.count_neg = 0


def async_callback(counter, result, error, tokenizer):
    if error is not None:
        print(f"Error reception from server : {str(error)}")
        counter.count_neg += 1
    if result is not None:
        print("Triton server answer :")
        print(result)
        for item in result.as_numpy("prompts"):
            print(item)
        for item in result.as_numpy("tokens"):
            print(item)
            decoded_translation = tokenizer.decode(item, skip_special_tokens=True)
            print(f"Decoded translation: {decoded_translation.strip()}")
        for item in result.as_numpy("valid_mask"):
            print(item)
        counter.count_pos += 1

def test_model(model_name="sentence_trad_prepro"):
    # grpc url should be prefered
    client = tclient.InferenceServerClient(url="localhost:8001")
    tokenizer = AutoTokenizer.from_pretrained("./tower_onnx/")
    # Inputs
    prompts = ["Hello world", "other_sentence"]
    print(f"Sentences to translate :")
    print(f"{prompts}")
    text_obj = np.array(prompts, dtype="object")
    text_obj = text_obj.reshape([-1,1])
    src_obj = np.array(["English","English"], dtype="object")
    src_obj = src_obj.reshape([-1,1])
    tgt_obj = np.array(["French","French"], dtype="object")
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
        tclient.InferRequestedOutput("tokens"),
        tclient.InferRequestedOutput("valid_mask")
    ]

    counter = Counter()
    # Query
    client.async_infer(
        model_name, inputs=input_tensors, outputs=output, callback=lambda result, error: async_callback(counter, result, error, tokenizer)
    )

    while counter.count_neg + counter.count_pos < 1:
        time.sleep(0.1)
        print(f"neg {counter.count_neg}, pos {counter.count_pos}")
    assert counter.count_pos == 1


if __name__ == "__main__":
    
    test_model()
