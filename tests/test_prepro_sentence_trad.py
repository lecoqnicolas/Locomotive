
import numpy as np
import tritonclient.grpc as tclient
from tritonclient.utils import np_to_triton_dtype
import time
import argparse


class Counter:
    def __init__(self):
        self.count_pos = 0
        self.count_neg = 0


def async_callback(counter, result, error):
    if error is not None:
        print(f"Error reception from server : {str(error)}")
        counter.count_neg += 1
    if result is not None:
        print("Triton server answer :")
        print(result)
        for item in result.as_numpy("prompts"):
            print(item.decode("UTF-8"))
        for item in result.as_numpy("tokens"):
            print(item)
        for item in result.as_numpy("valid_mask"):
            print(item)
        counter.count_pos += 1
def test_sentence_trad(model_name):
    # grpc url should be prefered
    client = tclient.InferenceServerClient(url="localhost:8001")
    
    # Inputs
    prompts = ["Hello world"]
    print(f"Sentences to translate :")
    print(f"{prompts}")
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
    output = [
        tclient.InferRequestedOutput("prompts"),
        tclient.InferRequestedOutput("tokens"),
        tclient.InferRequestedOutput("valid_mask")
    ]

    counter = Counter()
    # Query
    client.async_infer(
        model_name, inputs=input_tensors, outputs=output, callback=lambda result, error: async_callback(counter, result, error)
    )

    while counter.count_neg + counter.count_pos < 1:
        time.sleep(0.1)
        print(f"neg {counter.count_neg}, pos {counter.count_pos}")
    assert counter.count_pos == 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton client for model inference")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use for inference")
    args = parser.parse_args()
    
    test_sentence_trad(args.model_name)
