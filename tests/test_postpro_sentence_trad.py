
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


def async_callback(counter, result, error):
    if error is not None:
        print(f"Error reception from server : {str(error)}")
        counter.count_neg += 1
    if result is not None:
        print("Triton server answer :")
        for item in result.as_numpy("translation"):
            print("Translation output:", item.decode("UTF-8"))
        counter.count_pos += 1

def test_model(model_name="sentence_trad_postpro"):
    # grpc url should be prefered
    client = tclient.InferenceServerClient(url="localhost:8001")
    # Inputs
    prompts = ["Translate the following text from English to French:\nEnglish: Hello world\nFrench:"]
    tokens = np.array([[1, 28705, 6352, 4372, 395, 272, 2784, 15029, 28723, 9268, 7282, 442, 8291, 574, 11194, 28723, 4335, 10020, 272, 2296, 2245, 477, 4300, 298, 4949, 714, 13, 27871, 28747, 22557, 1526, 13, 28765, 4284, 28747]],dtype=np.int64)
    valid_mask = np.array([[True]], dtype=bool)

    prompt_obj = np.array(prompts, dtype="object").reshape([-1, 1])
    tokens_obj = tokens.reshape([1, -1])
    valid_mask_obj = valid_mask.reshape([1, -1])
    input_tensors = [
        tclient.InferInput("prompts", prompt_obj.shape, np_to_triton_dtype(prompt_obj.dtype)),
        tclient.InferInput("tokens", tokens_obj.shape, np_to_triton_dtype(tokens_obj.dtype)),
        tclient.InferInput("valid_mask", valid_mask_obj.shape, np_to_triton_dtype(valid_mask_obj.dtype)),
    ]
    input_tensors[0].set_data_from_numpy(prompt_obj)
    input_tensors[1].set_data_from_numpy(tokens_obj)
    input_tensors[2].set_data_from_numpy(valid_mask_obj)

    # Set outputs
    output = [tclient.InferRequestedOutput("translation")]

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
    
    test_model()
