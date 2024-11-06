import numpy as np
import tritonclient.grpc as tclient
from tritonclient.utils import np_to_triton_dtype
import time


def async_callback(result, error):
    #print(query_response)
    if error is not None:
        print(f"Error reception from server : {str(error)}")
    if result is not None:
        print("Triton server answer :")
        for item in result.as_numpy("translation"):
            print(item.decode("UTF-8"))

def main():
    # to test the http protocol
    #client = tclient.InferenceServerClient(url="localhost:8000")
    # grpc url should be prefered
    client = tclient.InferenceServerClient(url="localhost:8001")
    
    # Inputs
    prompts = ["Hello world", "not hellow world"]
    print(f"Sentences to translate :")
    print(f"{prompts}")
    text_obj = np.array(prompts, dtype="object")
    src_obj = np.array(["English", "English"], dtype="object")
    tgt_obj = np.array(["French", "German"], dtype="object")
    
    # Set Inputs
    input_tensors = [
        tclient.InferInput(
            "text_to_translate", text_obj.shape, np_to_triton_dtype(text_obj.dtype)
        ),
        tclient.InferInput(
            "src_name", text_obj.shape, np_to_triton_dtype(text_obj.dtype)
        ),
        tclient.InferInput(
            "tgt_name", text_obj.shape, np_to_triton_dtype(text_obj.dtype)
        ),
    ]
    input_tensors[0].set_data_from_numpy(text_obj)
    input_tensors[1].set_data_from_numpy(src_obj)
    input_tensors[2].set_data_from_numpy(tgt_obj)

    # Set outputs
    output = [
        tclient.InferRequestedOutput("translation")
    ]

    # Query
    client.async_infer(
        model_name="sentence_trad", inputs=input_tensors, outputs=output, callback=async_callback
    )

    print("Doing other stuff while the answer is computed")
    print(time.sleep(60))


if __name__ == "__main__":
    main()
