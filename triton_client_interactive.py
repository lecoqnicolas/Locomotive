import numpy as np
import tritonclient.grpc as tclient
from tritonclient.utils import np_to_triton_dtype
import time
import argparse

def async_callback(result, error):
    #print(query_response)
    if error is not None:
        print(f"Error reception from server : {str(error)}")
    if result is not None:
        
        translated_text = str(result.as_numpy("translation")[0].decode('UTF-8'))
        print(f"\n(French)>{translated_text} \n(English)>")
    #print(str(query_response.as_numpy("translation")[0]))


def main(model_name):
    # to test the http protocol
    #client = tclient.InferenceServerClient(url="localhost:8000")
    # grpc url should be prefered
    client = tclient.InferenceServerClient(url="localhost:8001")
    src_name = "English"
    tgt_name = "French"
    # Inputs
    translations = []
    try:
        while True:
            text = input(f"({src_name})>")
            if text.lower() == 'exit':
                print("Exiting translation interactive mode.")
                break
            text_obj = np.array([[text]], dtype="object")
            src_lang = np.array([[src_name]], dtype="object")
            tgt_lang = np.array([[tgt_name]], dtype="object")

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
            input_tensors[1].set_data_from_numpy(src_lang)
            input_tensors[2].set_data_from_numpy(tgt_lang)

            # Set outputs
            output = [
                tclient.InferRequestedOutput("translation")
            ]

            # Query
            client.async_infer(
                model_name, inputs=input_tensors, outputs=output, callback=async_callback
            )
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton client for model inference")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use for inference")
    args = parser.parse_args()
    
    main(args.model_name)
