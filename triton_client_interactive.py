import argparse
import logging
import time

from locomotive_llm.utils import TritonLlmClient, RequestCounter


def async_callback(counter: RequestCounter, result, error):
    if error is not None:
        print(f"Error reception from server : {str(error)}")
        counter.neg_count += 1
    elif result is not None:
        translated_text = str(result.as_numpy("translation")[0].decode('UTF-8'))
        print(f"(French)>{translated_text} ")
        counter.pos_count += 1


def main(model_name):
    client = TritonLlmClient()

    src_name = "Russian"
    tgt_name = "French"
    # Inputs
    try:
        while True:
            text = input(f"({src_name})>")
            if text.lower() == 'exit':
                print("Exiting translation interactive mode.")
                break
            counter = RequestCounter()
            client.infer(model_name=model_name, texts=[text], src_lang=[src_name], tgt_lang=[tgt_name],
                         callback=lambda result, error: async_callback(counter, result, error))
            while counter.request_count < 1:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton client for model inference")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use for inference")
    args = parser.parse_args()
    
    main(args.model_name)
