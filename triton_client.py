import argparse
import logging
import time

from locomotive_llm.utils import TritonLlmClient, RequestCounter, get_callback_with_counter


def main(model_name):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    client = TritonLlmClient()
    
    # Inputs
    prompts = ["Hello, my name is Triton"]
    src = ["English"]
    targets = ["French"]
    logging.debug(f"Sentences to translate :")
    logging.debug(f"{prompts}")
    counter = RequestCounter()
    callback = get_callback_with_counter(counter)
    # Set Inputs
    client.infer(model_name=model_name, texts=prompts, src_lang=src, tgt_lang=targets, callback=callback)

    logging.info("Doing other stuff while the answer is computed")
    while counter.request_count < 1:
        time.sleep(0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton client for model inference")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use for inference")
    args = parser.parse_args()
    
    main(args.model_name)
