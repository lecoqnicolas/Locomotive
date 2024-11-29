# Ensure we find locomotive llm in the pythonpath, as pytest do not add it.
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import argparse
from locomotive_llm.utils import RequestCounter, get_callback_with_counter, TritonLlmClient
import logging
import time


def test_model(model_name="madlad"):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    client = TritonLlmClient()

    # Inputs
    prompts = ["Hello, how are you?", "Hi I am ikram", "Je suis Triton"]
    src = ["English", "English", "French"]
    targets = ["French", "French", "German"]
    logging.debug(f"Sentences to translate :")
    logging.debug(f"{prompts}")
    counter = RequestCounter()
    callback = get_callback_with_counter(counter)
    # Set Inputs
    client.infer(model_name, texts=prompts, src_lang=src, tgt_lang=targets, callback=callback)

    logging.info("Doing other stuff while the answer is computed")
    while counter.request_count < 1:
        time.sleep(0.1)
    assert counter.pos_count == 1
if __name__ == "__main__":

    test_model()

    

