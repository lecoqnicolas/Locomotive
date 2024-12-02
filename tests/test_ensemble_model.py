# Ensure we find locomotive llm in the pythonpath, as pytest do not add it.
import sys
import os
import logging
import time
import tritonclient.grpc as tclient
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from locomotive_llm.utils import RequestCounter
import numpy as np
from tritonclient.utils import np_to_triton_dtype


def async_callback(counter:RequestCounter , result, error):
    if error:
        logging.error(f"Error during inference: {error}")
        counter.neg_count += 1
        return

    if result is None:
        logging.error("Received None result from inference.")
        counter.neg_count += 1
        return

    try:
        translation = result.as_numpy("translation")

        logging.debug(f"translation: {translation}")
        counter.pos_count += 1
    except Exception as e:
        logging.error(f"Exception processing result: {str(e)}")
        counter.neg_count += 1

def eval_ensemble_model(model_name, prompts, src, targets):
    """
    General evaluation function for ensemble models.

    Args:
        model_name (str): Name of the model to test.
        prompts (list): List of sentences to translate.
        src (list): List of source languages.
        targets (list): List of target languages.
    """
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().setLevel(logging.DEBUG)

    client = tclient.InferenceServerClient(url="localhost:8001")
    counter = RequestCounter()


    logging.debug("Sentences to translate:")
    logging.debug(f"{prompts}")
    text_obj = np.array(prompts, dtype="object")
    text_obj = text_obj.reshape([-1,1])
    src_obj = np.array(src, dtype="object")
    src_obj = src_obj.reshape([-1,1])
    tgt_obj = np.array(targets, dtype="object")
    tgt_obj = tgt_obj.reshape([-1,1])
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
        tclient.InferRequestedOutput("translation"),
    ]

    client.async_infer(model_name, inputs=input_tensors, outputs=output, callback=lambda result, error: async_callback(counter, result, error))


    start_time = time.time()
    timeout = 60
    while counter.request_count < 1:
        time.sleep(0.1)
        logging.debug(f"neg {counter.neg_count}, pos {counter.pos_count}")
        if time.time() - start_time > timeout:
            logging.error("Timeout reached. Exiting.")
            break
    assert counter.pos_count == 1



def test_madlad():
    """
    Test the Madlad model with basic inputs.
    """
    prompts = ["Hello, how are you?", "Hi I am Ikram", "Je suis Triton"]
    src = ["English", "English", "French"]
    targets = ["French", "French", "German"]
    eval_ensemble_model("madlad", prompts, src, targets)


def test_ensemble_model_tower():
    """
    Test the Tower model with basic inputs.
    """
    prompts = ["Hello, how are you?", "Hi I am Ikram", "Je suis Triton"]
    src = ["English", "English", "French"]
    targets = ["French", "French", "German"]
    eval_ensemble_model("sentence_trad_tower", prompts, src, targets)


def test_ensemble_model_tower_docs():
    """
    Test the Tower Docs model with basic inputs.
    """
    prompts = ["Hello, how are you?", "Hi I am Ikram", "Je suis Triton"]
    src = ["English", "English", "French"]
    targets = ["French", "French", "German"]
    eval_ensemble_model("sentence_trad_tower_docs", prompts, src, targets)




def test_empty_phrase():
    """
    Test with empty prompts.
    """
    prompts = ["", "", ""]
    src = ["English", "English", "French"]
    targets = ["French", "French", "German"]

    # Skip evaluation for empty prompts
    if not any(prompts):
        logging.error("Empty prompts detected. Skipping test.")
        return

    eval_ensemble_model("sentence_trad_tower", prompts, src, targets)


def test_long_phrase():
    """
    Test with very long phrases.
    """
    long_text = "This is a very long sentence." * 50  # Create a long input
    prompts = [long_text, long_text, long_text]
    src = ["English", "English", "English"]
    targets = ["French", "French", "French"]
    eval_ensemble_model("sentence_trad_tower", prompts, src, targets)






if __name__ == "__main__":
    # Run all tests
    test_madlad()
    test_ensemble_model_tower()
    test_ensemble_model_tower_docs()
    test_empty_phrase()
    test_long_phrase()

   
