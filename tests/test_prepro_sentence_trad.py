import sys
import os
import numpy as np
import tritonclient.grpc as tclient
from tritonclient.utils import np_to_triton_dtype
import time
from transformers import AutoTokenizer
import logging
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from locomotive_llm.utils import RequestCounter
from locomotive_llm.utils import TritonLlmClient, get_callback_with_counter


def test_model(model_name="sentence_trad_prepro"):
    """Test the sentence translation model."""
    logging.basicConfig(level=logging.DEBUG)
    counter = RequestCounter()
    callback = get_callback_with_counter(counter)
    client = TritonLlmClient()
    tokenizer = AutoTokenizer.from_pretrained("./tower_onnx_2/")

    # Inputs
    prompts = ["Hello world"]
    logging.debug(f"Sentences to translate: {prompts}")
    
    text_obj = np.array(prompts, dtype="object").reshape([-1, 1])
    src = ["English"]
    targets = ["French"]

    # Set Inputs
    client.infer(model_name=model_name, texts=prompts, src_lang=src, tgt_lang=targets, callback=callback)



    try:
        # Query the model asynchronously
        client.infer(model_name=model_name, texts=prompts, src_lang=src, tgt_lang=targets, callback=callback)


        # Wait for the request to complete or time out after 30 seconds
        start_time = time.time()
        while counter.request_count < 1:
            time.sleep(0.1)
            logging.debug(f"neg {counter.neg_count}, pos {counter.pos_count}")
            if time.time() - start_time > 30:  # Timeout after 30 seconds
                raise TimeoutError("Model inference timed out")

        # Assert successful response
        assert counter.pos_count == 1

    except Exception as e:
        logging.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    test_model()
