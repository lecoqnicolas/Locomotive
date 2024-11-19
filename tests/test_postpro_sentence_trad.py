# Ensure we find locomotive llm in the pythonpath, as pytest do not add it.
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import logging
import time

import numpy as np
import tritonclient.grpc as tclient
from tritonclient.utils import np_to_triton_dtype

from locomotive_llm.utils import RequestCounter, get_callback_with_counter


def test_model(model_name="sentence_trad_postpro"):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

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
        tclient.InferInput("translated_tokens", tokens_obj.shape, np_to_triton_dtype(tokens_obj.dtype)),
        tclient.InferInput("valid_mask", valid_mask_obj.shape, np_to_triton_dtype(valid_mask_obj.dtype)),
    ]
    input_tensors[0].set_data_from_numpy(prompt_obj)
    input_tensors[1].set_data_from_numpy(tokens_obj)
    input_tensors[2].set_data_from_numpy(valid_mask_obj)

    # Set outputs
    output = [tclient.InferRequestedOutput("translation")]

    counter = RequestCounter()
    # Query
    client.async_infer(
        model_name, inputs=input_tensors, outputs=output,
        callback=get_callback_with_counter(counter)
    )

    while counter.request_count < 1:
        time.sleep(0.1)
        logging.debug(f"neg {counter.neg_count}, pos {counter.pos_count}")
    assert counter.pos_count == 1
