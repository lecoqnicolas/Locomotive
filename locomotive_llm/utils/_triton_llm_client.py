import numpy as np
import tritonclient.grpc as tclient
from tritonclient.utils import np_to_triton_dtype
from typing import Callable


class TritonLlmClient:
    def __init__(self, url: str = "localhost:8001"):
        self._client = tclient.InferenceServerClient(url=url)
        self._translate_input = "text_to_translate"
        self._src_input = "src_name"
        self._target_input = "tgt_name"
        self._output_name = "translation"

    def infer(self, model_name: str, texts: list[str], src_lang: list[str], tgt_lang: list[str], callback: Callable)\
            -> None:
        text_obj = np.array(texts, dtype="object").reshape(-1,1)
        src_obj = np.array(src_lang, dtype="object").reshape(-1,1)
        tgt_obj = np.array(tgt_lang, dtype="object").reshape(-1,1)
        print(text_obj)
        print(src_obj)
        print(tgt_obj)
        # Set Inputs
        input_tensors = [
            tclient.InferInput(
                self._translate_input, text_obj.shape, np_to_triton_dtype(text_obj.dtype)
            ),
            tclient.InferInput(
                self._src_input, src_obj.shape, np_to_triton_dtype(text_obj.dtype)
            ),
            tclient.InferInput(
                self._target_input, tgt_obj.shape, np_to_triton_dtype(text_obj.dtype)
            ),
        ]
        input_tensors[0].set_data_from_numpy(text_obj)
        input_tensors[1].set_data_from_numpy(src_obj)
        input_tensors[2].set_data_from_numpy(tgt_obj)

        # Set outputs
        output = [tclient.InferRequestedOutput(self._output_name)]

        # Asynchronous Query
        self._client.async_infer(
            model_name=model_name,
            inputs=input_tensors,
            outputs=output,
            callback=callback
        )
