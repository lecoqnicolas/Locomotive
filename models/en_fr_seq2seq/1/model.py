import os
from pathlib import Path
import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from locomotive_llm.load import load_config
from locomotive_llm.model import get_pipeline
from seq2seq_model.inference import Seq2SeqInference

class TritonPythonModel:
    def initialize(self, args):
        file_path = os.path.realpath(__file__)
        self._file_dir = Path(os.sep.join(file_path.split(os.sep)[:-1]))
        BASE_DIR = os.path.join(os.path.dirname(self._file_dir, "translate-en_fr-1_10"))
	self._inference_engine = Seq2SeqInference(BASE_DIR)

    def execute(self, requests):
        responses = []
        request_sizes = []
        texts = []
        languages_src = []
        languages_dest = []
        # batch the input requests
        for request in requests:
            text_tensor = pb_utils.get_input_tensor_by_name(request, "text_to_translate")
            src_name = pb_utils.get_input_tensor_by_name(request, "src_name")
            tgt_name = pb_utils.get_input_tensor_by_name(request, "tgt_name")
            print("Model received", flush=True)
            print(text_tensor.as_numpy(), flush=True)
            print(src_name.as_numpy(), flush=True)
            print(tgt_name.as_numpy(), flush=True)
            request_sizes.append(text_tensor.as_numpy().shape[0])
            texts.extend([text[0].decode("UTF-8") for text in text_tensor.as_numpy()])
            languages_src.extend([name[0].decode("UTF-8") for name in src_name.as_numpy()])
            languages_dest.extend([name[0].decode("UTF-8") for name in tgt_name.as_numpy()])
        translated_text = self._inference_engine(texts)

        # unbatch and send each translation to the request
        print(translated_text, flush=True)
        tot_size = 0
        for request_size in request_sizes:
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "translation",
                        np.array(translated_text[tot_size:tot_size+request_size], dtype="object"),
                    )
                ]
            )
            tot_size += request_size
            responses.append(inference_response)
        return responses

