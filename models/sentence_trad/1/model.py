import sys
print(sys.path)
print(sys.version)
print(sys.executable)
import torch
import os
from pathlib import Path

from locomotive_llm.load import load_config
from locomotive_llm.model import get_pipeline

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        file_path = os.path.realpath(__file__)
        self._file_dir = Path(os.sep.join(file_path.split(os.sep)[:-1]))
        self._configuration_path = self._file_dir / "config.yaml"
        self._config = load_config(self._configuration_path)
        pipeline_class = get_pipeline(self._config)
        self._model = pipeline_class(model_id=self._config.llm_model,
                               device=self._config.device,
                               prompt_file=self._config.prompt,
                               max_tokens=self._config.max_token,
                               output_parser=self._config.response_parsing_method,
                               use_context=self._config.use_context)

    def execute(self, requests):
        responses = []
        request_sizes = []
        texts = []
        languages_src = []
        languages_dest = []
        
        for request in requests:
            text_tensor = pb_utils.get_input_tensor_by_name(request, "text_to_translate")
            src_name = pb_utils.get_input_tensor_by_name(request, "src_name")
            tgt_name = pb_utils.get_input_tensor_by_name(request, "tgt_name")
            print("Model received", flush=True)
            print(text_tensor.as_numpy(), flush=True)
            print(src_name.as_numpy(), flush=True)
            print(tgt_name.as_numpy(), flush=True)
            request_sizes.append(text_tensor.as_numpy().shape[0])
            texts.extend([text.decode("UTF-8") for text in text_tensor.as_numpy()])
            languages_src.extend([name.decode("UTF-8") for name in src_name.as_numpy()])
            languages_dest.extend([name.decode("UTF-8") for name in tgt_name.as_numpy()])
        translated_text = self._model.transform(texts,
                                                languages_src,
                                                languages_dest)
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
