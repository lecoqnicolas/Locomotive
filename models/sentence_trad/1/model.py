
import torch

from locomotive_llm.load import load_config
from locomotive_llm.model import get_pipeline

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self._config = load_config("config.yaml")
        pipeline_class = get_pipeline(self._config)
        self._model = pipeline_class(model_id=self._config.llm_model,
                               device="cuda" if torch.cuda.is_available() else "cpu",
                               prompt_file=self._config.prompt,
                               max_tokens=self._config.max_token,
                               output_parser=self._config.response_parsing_method,
                               use_context=self._config.use_context)

    def execute(self, requests):
        responses = []
        for request in requests:
            text_tensor = pb_utils.get_input_tensor_by_name(request, "text_to_translate")
            src_name = pb_utils.get_input_tensor_by_name(request, "src_name")
            tgt_name = pb_utils.get_input_tensor_by_name(request, "tgt_name")
            print("Model received", flush=True)
            print(text_tensor.as_numpy(), flush=True)
            print(src_name.as_numpy(), flush=True)
            print(tgt_name.as_numpy(), flush=True)
            translated_text = self._model.transform(text_tensor.as_numpy()[0], src_name.as_numpy[0], tgt_name.as_numpy[0])

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "translation",
                        np.array(translated_text),
                    )
                ]
            )
            responses.append(inference_response)
        return responses