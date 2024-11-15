import os
from pathlib import Path
from transformers import AutoTokenizer

from locomotive_llm.load import load_config
from locomotive_llm.utils import get_device
from locomotive_llm.postprocess import BasicPostProcessor, LlmResponseParser

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        file_path = os.path.realpath(__file__)
        self._file_dir = Path(os.sep.join(file_path.split(os.sep)[:-1]))
        self._configuration_path = self._file_dir / "config.yaml"
        self._config = load_config(self._configuration_path)
        self._tokenizer = AutoTokenizer.from_pretrained("./tower_onnx/")
        self._device = get_device(self._config.device)
        self._postprocessor = BasicPostProcessor(output_parsing_method=LlmResponseParser.keep_first_line, output_field=None)

    def execute(self, requests):

        responses = []
      

        for request in requests:
            # Extract input tensors
            prompts = pb_utils.get_input_tensor_by_name(request, "prompts").as_numpy()
            tokens = pb_utils.get_input_tensor_by_name(request, "translated_tokens").as_numpy()
            valid_mask = pb_utils.get_input_tensor_by_name(request, "valid_mask").as_numpy()

            # Decode tokens to text
            decoded_translations = [self._tokenizer.decode(token_ids, skip_special_tokens=True) for token_ids in tokens]
            
            # Prepare the response with decoded translations
            translation_tensor = pb_utils.Tensor("translation", np.array(decoded_translations, dtype=object))
            inference_response = pb_utils.InferenceResponse(output_tensors=[translation_tensor])
            responses.append(inference_response)

        return responses
