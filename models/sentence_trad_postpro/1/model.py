import os
from pathlib import Path
from transformers import AutoTokenizer
import numpy as np
import triton_python_backend_utils as pb_utils
from locomotive_llm.load import load_config
from locomotive_llm.utils import get_device
from locomotive_llm.postprocess import BasicPostProcessor, LlmResponseParser

class TritonPythonModel:
    def initialize(self, args):
        # Get file path and configuration
        file_path = os.path.realpath(__file__)
        self._file_dir = Path(os.sep.join(file_path.split(os.sep)[:-1]))
        self._configuration_path = self._file_dir / "config.yaml"
        self._config = load_config(self._configuration_path)
        self._version = "model_postpro_tower"
        self._logger = pb_utils.Logger
        self._logger.log_info(f"Model postprocessing {self._version} loaded")
        self._tokenizer = AutoTokenizer.from_pretrained(self._config.llm_model)
        self._device = get_device(self._config.device)

        # Post-processor setup
        self._postprocessor = BasicPostProcessor(output_parsing_method=LlmResponseParser.keep_first_line, output_field=None)

    def execute(self, requests):
        responses = []

        for request in requests:
            print(request, flush=True)
            # Extract input tensors
            prompts = pb_utils.get_input_tensor_by_name(request, "prompts").as_numpy()
            print(prompts, flush=True)
            prompts = [text[0].decode("UTF-8") for text in prompts]
            tokens = pb_utils.get_input_tensor_by_name(request, "translated_tokens").as_numpy()
            valid_mask = pb_utils.get_input_tensor_by_name(request, "valid_mask").as_numpy()
            print(tokens)
            # Check the shape of tokens
            print(f"Tokens shape before reshaping: {tokens.shape}", flush=True)
            print(f"masks : {valid_mask}", flush=True)
            # Decode tokens into text using the tokenizer
            translations = self._tokenizer.batch_decode(tokens, skip_special_tokens=True)
            decoded_translations = self._postprocessor.transform(valid_mask=valid_mask , input_prompts=prompts, outputs=translations)
            
            # Prepare the response with decoded translations
            translation_tensor = pb_utils.Tensor("translation", np.array(decoded_translations, dtype=object))
            inference_response = pb_utils.InferenceResponse(output_tensors=[translation_tensor])
            responses.append(inference_response)

        return responses
