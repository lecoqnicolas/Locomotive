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
        self._tokenizer = AutoTokenizer.from_pretrained("./tower_onnx/")
        self._device = get_device(self._config.device)

        # Post-processor setup
        self._postprocessor = BasicPostProcessor(output_parsing_method=LlmResponseParser.keep_first_line, output_field=None)

    def execute(self, requests):
        responses = []

        for request in requests:
            # Extract input tensors
            prompts = pb_utils.get_input_tensor_by_name(request, "prompts")
            tokens = pb_utils.get_input_tensor_by_name(request, "translated_tokens").as_numpy()
            valid_mask = pb_utils.get_input_tensor_by_name(request, "valid_mask")

            # Check the shape of tokens
            print(f"Tokens shape before reshaping: {tokens.shape}")

            # Ensure tokens are a 2D NumPy array (if it's a list of lists, convert to NumPy array)
            if isinstance(tokens, list):
                tokens = np.array(tokens)

            # Check if the shape is correct for decoding
            if tokens.ndim == 3:
                # Assuming tokens have shape (batch_size, seq_len, vocab_size), truncate the last dimension
                tokens = tokens.reshape(tokens.shape[0], tokens.shape[1])

            elif tokens.ndim == 2:
                # Tokens already in the expected shape
                pass
            else:
                raise ValueError(f"Unexpected shape for tokens: {tokens.shape}")

            # Check the number of tokens (make sure it's not too large for decoding)
            if tokens.size > 256:
                tokens = tokens[:, :256]  # Truncate tokens if necessary

            # Decode tokens into text using the tokenizer
            try:
                decoded_translations = [self._tokenizer.decode(token_ids, skip_special_tokens=True) for token_ids in tokens]
            except Exception as e:
                raise ValueError(f"Error during decoding: {str(e)}")

            # Prepare the response with decoded translations
            translation_tensor = pb_utils.Tensor("translation", np.array(decoded_translations, dtype=object))
            inference_response = pb_utils.InferenceResponse(output_tensors=[translation_tensor])
            responses.append(inference_response)

        return responses
