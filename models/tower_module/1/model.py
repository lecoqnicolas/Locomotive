import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoModelForCausalLM
import torch
import os
from pathlib import Path
from locomotive_llm.load import load_config
import torch.nn.functional as F

class TritonPythonModel:
    def initialize(self, args):
        """
        Initializes the ONNX model session and loads it into memory.
        """
        file_path = os.path.realpath(__file__)
        self._file_dir = Path(os.sep.join(file_path.split(os.sep)[:-1]))
        self._configuration_path = self._file_dir / "config.yaml"
        self._config = load_config(self._configuration_path)
        self._model = AutoModelForCausalLM.from_pretrained(self._config.llm_model, torch_dtype=torch.float16).to("cuda")
        self._model.eval()
        self._version = "tower_transformer"
        self.logger = pb_utils.Logger  # Instantiate the logger
        self.logger.log_info(f"Module Tower {self._version} loaded successfully.")

    def execute(self, requests):
        """
        Handles multiple inference requests and processes inputs and outputs.
        """
        responses = []
        request_sizes = []
        batch_id = []
        batch_mask = []
        for request in requests:
              
            # Retrieve input tensors
            input_ids = pb_utils.get_input_tensor_by_name(request, "input_ids").as_numpy()
            attention_mask = pb_utils.get_input_tensor_by_name(request, "attention_mask").as_numpy()
            print(f"input_ids shape in onnx model: {input_ids.shape}")
            print(f"attention_mask shape in onnx model: {attention_mask.shape}")
            print(input_ids)
            #input_ids = np.expand_dims(input_ids, axis=0) if input_ids.ndim == 1 else input_ids
            request_sizes.append(input_ids.shape[0])
            input_ids = input_ids.astype(np.int64)
            input_ids = torch.tensor(input_ids, device="cuda")
            attention_mask = attention_mask.astype(np.int64)
            attention_mask = torch.tensor(attention_mask, device="cuda")
            batch_id.append(input_ids)
            batch_mask.append(attention_mask)
        tensor_inputs = torch.cat(batch_id, 0)
        tensor_mask = torch.cat(batch_mask, 0)
        gen_tokens = self._model.generate(input_ids=tensor_inputs, attention_mask=tensor_mask, max_new_tokens=512, do_sample=False)
        #translated_tokens = gen_tokens.cpu().numpy() if hasattr(gen_tokens, "cpu") else np.array(gen_tokens)
        translated_tokens = gen_tokens.cpu().numpy()
        translated_tokens = translated_tokens.astype(np.int64)
        tot_size = 0
        for request_size in request_sizes:

            translated_tokens_slices = np.array(translated_tokens[tot_size:tot_size + request_size], dtype="int64")

            self.logger.log_info(f"Translated tokens shape: {translated_tokens_slices.shape}")
            print(f"Translated tokens shape: {translated_tokens.shape}")
            # Create Triton inference response
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor("translated_tokens", translated_tokens_slices)]
            )
            responses.append(inference_response)
            tot_size += request_size
        return responses
