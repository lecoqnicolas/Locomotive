import numpy as np
import os
from pathlib import Path
from optimum.onnxruntime import ORTModelForCausalLM
import triton_python_backend_utils as pb_utils
import torch


class TritonPythonModel:
    def initialize(self, args):
        """
        Initializes the ONNX model session and loads it into memory.
        """
        providers = ["CUDAExecutionProvider"]
        file_path = os.path.realpath(__file__)
        self._file_dir = Path(os.sep.join(file_path.split(os.sep)[:-1]))
        model_path = self._file_dir / "./tower_onnx_2/"
        self.model = ORTModelForCausalLM.from_pretrained(model_path, use_cache=False, providers=providers, use_io_binding=False).to("cuda")
        self._version = "onnx_tower"
        self.logger = pb_utils.Logger  # Instantiate the logger
        self.logger.log_info(f"Model ONNX {self._version} loaded successfully.")

    def execute(self, requests):
        """
        Handles multiple inference requests and processes inputs and outputs.
        """
        responses = []
        for request in requests:
            inp = pb_utils.get_input_tensor_by_name(request, "input_ids")
            print(inp)
            # Retrieve input tensors
            input_ids = pb_utils.get_input_tensor_by_name(request, "input_ids").as_numpy()
            attention_mask = pb_utils.get_input_tensor_by_name(request, "attention_mask").as_numpy()
            print(f"input_ids shape in onnx model: {input_ids.shape}")
            print(f"attention_mask shape in onnx model: {attention_mask.shape}")
            print(input_ids)
            print(input_ids.dtype)
            print(attention_mask)
            input_ids = np.expand_dims(input_ids, axis=0) if input_ids.ndim == 1 else input_ids

            input_ids = input_ids.astype(np.int64)
            inputs_ids = torch.tensor(input_ids, device="cuda")
            
            attention_mask = attention_mask.astype(np.int64)
            attention_mask  = torch.tensor(input_ids, device="cuda")
            print(attention_mask, flush=True)
            print(input_ids, flush=True)
            self.logger.log_info(f"input_ids shape: {inputs_ids.shape}, attention_mask shape: {attention_mask.shape}")
            # Perform inference
            gen_tokens = self.model.generate(
                input_ids=inputs_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
            )
            print(gen_tokens, flush=True)
            # Ensure proper formatting for Triton outputs
            translated_tokens = gen_tokens.cpu().numpy() if hasattr(gen_tokens, "cpu") else np.array(gen_tokens)
            translated_tokens = translated_tokens.astype(np.int64)
            self.logger.log_info(f"Translated tokens shape: {translated_tokens.shape}")
            print(f"Translated tokens shape: {translated_tokens.shape}")
            # Create Triton inference response
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor("translated_tokens", translated_tokens)]
            )
            responses.append(inference_response)

        return responses
