import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoModelForCausalLM
import torch
import torch.nn.functional as F

class TritonPythonModel:
    def initialize(self, args):
        """
        Initializes the ONNX model session and loads it into memory.
        """
        self._model = AutoModelForCausalLM.from_pretrained("Unbabel/TowerInstruct-Mistral-7B-v0.2", torch_dtype=torch.float16).to("cuda")
        self._model.eval()
        self._version = "tower_transformer"
        self.logger = pb_utils.Logger  # Instantiate the logger
        self.logger.log_info(f"Module Tower {self._version} loaded successfully.")

    def execute(self, requests):
        """
        Handles multiple inference requests and processes inputs and outputs.
        """
        responses = []
        for request in requests:
              
            # Retrieve input tensors
            input_ids = pb_utils.get_input_tensor_by_name(request, "input_ids").as_numpy()
            attention_mask = pb_utils.get_input_tensor_by_name(request, "attention_mask").as_numpy()
            print(f"input_ids shape in onnx model: {input_ids.shape}")
            print(f"attention_mask shape in onnx model: {attention_mask.shape}")
            print(input_ids)
            input_ids = np.expand_dims(input_ids, axis=0) if input_ids.ndim == 1 else input_ids

            input_ids = input_ids.astype(np.int64)
            input_ids = torch.tensor(input_ids, device="cuda")
            attention_mask = attention_mask.astype(np.int64)
            attention_mask  = torch.tensor(input_ids, device="cuda")
            with torch.no_grad():
                gen_tokens = self._model(input_ids=input_ids, attention_mask=attention_mask)
                self.logger.log_info(f"input_ids shape: {input_ids.shape}, attention_mask shape: {attention_mask.shape}")
                # Perform inference

                print(gen_tokens, flush=True)
                probabilities = F.softmax(gen_tokens.logits, dim=-1)
                gen_tokens = torch.argmax(probabilities, dim=-1)
            #translated_tokens = gen_tokens.cpu().numpy() if hasattr(gen_tokens, "cpu") else np.array(gen_tokens)
            translated_tokens = gen_tokens.cpu().numpy()
            translated_tokens = translated_tokens.astype(np.int64)
            self.logger.log_info(f"Translated tokens shape: {translated_tokens.shape}")
            print(f"Translated tokens shape: {translated_tokens.shape}")
            # Create Triton inference response
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor("translated_tokens", translated_tokens)]
            )
            responses.append(inference_response)

        return responses
