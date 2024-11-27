import numpy as np
from optimum.onnxruntime import ORTModelForCausalLM
import triton_python_backend_utils as pb_utils
import torch
class TritonPythonModel:
    def initialize(self, args):
        """
        Initializes the ONNX model session and loads it into memory.
        """
        providers = ["CUDAExecutionProvider"]
        model_path = "./tower_onnx/"
        self.model = ORTModelForCausalLM.from_pretrained(model_path, use_cache=False, providers=providers, use_io_binding=False)
        self._version = "onnx_tower"
        self.logger = pb_utils.Logger  # Instantiate the logger
        self.logger.log_info(f"Model ONNX {self._version} loaded successfully.")

    def execute(self, requests):
        """
        Handles multiple inference requests and processes inputs and outputs.
        """
        responses = []
        for request in requests:
            try:
              
                # Retrieve input tensors
                input_ids = pb_utils.get_input_tensor_by_name(request, "input_ids").as_numpy()
                attention_mask = pb_utils.get_input_tensor_by_name(request, "attention_mask").as_numpy()
                print(f"input_ids shape in onnx model: {input_ids.shape}")
                print(f"attention_mask shape in onnx model: {attention_mask.shape}")
                
                input_ids = np.expand_dims(input_ids, axis=0) if input_ids.ndim == 1 else input_ids
                
                input_ids = input_ids.astype(np.int64)
                attention_mask = attention_mask.astype(np.int64)

                self.logger.log_info(f"input_ids shape: {input_ids.shape}, attention_mask shape: {attention_mask.shape}")
                # Perform inference
                gen_tokens = self.model.generate(
                    input_ids=torch.tensor(input_ids),
                    attention_mask=torch.tensor(attention_mask),
                    max_new_tokens=256,
                )
                

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
            except Exception as e:
                
                error_message = f"Error during inference: {str(e)}"
                self.logger.log_error(error_message)
                responses.append(pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(error_message)
                ))

        return responses
