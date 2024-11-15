import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def __init__(self):
        pass
    
    def execute(self, requests):
        responses = []
        for request in requests:
            tokens = pb_utils.get_input_tensor_by_name(request, "tokens").as_numpy()
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "translated_tokens",
                        np.array(tokens, dtype="int64"),
                    )
                ]
            )
            responses.append(inference_response)
        return responses
