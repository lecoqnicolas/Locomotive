import numpy as np
from tritonclient import grpc as triton_grpc
from tritonclient.grpc import InferInput, InferRequestedOutput

class EnsembleModel:
    def __init__(self, model_name: str, server_url: str):
        self.model_name = model_name
        self.server_url = server_url
        self.client = triton_grpc.InferenceServerClient(url=server_url)

    def preprocess(self, text_to_translate: str, src_name: str, tgt_name: str) -> dict:
        # Call the preprocessing model (sentence_trad_prepro)
        inputs = [
            InferInput("text_to_translate", [1], "STRING"),
            InferInput("src_name", [1], "STRING"),
            InferInput("tgt_name", [1], "STRING")
        ]
        
        inputs[0].set_data_from_numpy(np.array([text_to_translate], dtype="object"))
        inputs[1].set_data_from_numpy(np.array([src_name], dtype="object"))
        inputs[2].set_data_from_numpy(np.array([tgt_name], dtype="object"))

        outputs = [
            InferRequestedOutput("prompts"),
            InferRequestedOutput("tokens"),
            InferRequestedOutput("valid_mask")
        ]

        # Send request to the preprocessing model
        response = self.client.infer("sentence_trad_prepro", inputs, outputs=outputs)

        # Extract results
        prompts = response.as_numpy("prompts")
        tokens = response.as_numpy("tokens")
        valid_mask = response.as_numpy("valid_mask")

        return {"prompts": prompts, "tokens": tokens, "valid_mask": valid_mask}

    def translate(self, prompts: np.ndarray, tokens: np.ndarray, valid_mask: np.ndarray) -> str:
        # Call the translation model (dummy_translation_model)
        inputs = [
            InferInput("prompts", prompts.shape, "STRING"),
            InferInput("tokens", tokens.shape, "FP32"),
            InferInput("valid_mask", valid_mask.shape, "BOOL")
        ]
        inputs[0].set_data_from_numpy(prompts)
        inputs[1].set_data_from_numpy(tokens)
        inputs[2].set_data_from_numpy(valid_mask)

        outputs = [InferRequestedOutput("prompts"), InferRequestedOutput("tokens"), InferRequestedOutput("valid_mask")]

        # Send request to the dummy translation model
        response = self.client.infer("dummy_translation_model", inputs, outputs=outputs)
        
        # Extract results
        translated_prompts = response.as_numpy("prompts")
        translated_tokens = response.as_numpy("tokens")
        translated_valid_mask = response.as_numpy("valid_mask")

        return translated_prompts, translated_tokens, translated_valid_mask

    def postprocess(self, translated_prompts: np.ndarray, translated_tokens: np.ndarray, translated_valid_mask: np.ndarray) -> str:
        # Call the postprocessing model (sentence_trad_postpro)
        inputs = [
            InferInput("prompts", translated_prompts.shape, "STRING"),
            InferInput("tokens", translated_tokens.shape, "FP32"),
            InferInput("valid_mask", translated_valid_mask.shape, "BOOL")
        ]
        inputs[0].set_data_from_numpy(translated_prompts)
        inputs[1].set_data_from_numpy(translated_tokens)
        inputs[2].set_data_from_numpy(translated_valid_mask)

        outputs = [InferRequestedOutput("translation")]

        # Send request to the postprocessing model
        response = self.client.infer("sentence_trad_postpro", inputs, outputs=outputs)
        
        # Extract final translated text
        final_translation = response.as_numpy("translation")[0]

        return final_translation

    def process(self, text_to_translate: str, src_name: str, tgt_name: str) -> str:
        # Full pipeline of preprocessing, translation, and postprocessing
        preprocessed_data = self.preprocess(text_to_translate, src_name, tgt_name)
        translated_prompts, translated_tokens, translated_valid_mask = self.translate(
            preprocessed_data["prompts"], preprocessed_data["tokens"], preprocessed_data["valid_mask"]
        )
        final_translation = self.postprocess(
            translated_prompts, translated_tokens, translated_valid_mask
        )

        return final_translation

if __name__ == "__main__":
    model = EnsembleModel("ensemble_model", "localhost:8001")
    text_to_translate = "Hello, how are you?"
    src_name = "en"
    tgt_name = "fr"

    result = model.process(text_to_translate, src_name, tgt_name)
    print(f"Translated Text: {result}")
