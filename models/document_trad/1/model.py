import os
from pathlib import Path
import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from locomotive_llm.load import load_config
from locomotive_llm.model import get_pipeline

class TritonPythonModel:
    def initialize(self, args):
        file_path = os.path.realpath(__file__)
        self._file_dir = Path(os.sep.join(file_path.split(os.sep)[:-1]))
        self._configuration_path = self._file_dir / "config.yaml"
        self._config = load_config(self._configuration_path)

        pipeline_class = get_pipeline(self._config)
        self._pipeline = pipeline_class(
            model_id=self._config.llm_model,
            device="cuda" if torch.cuda.is_available() else "cpu",
            prompt_file=self._config.prompt,
            batch_size=self._config.batch_size,
            output_parser=self._config.response_parsing_method,
            prompt_ignore=self._config.ignore_prompt,
            use_context=self._config.use_context,
            separateur_context=self._config.separateur_context,
            context_window=self._config.context_window,
        )

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
            texts = [text.decode("UTF-8") for text in text_tensor.as_numpy()[0]]
            src_lang = src_name.as_numpy()[0][0].decode("UTF-8")
            tgt_lang = tgt_name.as_numpy()[0][0].decode("UTF-8")

            translated_text = self._pipeline.transform(texts, src_lang, tgt_lang)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "translation",
                        np.array(translated_text, dtype="object"),
                    )
                ]
            )
            responses.append(inference_response)
        return responses

    def translate_documents(self, documents, src_lang, tgt_lang):
        translated_text = []
        for doc in documents:
            translated_lines = self._pipeline.transform([doc], src_lang, tgt_lang)
            translated_text.append(translated_lines[0])
        return translated_text
