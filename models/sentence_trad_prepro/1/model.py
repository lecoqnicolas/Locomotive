import os
from pathlib import Path
from transformers import AutoTokenizer

from locomotive_llm.load import load_config
from locomotive_llm.preprocess import BasicPreprocessor
from locomotive_llm.utils import get_device

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        file_path = os.path.realpath(__file__)
        self._file_dir = Path(os.sep.join(file_path.split(os.sep)[:-1]))
        self._configuration_path = self._file_dir / "config.yaml"
        self._config = load_config(self._configuration_path)
        self._preprocessor = BasicPreprocessor(self._config.prompt, self._config.ignore_prompt, use_lang_code=False)
        self._tokenizer = AutoTokenizer.from_pretrained("./tower_onnx/")
        self._device = get_device(self._config.device)

    def execute(self, requests):

        responses = []
        request_sizes = []
        texts = []
        languages_src = []
        languages_dest = []

        for request in requests:
            text_tensor = pb_utils.get_input_tensor_by_name(request, "text_to_translate")
            src_name = pb_utils.get_input_tensor_by_name(request, "src_name")
            tgt_name = pb_utils.get_input_tensor_by_name(request, "tgt_name")
            print("Model received", flush=True)
            print(text_tensor.as_numpy(), flush=True)
            print(src_name.as_numpy(), flush=True)
            print(tgt_name.as_numpy(), flush=True)
            request_sizes.append(text_tensor.as_numpy().shape[0])
            texts.extend([text[0].decode("UTF-8") for text in text_tensor.as_numpy()])
            languages_src.extend([name[0].decode("UTF-8") for name in src_name.as_numpy()])
            languages_dest.extend([name[0].decode("UTF-8") for name in tgt_name.as_numpy()])
        prompts, valid_mask = self._preprocessor.transform(texts, languages_src, languages_dest)
        tokens = self._tokenizer(prompts, return_tensors="pt", padding=True).to(self._device)

        print(prompts, flush=True)
        print(tokens, flush=True)
        tot_size = 0
        for request_size in request_sizes:
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "valid_mask",
                        np.array(valid_mask[tot_size:tot_size + request_size], dtype="bool"),
                    ),
                    pb_utils.Tensor(
                        "prompts",
                        np.array(prompts[tot_size:tot_size + request_size], dtype="object"),
                    ),
                    pb_utils.Tensor(
                        "tokens",
                        np.array(tokens[tot_size:tot_size + request_size], dtype="object"),
                    )
                ]
            )
            tot_size += request_size
            responses.append(inference_response)
        return responses
