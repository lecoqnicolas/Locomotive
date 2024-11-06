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
        self._model = pipeline_class(
            model_id=self._config.llm_model,
            device=self._config.device,
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
        request_sizes = []
        texts = []
        languages_src = []
        languages_dest = []
        # batch the input requests
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
        translated_text = self._model.transform(texts,
                                                languages_src,
                                                languages_dest)
        # unbatch and send each translation to the request
        print(translated_text, flush=True)
        tot_size = 0
        for request_size in request_sizes:
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "translation",
                        np.array(translated_text[tot_size:tot_size+request_size], dtype="object"),
                    )
                ]
            )
            tot_size += request_size
            responses.append(inference_response)
        return responses

