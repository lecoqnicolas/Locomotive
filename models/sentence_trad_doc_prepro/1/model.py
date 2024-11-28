import os
from pathlib import Path
from transformers import AutoTokenizer

from locomotive_llm.load import load_config
from locomotive_llm.preprocess import ContextPreprocessor
from locomotive_llm.utils import get_device
import torch
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        file_path = os.path.realpath(__file__)
        self._file_dir = Path(os.sep.join(file_path.split(os.sep)[:-1]))
        self._configuration_path = self._file_dir / "config.yaml"
        self._config = load_config(self._configuration_path)
        self._context_preprocessor = ContextPreprocessor(prompt_file=self._config.prompt, ignore_prompts=self._config.ignore_prompt, use_lang_code=False, context_window=self._config.context_window, context_sep=self._config.separateur_context)
        self._tokenizer = AutoTokenizer.from_pretrained(self._config.llm_model, local_files_only= True)

        self._device = get_device(self._config.device)

        self._version = "model_prepro_doc_tower"
        self._logger = pb_utils.Logger
        self._logger.log_info(f"Model preprocessing {self._version} loaded")

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
        prompts, valid_mask = self._context_preprocessor.transform(texts, languages_src, languages_dest)
        #prompts tokenization
        tokens = self._tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self._device)
        #input_ids = tokens['input_ids'].unsqueeze(1) 
        #input_ids = input_ids.squeeze(1)
        input_ids_np = tokens['input_ids'].cpu().numpy()
        attention_mask_np = tokens['attention_mask'].cpu().numpy()
        print(input_ids_np, flush=True)
        #traduction decoding
        tot_size = 0
        for request_size in request_sizes:
            valid_mask_np = np.array(valid_mask[tot_size:tot_size + request_size], dtype="bool").reshape(request_size, -1)
            prompts_np = np.array(prompts[tot_size:tot_size + request_size], dtype="object").reshape(request_size, -1)
            input_ids_np_slice = np.array(input_ids_np[tot_size:tot_size + request_size], dtype="int64")
            attention_mask_np_slice = np.array(attention_mask_np[tot_size:tot_size + request_size], dtype="int64")
            #
            print(input_ids_np_slice, flush=True)
            print(f"valid_mask shape: {valid_mask_np.shape}")
            print(f"prompts shape: {prompts_np.shape}")
            print(f"input_ids shape: {input_ids_np_slice.shape}")
            print(f"attention_mask shape: {attention_mask_np_slice.shape}")
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor("valid_mask", valid_mask_np),
                    pb_utils.Tensor("prompts", prompts_np),
                    pb_utils.Tensor("input_ids", input_ids_np_slice),
                    pb_utils.Tensor("attention_mask", attention_mask_np_slice),
                ]
            )

            tot_size += request_size
            responses.append(inference_response)
        print(len(responses), flush=True)
        return responses
