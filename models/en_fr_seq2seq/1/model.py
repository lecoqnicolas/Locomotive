import os
from pathlib import Path
import numpy as np
import triton_python_backend_utils as pb_utils
from seq2seq_model.inference import Seq2SeqInference


class TritonPythonModel:
    def initialize(self, args):
        file_path = os.path.realpath(__file__)
        self._file_dir = Path(os.sep.join(file_path.split(os.sep)[:-1]))
        BASE_DIR = self._file_dir / "seq2seq_model" / "translate-en_fr-1_10"
        self._inference_engine = Seq2SeqInference(str(BASE_DIR))

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
        translated_text = self._inference_engine.infer(texts)
        # original inference engine create a list of (list of one sentence)
        translated_text = [text[0] for text in translated_text]
        # unbatch and send each translation to the request
        print(translated_text, flush=True)
        tot_size = 0
        for request_size in request_sizes:
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "translation",
                        np.array(translated_text[tot_size:tot_size + request_size], dtype="object"),
                    )
                ]
            )
            tot_size += request_size
            responses.append(inference_response)
        return responses
