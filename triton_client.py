# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import tritonclient.grpc as tclient
from tritonclient.utils import np_to_triton_dtype


def async_callback(result, error):
    #print(query_response)
    if error is not None:
        print(f"Error reception from server : {str(error)}")
    if result is not None:
        print("Triton server answer :")
        print(str(result.as_numpy("translation")[0].decode('UTF-8')))
    #print(str(query_response.as_numpy("translation")[0]))


def main():
    # to test the http protocol
    #client = tclient.InferenceServerClient(url="localhost:8000")
    # grpc url should be prefered
    client = tclient.InferenceServerClient(url="localhost:8001")
    
    # Inputs
    prompts = ["Hello world"]
    print(f"Sentence to translate :")
    print(f"{prompts[0]}")
    text_obj = np.array([prompts], dtype="object")

    # Set Inputs
    input_tensors = [
        tclient.InferInput(
            "text_to_translate", text_obj.shape, np_to_triton_dtype(text_obj.dtype)
        ),
        tclient.InferInput(
            "src_name", text_obj.shape, np_to_triton_dtype(text_obj.dtype)
        ),
        tclient.InferInput(
            "tgt_name", text_obj.shape, np_to_triton_dtype(text_obj.dtype)
        ),
    ]
    input_tensors[0].set_data_from_numpy(text_obj)
    input_tensors[1].set_data_from_numpy(np.array([["English"]], dtype="object"))
    input_tensors[2].set_data_from_numpy(np.array([["French"]], dtype="object"))

    # Set outputs
    output = [
        tclient.InferRequestedOutput("translation")
    ]

    # Query
    client.async_infer(
        model_name="sentence_trad", inputs=input_tensors, outputs=output, callback=async_callback
    )


    


if __name__ == "__main__":
    main()