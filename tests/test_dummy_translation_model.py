import numpy as np
import tritonclient.grpc as tclient
from tritonclient.utils import np_to_triton_dtype
import time

class Counter:
    def __init__(self):
        self.count_pos = 0
        self.count_neg = 0


def async_callback(counter, result, error):
    if error is not None:
        print(f"Error reception from server : {str(error)}")
        counter.count_neg += 1
    if result is not None:
        print("Triton server answer :")
        for item in result.as_numpy("translated_tokens"):
            print("Translated Tokens output:", item)
        counter.count_pos += 1


def test_dummy_translation(model_name="dummy_translation_model"):
    # Se connecter au serveur Triton (gRPC URL recommandé)
    client = tclient.InferenceServerClient(url="localhost:8001")
    
    # Inputs (dummy data pour test)
    tokens = np.array([1, 28705, 6352, 4372, 395, 272, 2784, 15029, 28723, 9268], dtype=np.int64)
    
    input_tensors = [
        tclient.InferInput("tokens", tokens.shape, np_to_triton_dtype(tokens.dtype)),
    ]
    input_tensors[0].set_data_from_numpy(tokens)

    # Set outputs
    output = [tclient.InferRequestedOutput("translated_tokens")]

    counter = Counter()
    # Requête asynchrone
    client.async_infer(
        model_name, inputs=input_tensors, outputs=output, callback=lambda result, error: async_callback(counter, result, error)
    )

    # Attendre la réponse du serveur
    while counter.count_neg + counter.count_pos < 1:
        time.sleep(0.1)
        print(f"Waiting for response... Negatives: {counter.count_neg}, Positives: {counter.count_pos}")
    
    assert counter.count_pos == 1, "Test failed: No positive responses received."


if __name__ == "__main__":
    test_dummy_translation()
