import numpy as np
import tritonclient.grpc as tclient
from tritonclient.utils import np_to_triton_dtype
import time


# Parameters
NUM_REQUESTS = [100]
BATCH_SIZE = 1
DELAY = 0


class Counter:
    def __init__(self):
        self.count_pos = 0
        self.count_neg = 0


def async_callback(counter, result, error):
    if error is not None:
        print(f"Error from server: {str(error)}")
        counter.count_neg += 1
    elif result is not None:
        translation = result.as_numpy("translation")[0].decode('UTF-8')
        counter.count_pos += 1
        print("Translation:", translation)
        print(len(result.as_numpy("translation")))


def send_request(client, batch_size, counter, prompt="Despite the numerous challenges we faced throughout our journey, including unexpected weather conditions, logistical difficulties, and the need to adapt to different cultures and languages, we managed to persevere and achieve our goals, demonstrating the power of teamwork, determination, and resilience in the face of adversity."):
    text_obj = np.array([[prompt] for _ in range(batch_size)], dtype="object")
    src_obj = np.array([["English"] for _ in range(batch_size)], dtype="object")
    tgt_obj = np.array([["French"] for _ in range(batch_size)], dtype="object")
    
    input_tensors = [
        tclient.InferInput("text_to_translate", text_obj.shape, np_to_triton_dtype(text_obj.dtype)),
        tclient.InferInput("src_name", src_obj.shape, np_to_triton_dtype(text_obj.dtype)),
        tclient.InferInput("tgt_name", tgt_obj.shape, np_to_triton_dtype(text_obj.dtype)),
    ]
    input_tensors[0].set_data_from_numpy(text_obj)
    input_tensors[1].set_data_from_numpy(src_obj)
    input_tensors[2].set_data_from_numpy(tgt_obj)

    output = [tclient.InferRequestedOutput("translation")]

    client.async_infer(
        model_name="sentence_trad", inputs=input_tensors, outputs=output, callback=lambda result, error: async_callback(counter, result, error)
    )


def test_concurrent_requests(client, num_requests, batch_size, delay=0):

    counter = Counter()
    start_time = time.time()
    for i in range(num_requests):
        send_request(client, batch_size, counter)
        time.sleep(delay)

    while counter.count_neg+counter.count_pos < num_requests:
        time.sleep(0.1)
        print(f"neg {counter.count_neg}, pos {counter.count_pos} on {num_requests}")

    end_time = time.time()
    print(f"Time taken for {num_requests} requests: {end_time - start_time:.2f} seconds")


def main():
    url = "localhost:8001"
    client = tclient.InferenceServerClient(url=url)

    print("Testing concurrent requests...")
    for num_requests in NUM_REQUESTS:
        print(f"\nTesting with {num_requests} requests:")
        test_concurrent_requests(client, num_requests, BATCH_SIZE, delay=DELAY)


if __name__ == "__main__":
    main()
