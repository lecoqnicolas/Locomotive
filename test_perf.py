import numpy as np
import tritonclient.grpc as tclient
from tritonclient.utils import np_to_triton_dtype
import time
import argparse
from locomotive_llm.utils import TritonLlmClient, RequestCounter, get_callback_with_counter
import logging

# Parameters
NUM_REQUESTS = [100]
BATCH_SIZE = 10
DELAY = 0


def test_concurrent_requests(client, num_requests, batch_size, delay=0, model_name="sentence_trad"):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    counter = RequestCounter()
    callback = get_callback_with_counter(counter)
    start_time = time.time()
    for i in range(num_requests):
        texts = [f"{i} : Despite the numerous challenges we faced throughout our journey, including unexpected weather "
                  "conditions, logistical difficulties, and the need to adapt to different cultures and languages, we "
                  "managed to persevere and achieve our goals, demonstrating the power of teamwork, determination, and "
                  "resilience in the face of adversity." for i in range(batch_size)]
        src = ["English" for _ in range(batch_size)]
        tgt = ["French" for _ in range(batch_size)]
        client.infer(model_name=model_name, src_lang=src, tgt_lang=tgt, texts=texts, callback=callback)
        time.sleep(delay)

    while counter.request_count < num_requests:
        time.sleep(0.1)
        print(f"neg {counter.neg_count}, pos {counter.pos_count} on {num_requests}")

    end_time = time.time()
    print(f"Time taken for {num_requests} requests: {end_time - start_time:.2f} seconds")


def main(model_name):
    client = TritonLlmClient()

    print("Testing concurrent requests...")
    for num_requests in NUM_REQUESTS:
        print(f"\nTesting with {num_requests} requests for model '{model_name}':")
        test_concurrent_requests(client, num_requests, BATCH_SIZE, delay=DELAY, model_name=model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton client for model inference")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use for inference")
    args = parser.parse_args()
    
    main(args.model_name)
