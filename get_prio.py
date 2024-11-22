from locomotive_llm.utils import TritonLlmClient


client = TritonLlmClient()
print(client.list_models())
