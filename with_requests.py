import requests
import pandas as pd

print("variables")
model_name = "sentence-transformers/all-MiniLM-L6-v2"
api_key = "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"

def encode_text(text):
  endpoint_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"
  headers = {"Authorization": f"Bearer {api_key}"}
  payload = {"inputs": text}
  response = requests.post(endpoint_url, headers=headers, json=payload)
  if response.status_code == 200:
    data = response.json()
    return data
  else:
    raise Exception(f"Request failed with status code {response.status_code}")

text1 = "I love llamas"
text2 = "Llamas are awesome"
text3 = "Cats are cute"

embedding1 = encode_text(text1)
embedding2 = encode_text(text2)
embedding3 = encode_text(text3)

print("Embedding for '{}':".format(text1))
print(embedding1)
print("Embedding for '{}':".format(text2))
print(embedding2)
print("Embedding for '{}':".format(text3))
print(embedding3)
