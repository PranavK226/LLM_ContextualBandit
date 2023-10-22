import requests
import pandas as pd

print("variables")
model_id = "codellama/CodeLlama-7b-hf"
hf_token = "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"
api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}

print("function def")
def query(texts):
    response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
    return response.json()

texts = ["Does this work?"]

print("running queries")
output = query(texts)

print(output)