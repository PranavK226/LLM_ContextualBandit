from transformers import pipeline

model_id = "codellama/CodeLlama-7b-hf"
transcriber = pipeline(model=model_id, device_map="auto")

texts = ["Does this work?"]

output = transcriber(texts)
print(output)