from transformers import pipeline

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token="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")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", token="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")

texts = ["Does this work?"]
print(tokenizer(texts))