from transformers import AutoTokenizer, AutoModel

print("getting model")
model = AutoModel.from_pretrained("codellama/CodeLlama-34b-Python-hf")

print("getting tokenizer")
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-34b-Python-hf")

print("tokenizing prompt")
prompt = "Checking if it works."
input_ids = tokenizer.encode(prompt, return_tensors="pt")
attention_mask = tokenizer.get_attention_mask(input_ids)

print("making embeddings")
input_embeddings = model.embeddings(input_ids, attention_mask)

print(input_embeddings)