import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import time
from PIL import Image

# Load model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

def run_baseline(data, is_image=False):
    latencies = []
    results = []

    for _, row in data.iterrows():
        input_text = row["text"]

        start = time.time()
        inputs = tokenizer(input_text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        end = time.time()

        latencies.append(end - start)
        results.append("OK")

    avg_latency = sum(latencies) / len(latencies)
    return avg_latency, results

def run_optimized(data, batch_size=20, is_image=False):
    latencies = []
    results = []

    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i+batch_size]
        input_texts = batch["text"].tolist()

        start = time.time()
        inputs = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        end = time.time()

        batch_latency = (end - start) / len(batch)
        latencies.extend([batch_latency] * len(batch))
        results.extend(["OK"] * len(batch))

    avg_latency = sum(latencies) / len(latencies)
    return avg_latency, results

# Load Data
text_data = pd.read_csv("data/text_samples.csv")
image_data = pd.read_csv("data/image_samples.csv")

# TEXT BENCHMARKING
print("Running text baseline...")
text_base_latency, text_base_results = run_baseline(text_data)

print("Running text optimized...")
text_opt_latency, text_opt_results = run_optimized(text_data, batch_size=20)

# IMAGE BENCHMARKING (simulated as text prompts)
print("Running image baseline...")
img_base_latency, img_base_results = run_baseline(image_data)

print("Running image optimized...")
img_opt_latency, img_opt_results = run_optimized(image_data, batch_size=20)

# Combine results
avg_base_latency = (text_base_latency + img_base_latency) / 2
avg_opt_latency = (text_opt_latency + img_opt_latency) / 2
improvement = 100 * (avg_base_latency - avg_opt_latency) / avg_base_latency

# Print and save results
print(f"Baseline Latency: {avg_base_latency:.4f} s/sample")
print(f"Optimized Latency: {avg_opt_latency:.4f} s/sample")
print(f"Improvement: {improvement:.2f}%")

with open("results.txt", "w") as f:
    f.write(f"Baseline Latency: {avg_base_latency:.4f} s/sample\n")
    f.write(f"Optimized Latency: {avg_opt_latency:.4f} s/sample\n")
    f.write(f"Improvement: {improvement:.2f}%\n")
