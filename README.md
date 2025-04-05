# 🚀 Mini Multimodal Benchmark Demo

> 📡 Proof-of-concept for my DeepMind GSoC 2025 proposal: *Gemini Ascendant*

## 📦 Overview

This project benchmarks multimodal inference using Hugging Face Transformers.  
It compares **baseline vs. optimized (batched)** inference to show how we can accelerate **Gemini 2.0** models.

# Dataset

- **50 text prompts** — e.g., general knowledge, factual Q&A  
- **50 image-text prompts** — simulated vision-language tasks  
- Stored in `data/text_samples.csv` and `data/image_samples.csv`

 Setup

```bash
git clone https://github.com/your-username/mini-benchmark-demo.git
cd mini-benchmark-demo
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python benchmark.py
