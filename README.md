# Mini Multimodal Benchmark Demo

Proof-of-concept for my GSoC 2025 DeepMind proposal.

## Overview
- **Dataset**: 200 samples (100 text, 100 image-text prompts).  
- **Model**: `google/flan-t5-small` (text-only; images simulated).  
- **Goal**: Compare baseline vs. batched latency.  
- **Results**: Reduced latency from 0.0152 s/sample to 0.0011 s/sample (92.60% improvement).

## Setup
1. Clone: `git clone [your-repo-url]`
2. Virtual env: `python -m venv venv`
3. Activate: `source venv/bin/activate` (Mac/Linux)
4. Install: `pip install -r requirements.txt`
5. Run: `python benchmark.py`

## Results
- **Baseline Latency**: 0.0152 s/sample  
- **Optimized Latency**: 0.0011 s/sample  
- **Improvement**: 92.60%  
*(Demonstrates massive potential for Gemini 2.0 optimization.)*

## Files
- `data/text_samples.csv`: 100 text prompts.  
- `data/image_samples.csv`: 100 image-text prompts.  
- `benchmark.py`: Script with batch_size=20.  
- `results.txt`: Output.

## Next Steps
Fuels my GSoC goal to optimize Gemini 2.0 for 50%+ faster inference on a 5,000-sample benchmark.

## Author
Ravi Raja