<div align="center">

# 🚀 Fine-Tuning LLMs & SFT Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.32-yellow.svg)](https://huggingface.co/docs/transformers/index)
[![PEFT](https://img.shields.io/badge/PEFT-0.5.0-green.svg)](https://github.com/huggingface/peft)
[![DeepSpeed](https://img.shields.io/badge/DeepSpeed-Enabled-red.svg)](https://www.deepspeed.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**An end-to-end, production-grade framework for Pre-Training, Supervised Fine-Tuning (SFT), Parameter-Efficient Fine-Tuning (PEFT), and Knowledge Distillation of Large Language Models.**

[Features](#-key-features) •
[Tech Stack](#-tech-stack) •
[Installation](#-installation) •
[Usage](#-usage) •
[Architecture](#-architecture)

</div>

---

## 📖 Overview

This repository houses a comprehensive suite of tools and pipelines designed for the lifecycle of modern Large Language Models (LLMs). From preparing massive datasets for pre-training to fine-tuning state-of-the-art models using SFT and PEFT techniques, this codebase demonstrates advanced proficiency in LLM engineering. It also includes cutting-edge modules for **Knowledge Distillation** and **Synthetic Data Generation** using OpenAI APIs.

Whether you are looking to train a model from scratch, adapt Llama-2/Mistral to a specific domain, or compress a giant model into a deployable edge device format, this repository provides the blueprint.

## ✨ Key Features

### 1. 🏗️ Pre-Training & Data Engineering
- **Efficient Sharding**: custom logic to handle terabyte-scale datasets using HuggingFace `datasets`.
- **Advanced Tokenization**: Scripts to train and utilize `WordPiece` and `BPE` tokenizers (e.g., BERT-base uncased).
- **Data Packaging**: Optimized routines to pack data for causal language modeling objectives.

### 2. 🎯 Supervised Fine-Tuning (SFT)
- **Robust Training Loops**: Leveraging `trl` (Transformer Reinforcement Learning) library for SFTTrainer.
- **Custom Chat Templates**: Support for formatting datasets into conversational formats (ShareGPT, Alpaca, etc.).
- **Experiment Tracking**: Integrated with **Weights & Biases (WandB)** for real-time loss logging and metric visualization.

### 3. ⚡ Parameter-Efficient Fine-Tuning (PEFT)
- **LoRA & QLoRA**: Implementation of Low-Rank Adapters for efficient fine-tuning on consumer hardware.
- **Quantization**: 4-bit and 8-bit training support via `bitsandbytes`.
- **Memory Optimization**: DeepSpeed ZeRO stage integration for distributed training.

### 4. 🧪 Distillation & Synthetic Data
- **Knowledge Distillation**: Pipelines to train smaller "student" models to mimic larger "teacher" models.
- **Synthetic Data Generation**: Automated pipelines using `OpenAI` and `pydantic` to generate high-quality, diverse instruction-tuning datasets.

## 🛠️ Tech Stack

- **Core Frameworks**: `PyTorch 2.0+`, `HuggingFace Transformers`, `Accelerate`
- **Optimization**: `DeepSpeed`, `BitsAndBytes` (QLoRA)
- **Fine-Tuning**: `PEFT`, `TRL`
- **DataOps**: `Datasets`, `DeepLake`, `OpenAI API`
- **Monitoring**: `WandB` (Weights & Biases)

## 🚀 Installation

System requirements: Linux/MacOS with NVIDIA GPU (CUDA 11.8+ recommended).

1. **Clone the repository**
   ```bash
   git clone https://github.com/santhoshkavi123/FineTuningLLMs-SFT.git
   cd FineTuningLLMs-SFT
   ```

2. **Set up the environment**
   We recommend using `uv` or `conda` for dependency management.
   ```bash
   # Using pip
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**
   Create a `.env` file for your API keys:
   ```env
   OPENAI_API_KEY=your_key_here
   WANDB_API_KEY=your_key_here
   ```

## 💻 Usage

### Pre-Training Data Preparation
Prepare your raw text data for training:
```bash
# Navigate to PreTraining directory
cd PreTraining
# Run the packaging notebook or script
jupyter notebook NTBK_PackagingDataForPretraining.ipynb
```

### Running Supervised Fine-Tuning
Launch a fine-tuning job with `accelerate`:
```bash
accelerate launch main.py \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --dataset_path "./data/processed" \
    --use_peft \
    --lora_r 8 \
    --precision "bf16"
```

### Synthetic Data Pipeline
Generate new instruction datasets:
```bash
cd SyntheticDataPipeline
jupyter notebook DataPipeline.ipynb
```

## 🔮 Future Roadmap

- [ ] **RLHF / DPO**: Implementation of Direct Preference Optimization.
- [ ] **Model Merging**: Scripts for SLERP and Linear/Task Arithmetic merging of adapters.
- [ ] **Evaluation Harness**: Integration with `lm-evaluation-harness` for automated benchmarks.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
