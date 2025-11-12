# 🦙 Llama-3.2-1B QLoRA Fine-tuning (PEFT + TRL)

Fine-tuning **Llama-3.2-1B** on **Databricks Dolly-15k** using **QLoRA** and **PEFT**.  
This notebook demonstrates low-VRAM fine-tuning with 4-bit quantization and LoRA adapters.

---

## 🚀 Overview

Large models like Llama are powerful but heavy to train.  
**QLoRA (Quantized LoRA)** lets us fine-tune models efficiently on smaller GPUs by:
- Loading models in **4-bit** quantization (NF4)
- Training **LoRA adapters** on selected layers
- Using **PEFT** + **TRL** for streamlined fine-tuning

---

## 📘 Notebook

All code is inside: notebooks/finetune_qlora_llama32_1b_dolly15k.ipynb

This notebook walks through the complete fine-tuning workflow:
1. **Dataset loading** — Uses `databricks/databricks-dolly-15k` from Hugging Face Datasets.  
2. **Preprocessing** — Formats samples into `Instruction → Response` pairs (and includes context when available).  
3. **Model preparation** — Loads `meta-llama/Llama-3.2-1B` in 4-bit (NF4) mode using `bitsandbytes`.  
4. **PEFT configuration** — Applies LoRA adapters (`r=8, alpha=16, dropout=0.05`) to attention and MLP layers.  
5. **Training** — Runs supervised fine-tuning with TRL’s `SFTTrainer` for one epoch.  
6. **Saving & merging** — Exports both the adapter and a merged version of the model.  
7. **Inference** — Demonstrates generation examples using base, adapter-loaded, and merged models.

> 💡 The notebook is fully self-contained — you can open it in Jupyter or Google Colab and run it cell by cell.
> Small Note: During this work, it was ran on Kaggle.

---

## ⚙️ Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

## 🧠 Training Configuration

| Parameter | Value |
|------------|--------|
| **Base model** | `meta-llama/Llama-3.2-1B` |
| **Dataset** | `databricks/databricks-dolly-15k` |
| **LoRA rank (r)** | 8 |
| **LoRA alpha** | 16 |
| **LoRA dropout** | 0.05 |
| **Quantization** | 4-bit NF4 |
| **Batch size** | 2 |
| **Gradient accumulation** | 8 |
| **Learning rate** | 2e-4 |
| **Epochs** | 1 |

