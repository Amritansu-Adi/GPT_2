# GPT2-FineWeb-Edu

An attempt to **recreate (and possibly surpass)** the original GPT-2 model by OpenAI, using **PyTorch** and **modern training practices**. This project leverages the [📚 FineWeb-Edu dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) — a high-quality web corpus curated specifically for educational and scientific text.

---

## 🚀 Overview

This project focuses on building a **GPT-2-style transformer-based language model** from scratch and training it on large-scale, clean, web-based text data. The ultimate goal is to build a performant, open-source alternative to GPT-2, suitable for **scientific research**, **education**, and **custom pretraining tasks**.

---

## 📦 Dataset

- **Dataset:** [FineWeb-Edu (HuggingFace)](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)  
- **Type:** Pre-filtered, high-quality educational corpus  
- **Size:** ~10B tokens

The dataset is downloaded, tokenized, and split into shard files for efficient training.

---
