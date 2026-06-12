# vLLM on Spyre (Granite Model) 🚀

## 📌 Overview

This project demonstrates running **vLLM on IBM Spyre backend** using offline inference.

The goal was to:

- Run vLLM on Spyre 
- Perform offline inference 
- Use Granite model 
- Enforce single-core execution 
- Validate Tensor Parallelism (TP=2) or TP=1

---

## ⚙️ Environment Setup

The following environment variables were used to enforce **single-core execution**:

bash
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_WORKER_MULTIPROCESSING_GPU_ID=0
export OMP_NUM_THREADS=1
