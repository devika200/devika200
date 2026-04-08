<!-- header -->
<div align="center">

```
  ·  D E V I K A  S A N J E E V  ·
  ───────────────────────────────
  ml engineer · builder
```

![](https://img.shields.io/badge/ML-obsessed-4A90D9?style=flat-square)
![](https://img.shields.io/badge/RAG-systems-2ECC71?style=flat-square)
![](https://img.shields.io/badge/PyTorch-internals-E74C3C?style=flat-square)
![](https://img.shields.io/badge/open_to-opportunities-F39C12?style=flat-square)

<div align="center">

*I don't stop at "it works." I stop at "I understand why it works."*

</div>
---

## Who I Am

CS graduate from Model Engineering College, Kochi (2025). Based in Alappuzha, Kerala.

Spent the last year building ML systems outside of coursework — RAG pipelines, distributed training, sequential models. Genuinely interested in how cognition and ML intersect. Looking for my first role in ML/AI.

---

## What I've Built

### 🧠 HiveMind — Hybrid RAG over 10,000 arXiv Papers
> *The project that made me fall in love with retrieval.*

The problem: dense encoders trained on general text fail on rare domain-specific terminology. Semantic search drowns out the exact terms that matter most.

My solution: an **IDF-aware query router** that dynamically shifts sparse/dense weights based on query type. The router calculates z-score normalized IDF scores across query terms — detecting whether a query is keyword-heavy, conceptual, temporal, or hybrid — then adjusts the BM25/dense balance accordingly.

Then layered **Voyage rerank-2** on top to catch what retrieval missed.

**Result:** Recall@5 jumped from `0.254 → 0.501` across 6 evaluated configurations using the BEIR framework.

```
Stack: Endee hybrid search (HNSW + BM25) · Voyage AI embeddings · 
       Voyage rerank-2 · Groq LLaMA-3.3-70b · Streamlit
Eval:  BEIR framework · 6 configs · ~10k arXiv papers (2019–2024)
```

🔗 [github.com/devika200/endee](https://github.com/devika200/endee)

---

### ⚡ FastFashion-MoE — Distributed Mixture of Experts in PyTorch RPC
> *The project that broke me, then rebuilt me.*

I wanted to understand distributed training at the process level. Not abstractions — actual RPC calls, actual gradient flow across machines.

Built a Mixture of Experts where the router lives on the master process and experts run on remote workers via `rpc.remote()`. Training uses `dist_autograd.context()` + `DistributedOptimizer` so gradients flow across process boundaries.

Then hit a wall.

**The bug:** Tensors pickled across RPC processes were silently severing the autograd graph. Experts ran. No errors. No gradients. Just frozen weights and a loss that never moved.

No stack trace. No warning. Just wrong behavior.

I read PyTorch internals for days. Traced the pickling mechanism. Found exactly where autograd context was being dropped. Fixed it in 3 lines.

That bug taught me more about how ML systems actually work than any course ever could.

```
Stack: PyTorch RPC (TensorPipe) · dist_autograd · DistributedOptimizer
       rpc.remote() · rpc_sync() · FashionMNIST
Env:   Linux / WSL2 / Colab
```

🔗 [github.com/devika200/FastFashion-MOE-Parallel--RPC](https://github.com/devika200/FastFashion-MOE-Parallel--RPC)

---

### 🔐 UPISecure — Sequential Fraud Detection
> *Where I learned that classical ML wins when the problem fits.*

Built a fraud detection system on UPI transaction sequences. The temptation was to reach for a transformer — everyone's doing it. But the data was structured, the dataset was limited, and the output needed to be explainable to a compliance team.

So I chose deliberately: **AR-HMM (3 lags) + CRF ensemble**. The HMM models hidden behavioral states across transaction sequences. The CRF precisely labels boundaries. Ensemble score averages both probabilities.

3-class output: `Normal / Suspicious / Fraud`.

Classical sequence models. Right tool for the right problem.

```
Stack: hmmlearn · sklearn-crfsuite · Flask REST API · JWT auth
       MongoDB · React 18 · Vite · Netlify + Render
```

🔗 [github.com/devika200/UPI-Secure](https://github.com/devika200/UPI-Secure)

---

## How I Think

```python
while not_solved:
    read_internals()
    question_assumptions()
    stare_at_logs()
    if clarity:
        write_3_lines()
        break
```

Most of my best work happens before I touch code. I've spent days staring at retrieval eval results and gradient flow logs — not because I was stuck, but because I was *thinking*. Understanding what the system is actually doing versus what I think it's doing. That gap is where the real work lives.

---

## What I'm Obsessed With

- Retrieval systems and how they fail at scale
- The gap between research benchmarks and production behavior
- Why classical ML still quietly powers most of what matters
- Distributed training internals — not the abstractions, the actual mechanics
- How cognition and ML intersect (genuinely, not as a talking point)

---

## Stack

```
Languages  →  Python · SQL · JavaScript
ML/DL      →  PyTorch · scikit-learn · hmmlearn · sklearn-crfsuite
RAG/LLM    →  LangChain · Voyage AI · Groq · Streamlit · FAISS · Qdrant
Backend    →  Flask · FastAPI · MongoDB · JWT · REST APIs
Frontend   →  React 18 · Vite · Netlify
Eval       →  BEIR framework · Recall@K · custom eval pipelines
```

---

## Find Me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-devika--sanjeev-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/devika-sanjeev)
[![GitHub](https://img.shields.io/badge/GitHub-devika200-181717?style=flat&logo=github)](https://github.com/devika200)
[![Resume](https://img.shields.io/badge/Resume-Devika_resume-4A90D9?style=flat&logo=adobeacrobatreader)](https://github.com/devika200/devika200/blob/main/Devika_resume.pdf)

📍 Alappuzha, Kerala, India &nbsp;·&nbsp; 📧 devikasanjeev.mec@gmail.com
---

*"The fix was 3 lines. Finding it took everything."*
