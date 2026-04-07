# 🏎️ GPU Inference Optimization — Microsecond-Scale Systems Engineering
### *High-Performance Mixture-of-Experts (MoE) & MXFP4 Quantization Pipelines*

<p align="center">
  <img src="https://img.shields.io/badge/Performance-Top_Leaderboard-D4AF37?style=flat-square&logo=target" alt="Elite Performance">
  <img src="https://img.shields.io/badge/Latency-24.39_µs-C0C0C0?style=flat-square&logo=speedtest" alt="Ultra Low Latency">
  <img src="https://img.shields.io/badge/Hardware-AMD_Instinct_MI300X-ED1C24?style=flat-square&logo=amd" alt="Hardware">
  <img src="https://img.shields.io/badge/Architecture-GFX942-blue?style=flat-square" alt="Architecture">
  <img src="https://img.shields.io/badge/Compute-Triton_%2F_CUDA-4E4E4E?style=flat-square&logo=nvidia" alt="Compute Stack">
</p>

---

## 🎖️ Engineering Achievement: AMD GPU MODE Hackathon
This repository documents the optimization logic and architectural decisions that secured a **Top Leaderboard Finish** in the **AMD GPU MODE Hackathon (March 2026)**. By optimizing for the **AMD Instinct™ MI300X** (GFX942) architecture, the following milestones were reached:

* **Ultra-Low Latency Execution:** Achieved an **MXFP4 GEMM kernel** latency of **~24.39 µs**.
* **The 0.003µs Margin:** Final optimization was within **0.003 µs** of the global 1st place position, representing the absolute frontier of kernel-level instruction tuning.
* **Systems Mastery:** Successfully navigated the **ROCm 6.x** stack to achieve near-theoretical maximum throughput on 4-bit microscaling formats.

---

## 🧩 The Core Challenge: Breaking the Inference Wall
Modern Large Language Models (LLMs) are strictly "Memory Bound." As parameter counts scale toward the trillion-mark, the bottleneck shifts from raw compute (TFLOPS) to memory bandwidth (HBM throughput). This project attacks the "Inference Wall" via two primary vectors:

### 1. Microscaling Formats (MXFP4)
By packing 4-bit weights with specialized block-level scaling factors, we effectively quadruple the available memory bandwidth. The challenge lies in performing dequantization without adding a latency penalty that outweighs the bandwidth gains.

### 2. Sparse Execution (MoE)
Mixture-of-Experts (MoE) architectures allow for massive model capacity with lower active-compute costs. However, the routing and dispatching of tokens to different "experts" can introduce significant micro-latencies if not handled with hardware-aware orchestration.

---

## 🧪 Deep Dive: Optimization Engineering Log
Achieving performance at the microsecond scale requires moving beyond high-level abstractions to **Hardware-Aware System Design**.

### ⚡ Kernel-Level Tuning (MXFP4 GEMM)
Standard kernels often fail to account for the overhead of unpacking 4-bit data. My approach fused the unpacking, scaling, and accumulation logic into a single high-performance loop:
* **LDS (Shared Memory) Management:** Maximized Local Data Store usage to cache weight blocks, drastically reducing Global Memory (VRAM) round-trips.
* **Vectorized Memory Access:** Utilized 128-bit wide loads to fully saturate the MI300X memory bus.
* **Instruction Scheduling:** Manually tuned the inner accumulation loops to minimize branch misprediction and maximize Instruction-Level Parallelism (ILP).

### 🧠 MoE Routing & Dispatch logic
In sparse architectures, the router must be asynchronous and high-speed:
* **Manual Orchestration:** Built a custom dispatch pipeline that executes routing logic in parallel with expert computation.
* **CU Saturation:** Implemented strategies to ensure all 304 Compute Units (CUs) on the MI300X remain occupied, preventing GPU starvation during expert transitions.

### 🔋 Memory Layout & Alignment
* **HBM3 Coalescing:** Structured tensor layouts to ensure every memory request triggers a high-speed burst-mode access.
* **Zero-Copy Pipelines:** Eliminated Python-level latencies by utilizing contiguous tensor management and function caching, avoiding expensive `memcpy` operations.

---

## 📊 Technical Benchmarks
The pipeline is evaluated based on logical correctness and microsecond-scale execution efficiency.

| Metric | Achievement |
| :--- | :--- |
| **MXFP4 GEMM Latency** | **~24.39 µs** |
| **Leaderboard Margin** | **0.003 µs from Rank 1** |
| **Quantization Format** | 4-bit Microscaled (MXFP4) |
| **Target Hardware** | AMD Instinct™ MI300X |

---

## 🏛️ Architectural Significance
This project is a technical blueprint for **Production-grade AI Infrastructure**. The techniques implemented here—such as fused dequantization and hardware-aligned memory layouts—are the foundational building blocks for high-throughput inference engines like **vLLM** and **DeepSeek**. Mastering the microsecond is the key to enabling real-time, large-scale AI serving.

---

## 📂 Repository Structure
```plaintext
gpu-inference-optimization/
├── kernels/             # Optimized Triton & CUDA/HIP implementations
│   ├── mxfp4_gemm.py    # The core achievement: ~24.39 µs kernel
│   └── fused_dequant.cu # C++ CUDA/HIP fused logic for unpacking
├── moe/                 # Expert routing & dispatch pipelines
├── quant/               # MXFP4 quantization & scaling logic
├── benchmarks/          # Precision, bit-correctness, and latency scripts
├── utils/               # Memory alignment & tensor management
└── README.md            # Technical Dashboard
