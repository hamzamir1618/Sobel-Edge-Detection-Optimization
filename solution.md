

```markdown
# Optimizing GPU-Accelerated Sobel Edge Detection

**Based on "Implementation of Sobel filter using CUDA" by Akshita Akasapu et al.**

**22i-0775** M. Basim Kafeel  
**22i-1003** Hamza Ahmad Mir

---

## Paper Overview

### Objective
Implement Sobel filter for edge detection on **CPU (serial)** and **GPU (parallel)**.

### Context
**Target:** high-volume imagery (satellite/remote sensing) where serial processing bottlenecks appear.

---

## The Problem

### Challenge
Processing large, high-resolution imagery (e.g., satellite TIFF) is computationally intensive.

### Bottleneck
Serial C execution struggles with throughput as image dimensions increase, causing delays.

### Opportunity
Edge detection is **embarrassingly parallel**—pixel computations are independent and suitable for GPU acceleration.

---

## Core Computational Idea: Sobel Filter

The operator detects **horizontal (Gx)** and **vertical (Gy)** edges using a **3×3 neighborhood**. It computes gradient magnitude per pixel:

> **|G| = √(Gx² + Gy²)**

Convolution isolates contours and reduces data dimensionality.

---

## Parallelization Strategy

### CUDA Implementation
Map pixel operations across GPU cores for concurrent processing.

### Thread Hierarchy
Authors use **8×8 thread blocks** (64 threads/block) to partition the image.

### Data Handling
Performance compared across formats (**PGM**, **TIFF**) to assess throughput differences.

---

## Key Results

### Serial
**Lena (1024×1024)** ≈ **0.119 s** (Static serial approach).

### Parallel (reported)
GPU on **Lena (512×512)** took **0.334 s** — slower than CPU for that size.

### Authors' Conclusion
GPUs outperform for larger, high-resolution data, though the **512×512 slowdown** is unexplained.

---

## Critical Analysis — Why This Paper

Provides a solid baseline for data decomposition and memory-access patterns in CUDA but leaves **optimization** and **anomalies** unaddressed.

### Naive Memory Usage
Relies on **global memory** (high-latency) instead of exploiting **shared memory**.

### Suboptimal Threading
**8×8 block size** is used without justification or tuning for occupancy.

### Unexplained Anomaly
Does not analyze why GPU was slower for **512×512**; data-transfer overheads are not profiled.

---

## Proposed Improvement — Shared Memory Tiling

Introduce **shared-memory tiling**: each block loads a tile including **halo** (3×3 neighborhoods) into on-chip shared memory to improve cache locality and eliminate redundant global memory reads.

### Implementation
Load overlapping tiles to serve all threads in a block, minimizing redundant loads.

### Goal
Reduce global memory traffic and latency to increase throughput on large images.

---

## Optimization Plan & Analysis

- **Block Experiments**  
  Test **16×16** and **32×32** blocks to find optimal occupancy across GPUs.

- **Profile Transfers**  
  Measure Host-to-Device overhead to explain **512×512** slowdown and reduce transfer costs.

- **Metrics**  
  Report **Speedup** and **Efficiency** across resolutions up to **4K**.

---

## Expected Outcomes

- **Performance**  
  Substantial reduction in execution time versus the naive CUDA implementation.

- **Insights**  
  Clear separation of memory-bounded vs. compute-bounded regimes in image processing.

- **Validation**  
  Robust, optimized Sobel tool capable of handling satellite-scale imagery efficiently.

**Next steps:** implement tiled kernel, run block-size sweep, profile transfers, and publish comparative benchmarks.

---

**Thank you**
```

This Markdown version preserves the structure, headings, and key points from all slides in the presentation. Let me know if you want any modifications (e.g., adding image descriptions, different heading levels, or tables).