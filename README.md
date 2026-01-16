Below is a **paper-style experimental README**, written like a **systems / performance evaluation section** you’d see in an academic paper or NVIDIA internal benchmark doc, **with a clear comparison table added**.
You can paste this directly into `README.md`.

---

# Experimental Evaluation of a CUDA-Accelerated Residual MLP on MNIST

## Abstract

This work presents a systematic performance evaluation of a **Residual Multi-Layer Perceptron (ResMLP)** trained on MNIST, implemented **from scratch** in both **CPU (C)** and **GPU (CUDA)**.
We analyze end-to-end training performance and provide a fine-grained breakdown of **forward pass**, **backward pass**, and **optimization steps**, highlighting how residual architectures stress different parts of the compute pipeline across hardware backends.

---

## 1. Model Architecture

The evaluated model follows a lightweight residual design inspired by modern deep learning architectures:

* **Input**: 784-dimensional flattened MNIST image
* **Projection Layer**: Linear (784 → 512) + ReLU
* **Residual Blocks** (×4):

  * Linear (512 → 512) + ReLU
  * Linear (512 → 512)
  * Residual addition + ReLU
* **Classification Head**: Linear (512 → 10)
* **Loss Function**: Softmax + Cross-Entropy
* **Optimizer**: Stochastic Gradient Descent (SGD)

This architecture enables analysis of **residual connections**, **deep backpropagation**, and **memory reuse** without the complexity of convolutional layers.

---

## 2. Implementations

### 2.1 CPU Implementation

* Written in **C**, without external numerical libraries
* Explicit loops for forward, backward, and weight updates
* Used as a **baseline** for performance comparison

### 2.2 CUDA Implementation

* Custom CUDA kernels for:

  * Matrix multiplication
  * Bias addition
  * ReLU forward and backward
  * Residual additions
* Explicit GPU memory management
* Fine-grained timing instrumentation using `clock_gettime`
* Designed to expose **compute vs memory vs control overheads**

---

## 3. Experimental Setup

* **Dataset**: MNIST
* **Training samples**: 10,000
* **Batch size**: 32
* **Epochs**: 10
* **Precision**: FP32
* **Hardware**:

  * CPU: x86-64 (single-threaded)
  * GPU: NVIDIA CUDA-capable GPU

All timings are accumulated across batches and epochs to ensure stable measurements.

---

## 4. Timing Methodology

Each implementation records the following metrics:

* Data loading and transfer time
* Forward pass time:

  * Projection
  * Residual blocks
  * Classification head
  * Softmax
* Loss computation time
* Backward pass time:

  * Softmax gradient
  * Residual blocks
  * Projection
* Weight update time
* Total end-to-end training time

This enables identification of **dominant bottlenecks** at both architectural and hardware levels.

---

## 5. Results

### 5.1 CPU Timing Breakdown

```
Total training time: 50.0 seconds

Forward total:   15.056s (30.1%)
Backward total:  31.091s (62.2%)
Updates:          2.376s ( 4.8%)
```

Backpropagation dominates runtime on CPU, particularly inside the residual blocks.

---

### 5.2 CUDA Timing Breakdown

```
Total training time: 1.6 seconds

Forward total:   0.821s (51.3%)
Backward total:  0.410s (25.6%)
Updates:         0.005s ( 0.3%)
```

On GPU, compute-heavy residual blocks are efficiently parallelized, and optimization overhead becomes negligible.

---

## 6. Performance Comparison

### 6.1 End-to-End Runtime

| Implementation | Total Time (s) | Speedup |
| -------------- | -------------: | ------: |
| CPU (C)        |           50.0 |      1× |
| CUDA (custom)  |            1.6 | **31×** |

---

### 6.2 Phase-wise Comparison

| Phase          | CPU Time (s) | CUDA Time (s) |  Speedup |
| -------------- | -----------: | ------------: | -------: |
| Forward Pass   |        15.06 |          0.82 |      18× |
| Backward Pass  |        31.09 |          0.41 |  **76×** |
| Weight Updates |         2.38 |         0.005 | **476×** |

---

## 7. Analysis

* **Backpropagation is the dominant cost on CPU**, accounting for over 60% of runtime.
* **Residual blocks are the primary compute hotspot** across both implementations.
* CUDA execution shifts the bottleneck from arithmetic to:

  * Memory transfers
  * Softmax and loss computation
* Weight updates are effectively free on GPU due to massive parallelism.
* Residual connections introduce minimal overhead relative to dense layers on GPU.

---

## 8. Key Takeaways

* Custom CUDA kernels achieve a **31× end-to-end training speedup**
* GPU acceleration is especially impactful for **deep residual backpropagation**
* Fine-grained instrumentation is essential for identifying real bottlenecks
* Framework-free implementations provide valuable insight into training dynamics

---

## 9. Reproducibility

### CPU

```bash
gcc -O3 -march=native -ffast-math -o mlp_cpu mlp_cpu.c -lm
./mlp_cpu
```

### CUDA

```bash
nvcc -O2 -o resmlp_cuda resmlp_cuda.cu
./resmlp_cuda
```

Required files:

```
data/X_train.bin
data/y_train.bin
data/X_test.bin
data/y_test.bin
```

---




