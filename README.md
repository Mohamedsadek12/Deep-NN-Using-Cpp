<div align="center">

# 🧠 Deep Neural Network Using C++

**A from-scratch Deep Neural Network built in C++17 — no ML frameworks, just linear algebra, calculus, and raw performance engineering.**

Built to explore how neural networks actually work under the hood, and to measure how far low-level optimization and CPU parallelization (OpenMP) can push a hand-written implementation against Python/NumPy.

![C++](https://img.shields.io/badge/C%2B%2B-17-blue?logo=cplusplus)
![OpenMP](https://img.shields.io/badge/Parallelism-OpenMP-orange)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey)

</div>

---

## 📚 Table of Contents

- [Project Highlights](#-project-highlights)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Build & Run](#-build--run)
- [Matrix Benchmark](#-matrix-multiplication-benchmark)
- [Neural Network Performance](#️-neural-network-training-performance)
- [C++ vs Python / NumPy](#-c-vs-python--numpy)
- [Activation Functions](#-activation-functions)
- [Loss Functions](#-loss-functions)
- [Key Takeaways](#-key-takeaways)
- [Roadmap](#-roadmap)
- [License](#-license)

---

## 🚀 Project Highlights

| Feature | Description |
|---|---|
| 🔢 Custom Matrix Library | Hand-written matrix engine (allocation, transpose, multiplication, element-wise ops) |
| 🧬 Neural Network Core | Fully connected layers, forward propagation, backpropagation |
| ⚡ OpenMP Parallelization | Multi-threaded matrix multiplication for major CPU speedups |
| 📉 Gradient Descent | Custom training loop with configurable learning rate |
| 🎛️ Multiple Activations | Sigmoid, ReLU, Tanh |
| 🎯 Multiple Loss Functions | Binary Cross-Entropy, Categorical Cross-Entropy, MSE, MAE |
| ✅ XOR Classification Test | Classic non-linearly-separable benchmark problem |
| 📊 Benchmarking Suite | Serial vs. optimized vs. OpenMP vs. Python/NumPy comparisons |
| 🧱 Clean Architecture | Strict header/source separation, GCC warnings + optimization flags enabled |

---

## 📁 Project Structure

```text
Deep-NN-Using-Cpp/
│
├── include/                 # Public headers
│   ├── Activation.h
│   ├── ActivationType.h
│   ├── DenseLayer.h
│   ├── Loss.h
│   ├── Matrix.h
│   └── NeuralNetwork.h
│
├── src/                      # Implementation files
│   ├── Activation.cpp
│   ├── DeepNN.cpp            # Entry point
│   ├── DenseLayer.cpp
│   ├── Loss.cpp
│   ├── Matrix.cpp
│   └── NeuralNetwork.cpp
│
├── build.bat                 # Windows build script
├── Test.py                   # Python/NumPy comparison benchmark
├── README.md
└── .gitignore
```

---

## 🔧 Requirements

- **C++17**-compatible compiler (GCC / G++ recommended)
- **OpenMP** support (bundled with most modern GCC installs)
- **Windows** or **Linux**
- *(Optional, for benchmarking)* Python 3 with NumPy — used by `Test.py` for the C++ vs. Python comparison

---
## 🛠 Build & Run

### Windows
```bash
build.bat
```

### Linux / macOS
```bash
chmod +x build.sh
./build.sh
./build/DeepNN
```

`build.sh` compiles with `-O3 -fopenmp` plus a strict warning set (`-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion -Wnull-dereference`, etc.) and drops the binary at `build/DeepNN`.

### Python Benchmark (optional comparison)
```bash
python Test.py
```

> 💡 The `-fopenmp` flag is required to enable multi-threaded matrix operations. Omitting it will silently fall back to single-threaded execution.
---

## 📊 Matrix Multiplication Benchmark

Three implementations of matrix multiplication were benchmarked to isolate the impact of algorithmic and parallel optimization:

| Implementation | Time (s) | Speedup vs. Original |
|---|---|---|
| Original (Transpose + Matmul) | 0.0960183 | 1.00× |
| Optimized Serial | 0.0992386 | 0.97× |
| **Optimized OpenMP** | **0.0219274** | **4.38×** |

**Result:** OpenMP parallelization delivered a **4.38× speedup** over the naive implementation — the single biggest performance win in the project.

---

## ⏱️ Neural Network Training Performance

| Stage | Training Time (s) | Speedup |
|---|---|---|
| Before Optimization | 1.524650 | 1.00× |
| **After Optimization** | **0.450281** | **3.39×** |

Applying the optimized/parallelized matrix operations throughout the training loop (forward pass, backpropagation, and gradient updates) cut end-to-end training time by **~70%**.

---

## 🐍 C++ vs Python / NumPy

| Implementation | Training Time (s) | Accuracy |
|---|---|---|
| Python / NumPy | 0.025144 | 100% |
| Optimized C++ (this project) | 0.450281 | 100% |

Both implementations converge to **100% accuracy** on the XOR classification task. NumPy remains faster on small workloads thanks to its highly optimized, decades-mature BLAS backend — but this comparison highlights *why*: it quantifies the real-world gap between a hand-rolled C++ engine and a production-grade, vectorized numerical library, and shows exactly how much of that gap parallelization alone can close.

---

## 🎛 Activation Functions

- **Sigmoid** — smooth, bounded output in (0, 1); classic choice for binary classification
- **ReLU** — fast, sparse activation; mitigates vanishing gradients in deeper networks
- **Tanh** — zero-centered, bounded output in (-1, 1)

---

## 📉 Loss Functions

- **Binary Cross-Entropy** — binary classification
- **Categorical Cross-Entropy** — multi-class classification
- **Mean Squared Error (MSE)** — regression
- **Mean Absolute Error (MAE)** — regression, robust to outliers

---

## 🔑 Key Takeaways

- Hand-implementing forward/backward propagation builds a much deeper understanding of how gradients actually flow through a network than using a framework does.
- Naive "optimized serial" code isn't always faster than the naive baseline — real gains came from **parallelism**, not just cleaner loops.
- OpenMP is a low-effort, high-impact tool for CPU-bound numerical workloads: a few pragmas produced a >4× speedup.
- Even a well-optimized C++ implementation has a long way to go to match NumPy's BLAS-backed performance — highlighting how much engineering goes into production numerical libraries.

---

## 🗺 Roadmap

- [ ] Add SIMD (AVX/SSE) vectorization on top of OpenMP
- [ ] Support convolutional layers (CNN)
- [ ] Add Adam / RMSProp optimizers
- [ ] Multi-threaded mini-batch training
- [ ] GPU backend (CUDA) for large-scale benchmarking
- [ ] Unit tests (Catch2 / GoogleTest)

---

<div align="center">

**⭐ If you found this project interesting, consider starring the repo!**

</div>
