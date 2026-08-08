# Deep Neural Network Using C++

A neural network framework implemented from scratch in **C++17**, with a custom matrix library, forward propagation, backpropagation, multiple activation and loss functions, and **OpenMP-based parallelization** for performance optimization.

The main goal of this project is to understand how neural networks work internally while exploring how low-level C++ optimizations can improve computational performance.

---

## Features

- Custom `Matrix` class implemented from scratch
- Matrix addition, subtraction, scalar multiplication
- Hadamard product
- Matrix transpose
- Matrix multiplication
- Serial matrix multiplication
- OpenMP parallel matrix multiplication
- Sigmoid activation
- ReLU activation
- Tanh activation
- Activation derivatives
- Binary Cross-Entropy loss
- Categorical Cross-Entropy loss
- Mean Squared Error
- Mean Absolute Error
- Backpropagation
- Gradient descent
- Neural network training
- Binary classification
- Prediction and accuracy calculation
- Performance benchmarking
- C++ vs Python/NumPy comparison

---

## Project Structure

```text
Deep-NN-Using-Cpp/
│
├── include/
│   ├── Activation.h
│   ├── DenseLayer.h
│   ├── Loss.h
│   ├── Matrix.h
│   └── NeuralNetwork.h
│
├── src/
│   ├── Activation.cpp
│   ├── DenseLayer.cpp
│   ├── Loss.cpp
│   ├── Matrix.cpp
│   ├── NeuralNetwork.cpp
│   └── DeepNN.cpp
│
├── build/
│   └── DeepNN
│
├── Test.py
├── build.bat
└── README.md
