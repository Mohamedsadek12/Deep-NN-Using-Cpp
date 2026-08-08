# Deep Neural Network Using C++

A **from-scratch Deep Neural Network implementation in C++17**, built to explore the internal mechanics of neural networks and investigate the performance benefits of low-level optimization and CPU parallelization.

The project includes a custom matrix library, neural network architecture, forward and backward propagation, gradient-based training, multiple activation and loss functions, and **OpenMP-accelerated matrix operations**.

A major focus of the project is comparing a custom C++ implementation against Python/NumPy and measuring the impact of optimization techniques.

---

## 🚀 Project Highlights

- Neural Network implemented from scratch in C++
- Custom Matrix library
- OpenMP parallel matrix multiplication
- Forward propagation
- Backpropagation
- Gradient descent optimization
- Multiple activation functions
- Multiple loss functions
- XOR classification test
- Performance benchmarking
- Python/NumPy comparison
- Clean separation between headers and source files
- GCC warning and optimization flags
- Measured performance improvements

---

## 📁 Project Structure

Deep-NN-Using-Cpp/
├── include/
│   ├── Activation.h
│   ├── ActivationType.h
│   ├── DenseLayer.h
│   ├── Loss.h
│   ├── Matrix.h
│   └── NeuralNetwork.h
│
├── src/
│   ├── Activation.cpp
│   ├── DeepNN.cpp
│   ├── DenseLayer.cpp
│   ├── Loss.cpp
│   ├── Matrix.cpp
│   └── NeuralNetwork.cpp
│
├── build.bat
├── README.md
├── Test.py
└── .gitignore

