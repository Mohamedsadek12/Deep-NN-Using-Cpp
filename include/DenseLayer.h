#pragma once

#include "Matrix.h"

class DenseLayer
{
private:

    Matrix W;
    Matrix b;

    // forward pass variables
    Matrix A_prev;
    Matrix Z;

    // backward pass variables
    Matrix dW;
    Matrix db;

public:
    DenseLayer(size_t inputSize, size_t outputSize);

    // Forward (Linear only)
    Matrix forward(const Matrix& input);

    // Backward
    Matrix backward(const Matrix& dZ, double learningRate);

    // Getters
    const Matrix& getWeights() const;
    const Matrix& getBias() const;

    const Matrix& getZ() const;
};
