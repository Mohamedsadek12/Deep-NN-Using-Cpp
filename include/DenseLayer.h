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
    Matrix A;

    // backward pass variables
    Matrix dW;
    Matrix db;

public:
    DenseLayer(size_t prevLayerSize, size_t currentLayerSize);

    // Forward (Linear only)
    Matrix forward(const Matrix& input);

    // Backward
    Matrix backward(const Matrix& dZ);

    
    // Mutable access for Optimizer
    Matrix& getWeights();
    Matrix& getBias();

    // Getters
    const Matrix& getWeights() const;
    const Matrix& getBias() const;
    const Matrix& getZ() const;
    const Matrix& getActivation() const;
    const Matrix& getWeightGradient() const;
    const Matrix& getBiasGradient() const;

    void setActivation(const Matrix& activation);
};
