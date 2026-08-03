#pragma once

#include "Matrix.h"

class DenseLayer
{
private:

    Matrix W;
    Matrix b;

public:
    DenseLayer(size_t inputSize, size_t outputSize);

    // Forward (Linear only)
    Matrix forward(const Matrix& A_prev) const;

    // Getters
    const Matrix& getWeights() const;
    const Matrix& getBias() const;
};