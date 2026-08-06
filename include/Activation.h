#pragma once

#include "Matrix.h"

namespace Activation
{
    Matrix sigmoid(const Matrix& Z);
    Matrix relu(const Matrix& Z);
    Matrix tanh(const Matrix& Z);

    Matrix sigmoidDerivative(const Matrix& Z);
    Matrix reluDerivative(const Matrix& Z);
    Matrix tanhDerivative(const Matrix& Z);
}
