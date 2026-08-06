#pragma once

#include "Matrix.h"

namespace Loss
{
    // Loss functions
    double binaryCrossEntropy(const Matrix& y_true, const Matrix& y_pred);
    double categoricalCrossEntropy(const Matrix& y_true, const Matrix& y_pred);
    double meanSquaredError(const Matrix& y_true, const Matrix& y_pred);
    double meanAbsoluteError(const Matrix& y_true, const Matrix& y_pred);

    // Loss gradients
    Matrix binaryCrossEntropyGradient(const Matrix& y_true, const Matrix& y_pred);
    Matrix meanSquaredErrorGradient(const Matrix& y_true, const Matrix& y_pred);
    Matrix meanAbsoluteErrorGradient(const Matrix& y_true, const Matrix& y_pred);
}
