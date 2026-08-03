#pragma once

#include "Matrix.h"

namespace Loss
{
    // Classification
    double binaryCrossEntropy(const Matrix& y_true, const Matrix& y_pred);
    double categoricalCrossEntropy(const Matrix& y_true, const Matrix& y_pred);

    // Regression
    double meanSquaredError(const Matrix& y_true, const Matrix& y_pred);
    double meanAbsoluteError(const Matrix& y_true, const Matrix& y_pred);
}