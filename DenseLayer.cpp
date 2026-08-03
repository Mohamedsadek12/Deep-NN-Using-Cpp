#include "DenseLayer.h"

// Constructor
DenseLayer::DenseLayer(size_t prevLayerSize, size_t currentLayerSize): 
    W(Matrix::random(currentLayerSize, prevLayerSize)), // W dimensions: rows = currentLayerSize, cols = prevLayerSize
    b(Matrix::zeros(currentLayerSize, 1)) {}


Matrix DenseLayer::forward(const Matrix& A_prev) const
{
    Matrix Z = W.matmul(A_prev);

    for (size_t i = 0; i < Z.rows(); i++)
    {
        for (size_t j = 0; j < Z.cols(); j++)
        {
            Z(i, j) += b(i, 0);
        }
    }
    return Z;
}

// Getters
const Matrix& DenseLayer::getWeights() const
{
    return W;
}

const Matrix& DenseLayer::getBias() const
{
    return b;
}