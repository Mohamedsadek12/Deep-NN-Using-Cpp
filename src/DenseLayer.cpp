#include "DenseLayer.h"

// Constructor
DenseLayer::DenseLayer(size_t prevLayerSize, size_t currentLayerSize): 
    W(Matrix::random(currentLayerSize, prevLayerSize)), // W dimensions: rows = currentLayerSize, cols = prevLayerSize
    b(Matrix::zeros(currentLayerSize, 1)) {}


Matrix DenseLayer::forward(const Matrix& input)
{
    A_prev = input;

    Z = W.matmul(A_prev);

    for (size_t i = 0; i < Z.rows(); i++)
    {
        for (size_t j = 0; j < Z.cols(); j++)
        {
            Z(i, j) += b(i,0);
        }
    }

    return Z;
}


Matrix DenseLayer::backward(const Matrix& dZ, double learningRate)
{
    size_t m = A_prev.cols(); // number of examples

    dW = dZ.matmul(A_prev.transpose()) * (1.0 / static_cast<double>(m));
    db = Matrix::zeros(dZ.rows(), 1);

    for (size_t i = 0; i < dZ.rows(); i++)
    {
        double sum = 0.0;
        for (size_t j = 0; j < dZ.cols(); j++)
        {
            sum += dZ(i, j);
        }
        db(i, 0) = sum / static_cast<double>(m);
    }

    Matrix dA_prev = W.transpose().matmul(dZ);

    // Gradient descent update
    W = W - dW * learningRate;
    b = b - db * learningRate;
    
    return dA_prev;
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

const Matrix& DenseLayer::getZ() const
{
    return Z;
}

void DenseLayer::setActivation(const Matrix& activation)
{
    A = activation;
}

const Matrix& DenseLayer::getActivation() const
{
    return A;
}
