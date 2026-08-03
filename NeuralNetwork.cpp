#include "NeuralNetwork.h"

#include <stdexcept>

using namespace std;

void NeuralNetwork::addLayer(size_t inputNeurons, size_t outputNeurons, ActivationType activation)
{
    layers.emplace_back(inputNeurons, outputNeurons);
    activations.push_back(activation);
}

Matrix NeuralNetwork::forward(const Matrix& X) const
{
    Matrix A = X;

    for (size_t i = 0; i < layers.size(); i++)
    {
        Matrix Z = layers[i].forward(A);

        switch (activations[i])
        {
            case ActivationType::Sigmoid:
                A = Activation::sigmoid(Z);
                break;
            case ActivationType::ReLU:
                A = Activation::relu(Z);
                break;
            case ActivationType::Tanh:
                A = Activation::tanh(Z);
                break;
            default:
                throw invalid_argument("Unsupported activation function.");
        }
    }
    return A;
}