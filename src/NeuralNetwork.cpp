#include "NeuralNetwork.h"
#include "Loss.h"

#include <stdexcept>
#include <iostream>

using namespace std;

void NeuralNetwork::addLayer(size_t inputNeurons, size_t outputNeurons, ActivationType activation)
{
    layers.emplace_back(inputNeurons, outputNeurons);
    activations.push_back(activation);
}


void NeuralNetwork::setOptimizer(OptimizerType type, double learningRate, double beta1, double beta2, double epsilon)
{
    optimizer = Optimizer(type, learningRate, beta1, beta2, epsilon);
    optimizerSet = true;
}


Matrix NeuralNetwork::forward(const Matrix& X)
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
        // Store activation for backward propagation
        layers[i].setActivation(A);
    }   
    return A;
}

void NeuralNetwork::backward(const Matrix& y_true, const Matrix& y_pred)
{
    if (layers.empty())
    {
        return;
    }

    Matrix dA = y_pred - y_true;

    optimizer.startStep(); // timestep is incremented ONCE per training step, not once per layer.


    for (size_t i = layers.size(); i-- > 0;)
    {
        Matrix dZ;

        // Output layer
        if (i == layers.size() - 1)
        {
            dZ = dA;
        }
        // Hidden layers
        else
        {
            switch (activations[i])
            {
                case ActivationType::Sigmoid:
                    dZ = dA.hadamard(Activation::sigmoidDerivative(layers[i].getZ()));
                    break;
                case ActivationType::ReLU:
                    dZ = dA.hadamard(Activation::reluDerivative(layers[i].getZ()));
                    break;
                case ActivationType::Tanh:
                    dZ = dA.hadamard(Activation::tanhDerivative(layers[i].getZ()));
                    break;
                default:
                    throw invalid_argument("Unsupported activation function.");
            }
        }
        // Calculate gradients
        dA = layers[i].backward(dZ);

        // Update parameters
        optimizer.update(i, layers[i].getWeights(), layers[i].getBias(), layers[i].getWeightGradient(), layers[i].getBiasGradient());
    }
}

void NeuralNetwork::train(const Matrix& X, const Matrix& y_true, size_t epochs, double learningRate)
{
    /*
        If the user didn't explicitly select an optimizer,
        use SGD with the supplied learning rate.
    */

    if (!optimizerSet)
    {
        setOptimizer(OptimizerType::SGD, learningRate);
    }
    else
    {
        optimizer.setLearningRate(learningRate);
    }

    for (size_t epoch = 0; epoch < epochs; ++epoch)
    {
        // Forward propagation
        Matrix y_pred = forward(X);
        
        // Calculate loss
        double loss = Loss::binaryCrossEntropy(y_true, y_pred);

        // Backward propagation + optimizer update
        backward(y_true, y_pred);

        // Print loss
        if ((epoch + 1) % 1000 == 0)
        {
            cout << "Epoch " << epoch + 1 << "  Loss = " << loss << endl;
        }
    }
}

Matrix NeuralNetwork::predict(const Matrix& X)
{
    Matrix probabilities = forward(X); // Get the probabilities from the forward pass

    Matrix predictions(probabilities.rows(), probabilities.cols());

    for (size_t i = 0; i < probabilities.rows(); ++i)
    {
        for (size_t j = 0; j < probabilities.cols(); ++j)
        {
            if (probabilities(i, j) >= 0.5)
            {
                predictions(i, j) = 1.0; // Class 1
            }
            else
            {
                predictions(i, j) = 0.0; // Class 0
            }
        }
    }
    return predictions;
}
