#include "NeuralNetwork.h"
#include "Loss.h"

#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <random>

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

Matrix NeuralNetwork::createMiniBatch(const Matrix& data, size_t start, size_t batchSize) const
{
    size_t end = min(start + batchSize, data.cols());
    size_t actualBatchSize = end - start;

    Matrix batch(data.rows(), actualBatchSize);

    for (size_t i = 0; i < data.rows(); ++i)
    {
        for (size_t j = 0; j < actualBatchSize; ++j)
        {
            batch(i, j) = data(i, start + j);
        }
    }
    return batch;
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

void NeuralNetwork::train(const Matrix& X, const Matrix& y_true, size_t epochs, double learningRate, size_t batchSize)
{
    /*
        If the user didn't explicitly select an optimizer,
        use SGD with the supplied learning rate.
    */

    if (X.cols() != y_true.cols())
    {
        throw invalid_argument("Input X and target y must have the same number of samples");
    }

    if (X.cols() == 0)
    {
        throw invalid_argument("Training data cannot be empty");
    }

    if (batchSize == 0)
    {
        throw invalid_argument(
            "Batch size must be greater than 0."
        );
    }

    if (!optimizerSet)
    {
        setOptimizer(OptimizerType::SGD, learningRate);
    }
    else
    {
        optimizer.setLearningRate(learningRate);
    }

    // Random number generator (for shuffling the dataset)
    random_device rd;
    mt19937 gen(rd());

    // Store sample indices
    vector<size_t> indices(X.cols());

    for (size_t i = 0; i < X.cols(); ++i)
    {
        indices[i] = i;
    }

    for (size_t epoch = 0; epoch < epochs; ++epoch)
    {
        // Shuffle sample indices at the beginning of every epoch
        shuffle(indices.begin(), indices.end(), gen);

        double epochLoss = 0.0;
        size_t numberOfBatches = 0;

        // Process the dataset batch by batch
        for (size_t start = 0; start < X.cols(); start += batchSize)
        {
            size_t end = min(start + batchSize, X.cols());
            size_t actualBatchSize = end - start;
            
            // Create mini-batches
            Matrix X_batch = createMiniBatch(X, start, batchSize);
            Matrix y_batch = createMiniBatch(y_true, start, batchSize);


            // Create shuffled mini-batch
            for (size_t i = 0; i < actualBatchSize; ++i)
            {
                size_t originalIndex = indices[start + i];

                // Copy X sample
                for (size_t row = 0; row < X.rows(); ++row)
                {
                    X_batch(row, i) = X(row, originalIndex);
                }

                // Copy corresponding y label
                for (size_t row = 0; row < y_true.rows(); ++row)
                {
                    y_batch(row, i) = y_true(row, originalIndex);
                }
            }


            // Forward propagation
            Matrix y_pred = forward(X_batch);

            // Calculate loss
            double batchLoss = Loss::binaryCrossEntropy(y_batch, y_pred);

            epochLoss += batchLoss;
            ++numberOfBatches;

            // Backward propagation + optimizer update
            backward(y_batch, y_pred);
        }
        
        // Average loss over all mini-batches
        epochLoss /= static_cast<double>(numberOfBatches);

        // Print loss
        if ((epoch + 1) % 1000 == 0)
        {
            cout << "Epoch " << epoch + 1 << "  Loss = " << epochLoss << endl;
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
