#pragma once

#include <vector>

#include "DenseLayer.h"
#include "Activation.h"
#include "ActivationType.h"

using namespace std;

class NeuralNetwork
{
private:

    vector<DenseLayer> layers;
    vector<ActivationType> activations;

public:

    NeuralNetwork() = default;

    void addLayer(size_t inputNeurons, size_t outputNeurons, ActivationType activation);
    
    Matrix forward(const Matrix& X);
    void backward(const Matrix& y_true, const Matrix& y_pred, double learningRate);

    void train(const Matrix& X, const Matrix& y_true, size_t epochs, double learningRate);

    Matrix predict(const Matrix& X);
};
