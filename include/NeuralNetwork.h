#pragma once

#include <vector>

#include "DenseLayer.h"
#include "Activation.h"
#include "ActivationType.h"
#include "Optimizer.h"

using namespace std;

class NeuralNetwork
{
private:

    vector<DenseLayer> layers;
    vector<ActivationType> activations;

    // Optimizer
    Optimizer optimizer;
    bool optimizerSet = false;

    // Helper function to create mini-batches
    Matrix createMiniBatch(const Matrix& data, size_t start, size_t batchSize) const;

public:

    NeuralNetwork() = default;

    void addLayer(size_t inputNeurons, size_t outputNeurons, ActivationType activation);
    
    Matrix forward(const Matrix& X);
    void backward(const Matrix& y_true, const Matrix& y_pred);

    void train(
        const Matrix& X, 
        const Matrix& y_true, 
        size_t epochs, 
        double learningRate, 
        size_t batchSize, 
        bool useLearningRateDecay = false, 
        double decayRate = 0.001
    );
    
        Matrix predict(const Matrix& X);

    void setOptimizer(
        OptimizerType type, 
        double learningRate = 0.01, 
        double beta1 = 0.9, 
        double beta2 = 0.999,
        double epsilon = 1e-8
    );

};
