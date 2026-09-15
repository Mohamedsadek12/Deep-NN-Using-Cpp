#pragma once

#include "Matrix.h"
#include <vector>
#include <cstddef>

enum class OptimizerType
{
    SGD,
    Momentum,
    Adam
};


class Optimizer
{
private:

    OptimizerType type;

    // Learning rate
    double learningRate;

    // Momentum / Adam parameters
    double beta1;
    double beta2;
    double epsilon;
    size_t timestep;

    /*
        Optimizer states for each layer.

        For Momentum:
            velocityW[layer]
            velocityB[layer]

        For Adam:
            momentW[layer]
            momentB[layer]

            secondMomentW[layer]
            secondMomentB[layer]
    */

    std::vector<Matrix> velocityW;
    std::vector<Matrix> velocityB;

    std::vector<Matrix> momentW;
    std::vector<Matrix> momentB;

    std::vector<Matrix> secondMomentW;
    std::vector<Matrix> secondMomentB;

public:

    Optimizer(
        OptimizerType type = OptimizerType::SGD,
        double learningRate = 0.01,
        double beta1 = 0.9,
        double beta2 = 0.999,
        double epsilon = 1e-8
    );


    /*
        Update the parameters of one layer.

        layerIndex: index of the layer being updated
        W, b: weights and biases
        dW, db: gradients of weights and biases
    */
    void update(size_t layerIndex, Matrix& W, Matrix& b, const Matrix& dW, const Matrix& db);

    // Reset all optimizer states
    void reset();

    // Learning-rate control
    void setLearningRate(double learningRate);

    double getLearningRate() const;

    // Used by NeuralNetwork to advance the optimizer
    void startStep();

    size_t getTimestep() const;
};