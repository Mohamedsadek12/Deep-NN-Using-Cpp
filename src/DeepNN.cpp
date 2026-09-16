#include <iostream>
#include <chrono>

#include "NeuralNetwork.h"
#include "Matrix.h"

using namespace std;
using namespace std::chrono;


// Test one optimizer
void testOptimizer(const string& optimizerName, OptimizerType optimizerType, double learningRate,
    const Matrix& X, const Matrix& y, size_t epochs)
{
    cout << "\n========================================\n";
    cout << "Testing: " << optimizerName << "\n";
    cout << "========================================\n";

    NeuralNetwork nn;

    nn.addLayer(2, 4, ActivationType::Tanh);
    nn.addLayer(4, 1, ActivationType::Sigmoid);
    
    nn.setOptimizer(optimizerType, learningRate);

    auto start = high_resolution_clock::now();
    nn.train(X, y, epochs, learningRate);
    auto end = high_resolution_clock::now();
    double trainingTime = duration<double>(end - start).count();

    Matrix predictions = nn.predict(X);

    size_t correct = 0;


    for (size_t i = 0; i < y.cols(); ++i)
    {
        if (predictions(0, i) == y(0, i))
        {
            ++correct;
        }
    }


    double accuracy = 100.0 * static_cast<double>(correct) / static_cast<double>(y.cols());

    cout << "\nOptimizer: " << optimizerName << "\n";

    cout << "Training Time: " << trainingTime << " seconds\n";


    cout << "Predictions: ";
    for (size_t i = 0; i < predictions.cols(); ++i)
    {
        cout << static_cast<int>(predictions(0, i)) << " ";
    }

    cout << "\nAccuracy: " << accuracy << "%\n";
}


int main()
{
    // XOR
    Matrix X(2, 4);


    X(0, 0) = 0.0;
    X(1, 0) = 0.0;

    X(0, 1) = 0.0;
    X(1, 1) = 1.0;

    X(0, 2) = 1.0;
    X(1, 2) = 0.0;

    X(0, 3) = 1.0;
    X(1, 3) = 1.0;


    // Target values
    Matrix y(1, 4);

    y(0, 0) = 0.0;
    y(0, 1) = 1.0;
    y(0, 2) = 1.0;
    y(0, 3) = 0.0;


    const size_t epochs = 10000;

    testOptimizer("SGD", OptimizerType::SGD, 0.1, X, y, epochs);
    testOptimizer("Momentum", OptimizerType::Momentum, 0.1, X, y, epochs);
    testOptimizer("Adam", OptimizerType::Adam, 0.01, X, y, epochs);

    cout << "\n========================================\n";
    cout << "All optimizers tested successfully.\n";
    cout << "========================================\n";

    return 0;
}
