#include <iostream>
#include <chrono>

#include "NeuralNetwork.h"
#include "Matrix.h"

using namespace std;
using namespace std::chrono;

int main()
{
    // XOR dataset
    Matrix X(2, 4);

    X(0, 0) = 0.0;
    X(1, 0) = 0.0;

    X(0, 1) = 0.0;
    X(1, 1) = 1.0;

    X(0, 2) = 1.0;
    X(1, 2) = 0.0;

    X(0, 3) = 1.0;
    X(1, 3) = 1.0;

    Matrix y(1, 4);

    y(0, 0) = 0.0;
    y(0, 1) = 1.0;
    y(0, 2) = 1.0;
    y(0, 3) = 0.0;

    // Same architecture as your previous test
    NeuralNetwork nn;

    nn.addLayer(2, 4, ActivationType::Tanh);
    nn.addLayer(4, 1, ActivationType::Sigmoid);

    const size_t epochs = 10000;
    const double learningRate = 0.1;

    // -----------------------------
    // Training
    // -----------------------------

    auto start = high_resolution_clock::now();

    nn.train(X, y, epochs, learningRate);

    auto end = high_resolution_clock::now();

    double trainingTime =
        duration<double>(end - start).count();

    // -----------------------------
    // Prediction
    // -----------------------------

    Matrix predictions = nn.predict(X);

    size_t correct = 0;

    for (size_t i = 0; i < y.cols(); ++i)
    {
        if (predictions(0, i) == y(0, i))
        {
            ++correct;
        }
    }

    double accuracy =
        100.0 * static_cast<double>(correct) /
        static_cast<double>(y.cols());

    // -----------------------------
    // Results
    // -----------------------------

    cout << "\n===========================\n";
    cout << "C++ Neural Network\n";
    cout << "===========================\n";

    cout << "Training Time: "
         << trainingTime
         << " seconds\n";

    cout << "\nPredictions:\n";

    for (size_t i = 0; i < predictions.cols(); ++i)
    {
        cout << static_cast<int>(predictions(0, i)) << " ";
    }

    cout << "\n\nAccuracy: "
         << accuracy
         << "%\n";

    return 0;
}
