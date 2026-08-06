#include "Activation.h"

#include <cmath>
#include <algorithm>

using namespace std;

namespace Activation
{
    Matrix sigmoid(const Matrix& Z)
    {
        Matrix result(Z.rows(), Z.cols());

        for (size_t i = 0; i < Z.rows(); i++)
        {
            for (size_t j = 0; j < Z.cols(); j++)
            {
                result(i, j) = 1.0 / (1.0 + exp(-Z(i, j)));
            }
        }
        return result;
    }

    Matrix relu(const Matrix& Z)
    {
        Matrix result(Z.rows(), Z.cols());

        for (size_t i = 0; i < Z.rows(); i++)
        {
            for (size_t j = 0; j < Z.cols(); j++)
            {
                result(i, j) = max(0.0, Z(i, j));
            }
        }
        return result;
    }

    Matrix tanh(const Matrix& Z)
    {
        Matrix result(Z.rows(), Z.cols());

        for (size_t i = 0; i < Z.rows(); i++)
        {
            for (size_t j = 0; j < Z.cols(); j++)
            {
                result(i, j) = std::tanh(Z(i, j));
            }
        }
        return result;
    }

    Matrix sigmoidDerivative(const Matrix& Z)
    {
        Matrix S = sigmoid(Z);
        Matrix result(Z.rows(), Z.cols());

        for (size_t i = 0; i < Z.rows(); i++)
        {
            for (size_t j = 0; j < Z.cols(); j++)
            {
                result(i, j) = S(i, j) * (1.0 - S(i, j));
            }
        }
        return result;
    }

    Matrix reluDerivative(const Matrix& Z)
    {
        Matrix result(Z.rows(), Z.cols());

        for (size_t i = 0; i < Z.rows(); i++)
        {
            for (size_t j = 0; j < Z.cols(); j++)
            {
                result(i, j) = (Z(i, j) > 0.0) ? 1.0 : 0.0;
            }
        }
        return result;
    }

    Matrix tanhDerivative(const Matrix& Z)
    {
        Matrix result(Z.rows(), Z.cols());

        for (size_t i = 0; i < Z.rows(); i++)
        {
            for (size_t j = 0; j < Z.cols(); j++)
            {
                double T = std::tanh(Z(i, j));
                result(i, j) = 1.0 - T * T;
            }
        }
        return result;
    }
}
