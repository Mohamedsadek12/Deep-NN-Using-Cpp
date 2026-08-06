#include "Loss.h"

#include <cmath>
#include <stdexcept>

using namespace std;

namespace Loss
{
    double binaryCrossEntropy(const Matrix& y_true, const Matrix& y_pred)
    {
        if (y_true.rows() != y_pred.rows() || y_true.cols() != y_pred.cols())
        {
            throw invalid_argument("Dimensions of y_true and y_pred must match.");
        }

        constexpr double epsilon = 1e-15;
        size_t m = y_true.cols();
        double loss = 0.0;

        for(size_t i = 0; i < y_true.rows(); ++i)
        {
            for(size_t j = 0; j < y_true.cols(); ++j)
            {
                double p = y_pred(i, j);
                double y = y_true(i, j);

                // avoid log(0)
                if (p < epsilon) p = epsilon; 
                if (p > (1.0 - epsilon)) p = 1.0 - epsilon;

                loss += (y * log(p)) + ((1.0 - y) * log(1.0 - p));
            }
        }
        
        double J = -loss / static_cast<double>(m);
        return J;
    }

    double categoricalCrossEntropy(const Matrix& y_true, const Matrix& y_pred)
    {
        if (y_true.rows() != y_pred.rows() || y_true.cols() != y_pred.cols())
        {
            throw invalid_argument("Dimensions of y_true and y_pred must match.");
        }

        constexpr double epsilon = 1e-15;
        size_t m = y_true.cols();
        double loss = 0.0;

        for(size_t i = 0; i < y_true.cols(); ++i) // iterate over each example
        {
            for(size_t j = 0; j < y_true.rows(); ++j) // iterate over each class
            {
                double p = y_pred(j, i);

                // avoid log(0)
                if (p < epsilon) p = epsilon;

                loss += y_true(j, i) * log(p);
            }
        }
        double J = -loss / static_cast<double>(m);
        return J;
    }

    double meanSquaredError(const Matrix& y_true, const Matrix& y_pred)
    {
        if (y_true.rows() != y_pred.rows() || y_true.cols() != y_pred.cols())
        {
            throw invalid_argument("Dimensions of y_true and y_pred must match.");
        }
        double loss = 0.0;
        size_t n = y_true.cols() * y_true.rows();

        for(size_t i = 0; i < y_true.rows(); ++i)
        {
            for(size_t j = 0; j < y_true.cols(); ++j)
            {
                double diff = y_true(i, j) - y_pred(i, j);
                loss += diff * diff;
            }
        }
        double J = loss / static_cast<double>(n);
        return J;
    }

    double meanAbsoluteError(const Matrix& y_true, const Matrix& y_pred)
    {
        if (y_true.rows() != y_pred.rows() || y_true.cols() != y_pred.cols())
        {
            throw invalid_argument("Dimensions of y_true and y_pred must match.");
        }
        double loss = 0.0;
        size_t n = y_true.cols() * y_true.rows();

        for(size_t i = 0; i < y_true.rows(); ++i)
        {
            for(size_t j = 0; j < y_true.cols(); ++j)
            {
                double diff = y_true(i, j) - y_pred(i, j);
                loss += abs(diff);
            }
        }
        double J = loss / static_cast<double>(n);
        return J;
    }


    Matrix binaryCrossEntropyGradient(const Matrix& y_true, const Matrix& y_pred)
    {
        if (y_true.rows() != y_pred.rows() || y_true.cols() != y_pred.cols())
        {
            throw invalid_argument("Dimensions of y_true and y_pred must match.");
        }

        constexpr double epsilon = 1e-15;

        Matrix dA(y_true.rows(), y_true.cols());

        for(size_t i = 0; i < y_true.rows(); ++i)
        {
            for(size_t j = 0; j < y_true.cols(); ++j)
            {
                double y = y_true(i, j);
                double p = y_pred(i, j);
                
                // avoid division by zero
                if (p < epsilon) p = epsilon; 
                if (p > (1.0 - epsilon)) p = 1.0 - epsilon;

                dA(i, j) = -(y / p) + ((1.0 - y) / (1.0 - p));
            }
        }

        return dA;
    }


    Matrix meanSquaredErrorGradient(const Matrix& y_true, const Matrix& y_pred)
    {
        Matrix dA(y_true.rows(), y_true.cols());

        size_t n = y_true.cols() * y_true.rows();

        for(size_t i = 0; i < y_true.rows(); ++i)
        {
            for(size_t j = 0; j < y_true.cols(); ++j)
            {
                dA(i, j) = (2.0 / static_cast<double>(n)) * (y_pred(i, j) - y_true(i, j));
            }
        }

        return dA;
    }

    Matrix meanAbsoluteErrorGradient(const Matrix& y_true, const Matrix& y_pred)
    {
        Matrix dA(y_true.rows(), y_true.cols());

        size_t n = y_true.cols() * y_true.rows();

        for(size_t i = 0; i < y_true.rows(); ++i)
        {
            for(size_t j = 0; j < y_true.cols(); ++j)
            {
                double diff = y_pred(i, j) - y_true(i, j);
                if(diff > 0)
                {
                    dA(i, j) = 1.0 / static_cast<double>(n);
                }
                else if(diff < 0)
                {
                    dA(i, j) = -1.0 / static_cast<double>(n);
                }
                else
                {
                    dA(i, j) = 0.0;
                }
            }
        }
        return dA;
    }
}
