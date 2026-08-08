#include "Matrix.h"
#include <stdexcept>
#include <iostream>
#include <random>
#include <functional>
#include <omp.h>

using namespace std;

// default constructor
Matrix::Matrix(): rows_(0), cols_(0) {}

// init the Matrix without value
Matrix::Matrix(size_t rows, size_t cols): rows_(rows), cols_(cols), data_(rows * cols) {}

// init the Matrix with value
Matrix::Matrix(size_t rows, size_t cols, double value): rows_(rows), cols_(cols), data_(rows * cols, value) {}

// Getters
size_t Matrix::rows() const
{
    return rows_;
}

size_t Matrix::cols() const
{
    return cols_;
}

// Element Access
double& Matrix::operator()(size_t row, size_t col)
{
    return data_[row * cols_ + col]; // to get the index in the 1D vector representation of the matrix
}

const double& Matrix::operator()(size_t row, size_t col) const
{
    return data_[row * cols_ + col];
}


// Factory Functions
Matrix Matrix::zeros(size_t rows, size_t cols)
{
    return Matrix(rows, cols, 0.0);
}

Matrix Matrix::ones(size_t rows, size_t cols)
{
    return Matrix(rows, cols, 1.0);
}

Matrix Matrix::random(size_t rows, size_t cols)
{
    Matrix result(rows, cols);

    static mt19937 gen(std::random_device{}()); // Mersenne Twister random number generator

    uniform_real_distribution<double> dist(-1.0, 1.0); // define the range

    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            result(i, j) = dist(gen);
        }
    }

    return result;
}

// Operations
Matrix Matrix::operator+(const Matrix& other) const
{
    if (rows_ != other.rows_ || cols_ != other.cols_)
        throw invalid_argument("Matrix dimensions must match.");

    Matrix result(rows_, cols_);

    const size_t size = data_.size();
    for (size_t i = 0; i < size; ++i)
    {
        result.data_[i] = data_[i] + other.data_[i];
    }

    return result;
}

Matrix Matrix::operator-(const Matrix& other) const
{
    if (rows_ != other.rows_ || cols_ != other.cols_)
        throw std::invalid_argument("Matrix dimensions must match.");

    Matrix result(rows_, cols_);

    const size_t size = data_.size();
    for (size_t i = 0; i < size; ++i)
    {
        result.data_[i] = data_[i] - other.data_[i];
    }

    return result;
}

Matrix Matrix::operator*(double scalar) const
{
    Matrix result(rows_, cols_);

    const size_t size = data_.size();
    for (size_t i = 0; i < size; ++i)
    {

        result.data_[i] = data_[i] * scalar;

    }

    return result;
}

// multiply each element of the matrix with the corresponding element of the other matrix
Matrix Matrix::hadamard(const Matrix& other) const 
{
    if (rows_ != other.rows_ || cols_ != other.cols_)
    {
        throw invalid_argument(
            "Hadamard product error: Matrix dimensions must match."
        );
    }

    Matrix result(rows_, cols_);

    const size_t size = data_.size();
    for (size_t i = 0; i < size; ++i)
    {

        result.data_[i] = data_[i] * other.data_[i];

    }

    return result;
}


Matrix Matrix::transpose() const
{
    Matrix result(cols_, rows_);

    for (size_t i = 0; i < rows_; i++)
    {
        for (size_t j = 0; j < cols_; j++)
        {
            result(j, i) = (*this)(i, j);
        }
    }

    return result;
}


// Matrix Multiplication

Matrix Matrix::matmulSerial(const Matrix& other) const
{
    if (cols_ != other.rows_)
    {
        throw invalid_argument(
            "Matrix multiplication error: Number of columns of the first matrix "
            "must equal the number of rows of the second matrix."
        );
    }
    
    Matrix result(rows_, other.cols_, 0.0);

    for (size_t i = 0; i < rows_; ++i)
    {
        for (size_t j = 0; j < cols_; ++j)
        {
            const double a = data_[i * cols_ + j];

            for (size_t k = 0; k < other.cols_; ++k)
            {
                result.data_[i * other.cols_ + k] += a * other.data_[j * other.cols_ + k];
            }
        }
    }

    return result;
}

Matrix Matrix::matmul(const Matrix& other) const
{
    return matmulSerial(other);
}

Matrix Matrix::matmulOpenMP(const Matrix& other) const
{
    if (cols_ != other.rows_)
    {
        throw invalid_argument(
            "Matrix multiplication error: Number of columns of the first matrix "
            "must equal the number of rows of the second matrix."
        );
    }

    Matrix result(rows_, other.cols_, 0.0);

    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(rows_); ++i)
    {
        const size_t row = static_cast<size_t>(i);
        for (size_t j = 0; j < cols_; ++j)
        {
            const double a = data_[row * cols_ + j];

            for (size_t k = 0; k < other.cols_; ++k)
            {
                result.data_[row * other.cols_ + k] += a * other.data_[j * other.cols_ + k];
            }
        }
    }

    return result;
}

Matrix Matrix::matmulTranspose(const Matrix& other) const
{
    // this = W
    // other = dZ
    // Calculate: W^T × dZ

    if (rows_ != other.rows_)
    {
        throw invalid_argument(
            "Matrix multiplication error: Number of rows of the first matrix "
            "must equal the number of rows of the second matrix."
        );
    }

    Matrix result(cols_, other.cols_, 0.0);

    for (size_t i = 0; i < cols_; ++i)
    {
        for (size_t j = 0; j < rows_; ++j)
        {
            const double a = data_[j * cols_ + i];
            for (size_t k = 0; k < other.cols_; ++k)
            {
                result.data_[i * other.cols_ + k] += a * other.data_[j * other.cols_ + k];            
            }
        }
    }

    return result;
}

Matrix Matrix::matmulTransposeOpenMP(const Matrix& other) const
{
    if (rows_ != other.rows_)
    {
        throw invalid_argument(
            "Matrix multiplication error: Number of rows of the first matrix "
            "must equal the number of rows of the second matrix."
        );
    }

    Matrix result(cols_, other.cols_, 0.0);

    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(cols_); ++i)
    {
        const size_t col = static_cast<size_t>(i);

        for (size_t j = 0; j < rows_; ++j)
        {
            const double a = data_[j * cols_ + col];

            for (size_t k = 0; k < other.cols_; ++k)
            {
                result.data_[col * other.cols_ + k] += a * other.data_[j * other.cols_ + k];
            }
        }
    }

    return result;
}



double Matrix::sum() const
{
    double total = 0.0;

    for (double value : data_)
    {
        total += value;
    }

    return total;
}

// Instead of rewriting the loops every time, write them once.
Matrix Matrix::apply(const function<double(double)>& func) const
{
    Matrix result(rows_, cols_);

    const size_t size = data_.size();
    for (size_t i = 0; i < size; ++i)
    {
        result.data_[i] = func(data_[i]);
    }

    return result;
}


void Matrix::print() const
{
    for (size_t i = 0; i < rows_; i++)
    {
        for (size_t j = 0; j < cols_; j++)
        {
            cout << (*this)(i, j) << " ";
        }

        cout << '\n';
    }
}
