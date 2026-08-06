#pragma once

#include <vector>
#include <iostream>
#include <functional>

using namespace std;

class Matrix
{
private:
    size_t rows_;
    size_t cols_;
    vector<double> data_;
    
public:
    // Constructors
    Matrix();
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, double value);

    // Getters
    size_t rows() const;
    size_t cols() const;

    // Element Access
    double& operator()(size_t row, size_t col);
    const double& operator()(size_t row, size_t col) const;

    // Factory functions
    static Matrix zeros(size_t rows, size_t cols);
    static Matrix ones(size_t rows, size_t cols);
    static Matrix random(size_t rows, size_t cols);

    // Basic operations
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(double scalar) const;

    Matrix transpose() const;
    
    // Matrix Multiplication
    Matrix matmul(const Matrix& other) const;
    Matrix matmulSerial(const Matrix& other) const;
    Matrix matmulOpenMP(const Matrix& other) const;
    
    double sum() const;
    Matrix hadamard(const Matrix& other) const;
    Matrix apply(const std::function<double(double)>& func) const;

    void print() const;
};
