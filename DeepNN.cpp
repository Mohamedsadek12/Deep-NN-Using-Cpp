#include <iostream>
#include <chrono>
#include "Matrix.h"

using namespace std;
using namespace std::chrono;

void benchmarkMatmul()
{
    const size_t N = 1000;

    Matrix A = Matrix::random(N, N);
    Matrix B = Matrix::random(N, N);

    auto start = high_resolution_clock::now();

    Matrix C = A.matmulOpenMP(B);

    auto end = high_resolution_clock::now();

    cout << "OpenMP Time: "
        << duration<double>(end - start).count()
        << " s\n";
}

int main()
{
    benchmarkMatmul();
    return 0;
}