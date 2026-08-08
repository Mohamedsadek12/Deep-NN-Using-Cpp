#!/bin/bash

set -e

echo
echo "==========================="
echo "Building DeepNN..."
echo "==========================="

mkdir -p build

g++ \
    -std=c++17 \
    -Wall \
    -Wextra \
    -Wpedantic \
    -Wshadow \
    -Wformat=2 \
    -Wcast-align \
    -Wconversion \
    -Wsign-conversion \
    -Wnull-dereference \
    -O3 \
    -fopenmp \
    -Iinclude \
    src/*.cpp \
    -o build/DeepNN

echo
echo "==========================="
echo "Build SUCCEEDED!"
echo "Output: build/DeepNN"
echo "==========================="