@echo off
setlocal

REM ==========================================
REM Deep Neural Network - C++ Build Script
REM ==========================================

echo.
echo ===========================
echo Building DeepNN...
echo ===========================

REM ==========================================
REM Create build directory
REM ==========================================

if not exist "build" (
    mkdir "build"
)

REM ==========================================
REM Output executable
REM ==========================================

set "OUTPUT=build\DeepNN.exe"

REM ==========================================
REM Compile
REM ==========================================

g++ ^
-std=c++17 ^
-Wall ^
-Wextra ^
-Wpedantic ^
-Wshadow ^
-Wformat=2 ^
-Wcast-align ^
-Wconversion ^
-Wsign-conversion ^
-Wnull-dereference ^
-O3 ^
-fopenmp ^
-Iinclude ^
src\*.cpp ^
-o "%OUTPUT%"

REM ==========================================
REM Check compilation result
REM ==========================================

if %errorlevel% neq 0 (
    echo.
    echo ===========================
    echo Build FAILED!
    echo ===========================
    exit /b %errorlevel%
)

REM ==========================================
REM Build successful
REM ==========================================

echo.
echo ===========================
echo Build SUCCEEDED!
echo ===========================
echo Output: %OUTPUT%
echo ===========================

echo.
echo To run the program:
echo %OUTPUT%

endlocal
