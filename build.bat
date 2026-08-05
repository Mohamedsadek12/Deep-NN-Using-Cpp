@echo off
setlocal

REM ==========================================
REM Create build directory if it doesn't exist
REM ==========================================
if not exist "build\Debug" (
    mkdir "build\Debug"
)

REM ==========================================
REM Output executable
REM ==========================================
set "OUTPUT=build\Debug\outDebug.exe"

echo.
echo ===========================
echo Building DeepNN...
echo ===========================

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
*.cpp ^
-o "%OUTPUT%"

if %errorlevel% neq 0 (
    echo.
    echo ===========================
    echo Build FAILED!
    echo ===========================
    exit /b %errorlevel%
)

echo.
echo ===========================
echo Build SUCCEEDED!
echo Output: %OUTPUT%
echo ===========================

endlocal