#pragma once
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define IN
#define OUT
#define INOUT

int MatMul2D(OUT float* C, IN const float* A, const float* B, int M, int N, int K);