#ifndef TYPES_H
#define TYPES_H
#include <iostream>
#include <exception>
#include <string>
#include <algorithm> 
using namespace std;

class Tensor4D {
private:
    int dim[4];
    int stride_N, stride_C, stride_H, stride_W;

    float* data;

    string view_1D(float* vec, int start, int dim);
    string view_2D(float* mat, int start, int dim);
    string view_3D(float* ten, int start, int dim);
    string view_4D(float* hyperten, int start, int dim);
public:
    Tensor4D(int N, int C, int H, int W);
    ~Tensor4D();
    bool CheckDim(int n, int c, int h, int w);
    float& operator()(int n, int c, int h, int w);
    void SetZero();
    void SetOnes();
    string view();
    int* shape();
    float* value();
};

#endif //TYPES_H