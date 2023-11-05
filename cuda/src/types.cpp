#include "types.hpp"
#include <random>
#include "cublas-MatMul2D.cuh"

Tensor4D::Tensor4D(int N, int C, int H, int W) {
    data = new float[N * C * H * W];
    dim[0] = N, dim[1] = C, dim[2] = H, dim[3] = W;
    // Strides
    stride_N = C * H * W;
    stride_C = H * W;
    stride_H = W;
    stride_W = 1;
}

Tensor4D::  ~Tensor4D()
{

}

bool Tensor4D::CheckDim(int n, int c, int h, int w)
{
    return !(n >= this->dim[0] || c >= this->dim[1] || h >= this->dim[2] || w >= this->dim[3]);
}

float& Tensor4D::operator()(int n, int c, int h, int w) {
    try {
        CheckDim(n, c, h, w);
    }
    catch (std::exception& e)
    {
        cout << "exception" << e.what() << endl;
        throw runtime_error(e.what());
    }
    return data[n * stride_N + c * stride_C + h * stride_H + w * stride_W];
}

void Tensor4D::SetZero()
{
    int SIZE = dim[0] * dim[1] * dim[2] * dim[3];
    memset(data, 0, sizeof(float) * SIZE);
}

void Tensor4D::SetOnes()
{
    int SIZE = dim[0] * dim[1] * dim[2] * dim[3];
    std::fill(data, data + SIZE, 1.0f);
}


string Tensor4D::view_1D(float* vec, int start, int dim) {
    string out1D;
    out1D = "[";
    for (int i = 0; i < this->dim[3]; ++i) {
        out1D += to_string(vec[i + start]);
        if (i < this->dim[3] - 1) {
            out1D += ", ";
        }
    }
    out1D += "]";
    return out1D;
}

string Tensor4D::view_2D(float* mat, int start, int dim) {
    string out2D;
    out2D = "[";
    for (int i = 0; i < this->dim[2]; ++i) {
        out2D += view_1D(mat, start + i * this->dim[3], dim);
        if (i < this->dim[2] - 1) {
            out2D += ",\n ";
            if (dim == 4) {
                out2D += "  ";
            }
            else if (dim == 3) {
                out2D += " ";
            }
        }
    }
    out2D += "]";
    return out2D;
}

string Tensor4D::view_3D(float* ten, int start, int dim) {
    string out3D;
    out3D = "[";
    for (int i = 0; i < this->dim[1]; ++i) {
        out3D += view_2D(ten, start + i * this->dim[2] * this->dim[3], dim);
        if (i < this->dim[1] - 1) {
            out3D += ",\n\n ";
            if (dim == 4) {
                out3D += " ";
            }
        }
    }
    out3D += "]";
    return out3D;
}

string Tensor4D::view_4D(float* hyperten, int start, int dim) {
    string out4D;
    out4D = "[";
    for (int i = 0; i < this->dim[0]; ++i) {
        out4D += view_3D(hyperten, start + i * this->dim[1] * this->dim[2] * this->dim[3], dim);
        if (i < this->dim[0] - 1) {
            out4D += ",\n\n\n\n ";
        }
    }
    out4D += "]";
    return out4D;
}

string Tensor4D::view() {
    string out;
    int dim;
    if (this->dim[0] == 1) {
        if (this->dim[1] == 1) {
            if (this->dim[2] == 1) {
                if (this->dim[3] == 1) {
                    dim = 0;
                }
                else {
                    dim = 1;
                }
            }
            else {
                dim = 2;
            }
        }
        else {
            dim = 3;
        }
    }
    else {
        dim = 4;
    }

    switch (dim) {
    case 0:
        out = to_string(this->data[0]);
        break;

    case 1:
        out = view_1D(this->data, 0, dim);
        break;

    case 2:
        out = view_2D(this->data, 0, dim);
        break;

    case 3:
        out = view_3D(this->data, 0, dim);
        break;

    case 4:
        out = view_4D(this->data, 0, dim);
        break;

    default:
        return "Something went wrong!";
    }
    return out;
}

int* Tensor4D::shape() {
    return dim;
}

float* Tensor4D::value() {
    return data;
}

