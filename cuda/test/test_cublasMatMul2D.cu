#include "types.hpp"
#include "cublas-MatMul2D.cuh"
#include <random>

Tensor4D MulcuBlas(Tensor4D tensor1, Tensor4D tensor2)
{
    int* DimT1 = tensor1.shape();
    int* DimT2 = tensor2.shape();
    //check dimension N, C equal 
    try
    {
        if (DimT1[3] != DimT2[2])
        {
            throw "An error occurred";
        }
    }
    catch (const char* msg) {
        msg = " Error! Tensor1 dim[3] != Tensor2 dim[2]!!!";
        std::cerr << msg << std::endl;
    }
    try
    {
        if ((DimT1[0] != DimT2[0]) || (DimT1[1] != DimT2[1]))
        {
            throw "An error occurred";
        }
    }
    catch (const char* msg) {
        msg = " Error!  Batch Number , Channels   Tensor1  != Tensor2 !!!";
        std::cerr << msg << std::endl;
    }
    int resN = DimT1[0];
    int resC = DimT1[1];
    int resH = DimT1[2];
    int resW = DimT2[3];

    int size = resN * resC * resH * resW;

    int K = DimT1[3];
    Tensor4D res(resN, resC, resH, resW);

    for (int n = 0; n < resN; n++)
    {
        for (int c = 0; c < resC; c++)
        {
            //cuda opt here, cuBlas BatchGemm
            int rtn = MatMul2D((float*)&res(n, c, 0, 0), (const float*)&tensor1(n, c, 0, 0), (const float*)&tensor2(n, c, 0, 0), resH, resW, K);

#if 0
            for (int h = 0; h < resH; h++)
            {
                for (int w = 0; w < resW; w++)
                {
                    float accum = 0;
                    for (int k = 0; k < K; k++)
                    {
                        accum += tensor1(n, c, h, k) * tensor2(n, c, k, w);
                    }
                    res(n, c, h, w) = accum;
                }
            }
#endif
        }
    }
    return res;
}

int main()
{
    cout << "\n\n ============================================================== " << endl;
    cout << " =========== vector  w(1, 1, 2, 2)  x(1, 1, 2, 1)   " << endl;
    cout << " ============================================================== \n\n" << endl;
    Tensor4D x(1, 1, 2, 1);
    x(0, 0, 0, 0) = 0.2;
    x(0, 0, 1, 0) = 0.4;

    Tensor4D w(1, 1, 2, 2);

    w(0, 0, 0, 0) = 0.1;
    w(0, 0, 1, 0) = -0.3;
    w(0, 0, 0, 1) = 0.5;
    w(0, 0, 1, 1) = 0.8;

    Tensor4D res0 = MulcuBlas(w, x);
    cout << res0.view() << endl;

    cout << "\n\n ============================================================== " << endl;
    cout << " =========== MatMul  Tensor1(1, 1, 2, 3)   Tensor2(1, 1, 3, 2) " << endl;
    cout << " ============================================================== \n\n" << endl;

    Tensor4D Tensor1(1, 1, 2, 3);
    Tensor1(0, 0, 0, 0) = 1;
    Tensor1(0, 0, 0, 1) = 2;
    Tensor1(0, 0, 0, 2) = 3;
    Tensor1(0, 0, 1, 0) = 4;
    Tensor1(0, 0, 1, 1) = 5;
    Tensor1(0, 0, 1, 2) = 6;

    Tensor4D Tensor2(1, 1, 3, 2);
    Tensor2(0, 0, 0, 0) = 1;
    Tensor2(0, 0, 0, 1) = 2;
    Tensor2(0, 0, 1, 0) = 1;
    Tensor2(0, 0, 1, 1) = 2;
    Tensor2(0, 0, 2, 0) = 1;
    Tensor2(0, 0, 2, 1) = 2;

    Tensor4D res = MulcuBlas(Tensor1, Tensor2);
    cout << res.view() << endl;

    cout << "\n\n ============================================================== " << endl;
    cout << " =========== MatMul  Tensor1(1, 2, 2, 3)   Tensor2(1, 2, 3, 2) " << endl;
    cout << " ============================================================== \n\n" << endl;

    Tensor4D Tensor3(1, 2, 2, 3);
    Tensor3(0, 0, 0, 0) = 1;
    Tensor3(0, 0, 0, 1) = 2;
    Tensor3(0, 0, 0, 2) = 3;
    Tensor3(0, 0, 1, 0) = 4;
    Tensor3(0, 0, 1, 1) = 5;
    Tensor3(0, 0, 1, 2) = 6;

    Tensor3(0, 1, 0, 0) = 1;
    Tensor3(0, 1, 0, 1) = 2;
    Tensor3(0, 1, 0, 2) = 3;
    Tensor3(0, 1, 1, 0) = 4;
    Tensor3(0, 1, 1, 1) = 5;
    Tensor3(0, 1, 1, 2) = 6;


    Tensor4D Tensor4(1, 2, 3, 2);
    Tensor4(0, 0, 0, 0) = 1;
    Tensor4(0, 0, 0, 1) = 2;
    Tensor4(0, 0, 1, 0) = 1;
    Tensor4(0, 0, 1, 1) = 2;
    Tensor4(0, 0, 2, 0) = 1;
    Tensor4(0, 0, 2, 1) = 2;

    Tensor4(0, 1, 0, 0) = 1;
    Tensor4(0, 1, 0, 1) = 2;
    Tensor4(0, 1, 1, 0) = 1;
    Tensor4(0, 1, 1, 1) = 2;
    Tensor4(0, 1, 2, 0) = 1;
    Tensor4(0, 1, 2, 1) = 2;

    Tensor4D res2 = MulcuBlas(Tensor3, Tensor4);
    cout << res2.view() << endl;

    cout << "\n\n ============================================================== " << endl;
    cout << " =========== MatMul  Tensor1(2, 2, 2, 3)   Tensor2(2, 2, 3, 2) " << endl;
    cout << " ============================================================== \n\n" << endl;

    Tensor4D Tensor5(2, 2, 2, 3);
    Tensor5(0, 0, 0, 0) = 1;
    Tensor5(0, 0, 0, 1) = 2;
    Tensor5(0, 0, 0, 2) = 3;
    Tensor5(0, 0, 1, 0) = 4;
    Tensor5(0, 0, 1, 1) = 5;
    Tensor5(0, 0, 1, 2) = 6;

    Tensor5(0, 1, 0, 0) = 1;
    Tensor5(0, 1, 0, 1) = 2;
    Tensor5(0, 1, 0, 2) = 3;
    Tensor5(0, 1, 1, 0) = 4;
    Tensor5(0, 1, 1, 1) = 5;
    Tensor5(0, 1, 1, 2) = 6;

    Tensor5(1, 0, 0, 0) = 1;
    Tensor5(1, 0, 0, 1) = 2;
    Tensor5(1, 0, 0, 2) = 3;
    Tensor5(1, 0, 1, 0) = 4;
    Tensor5(1, 0, 1, 1) = 5;
    Tensor5(1, 0, 1, 2) = 6;

    Tensor5(1, 1, 0, 0) = 1;
    Tensor5(1, 1, 0, 1) = 2;
    Tensor5(1, 1, 0, 2) = 3;
    Tensor5(1, 1, 1, 0) = 4;
    Tensor5(1, 1, 1, 1) = 5;
    Tensor5(1, 1, 1, 2) = 6;


    Tensor4D Tensor6(2, 2, 3, 2);
    Tensor6(0, 0, 0, 0) = 1;
    Tensor6(0, 0, 0, 1) = 2;
    Tensor6(0, 0, 1, 0) = 1;
    Tensor6(0, 0, 1, 1) = 2;
    Tensor6(0, 0, 2, 0) = 1;
    Tensor6(0, 0, 2, 1) = 2;

    Tensor6(0, 1, 0, 0) = 1;
    Tensor6(0, 1, 0, 1) = 2;
    Tensor6(0, 1, 1, 0) = 1;
    Tensor6(0, 1, 1, 1) = 2;
    Tensor6(0, 1, 2, 0) = 1;
    Tensor6(0, 1, 2, 1) = 2;

    Tensor6(1, 0, 0, 0) = 1;
    Tensor6(1, 0, 0, 1) = 2;
    Tensor6(1, 0, 1, 0) = 1;
    Tensor6(1, 0, 1, 1) = 2;
    Tensor6(1, 0, 2, 0) = 1;
    Tensor6(1, 0, 2, 1) = 2;

    Tensor6(1, 1, 0, 0) = 1;
    Tensor6(1, 1, 0, 1) = 2;
    Tensor6(1, 1, 1, 0) = 1;
    Tensor6(1, 1, 1, 1) = 2;
    Tensor6(1, 1, 2, 0) = 1;
    Tensor6(1, 1, 2, 1) = 2;


    Tensor4D res3 = MulcuBlas(Tensor5, Tensor6);
    cout << res3.view() << endl;

    return 0;
}