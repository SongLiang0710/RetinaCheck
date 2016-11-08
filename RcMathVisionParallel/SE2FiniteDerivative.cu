//
// @Author	Song Liang, Northeastern University ,CN 
//
// The code and binary file are attributed MIT license. See the detail and latest update in 
// https://github.com/SongLiang0710/RetinaCheck
//
// This copy of program is developed in Windows 7, Mathematica 11.0, CUDA Toolkit 7.5
//
// Functions for interpolated finite derivatives in SE2 space.

#include <math.h>
#include "RcMathVisionParallel.h"

#include "BSpline.cuh"

__global__ void cuSE2FiniteDerivativeTT(
    double *d_Dst,
    double *d_Src,
    int width,
    int height,
    int stack
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;
    const int id = idx + idy * width + idz * width * height;

    if (idx < width && idy < height) 
    {
        int idtF = idz < stack - 1 ? id + width * height : idx + idy * width;
        int idtB = idz > 0 ? id - width * height : id + (stack - 1) * width * height;

        d_Dst[id] = pow(stack / PI, 2) * (d_Src[idtF] - 2 * d_Src[id] + d_Src[idtB]);
    }
}

__global__ void cuSE2FiniteDerivativeT(
    double *d_Dst,
    double *d_Src,
    int width,
    int height,
    int stack
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;
    const int id = idx + idy * width + idz * width * height;

    if (idx < width && idy < height) 
    {
        int idtF = idz < stack - 1 ? id + width * height : idx + idy * width;
        int idtB = idz > 0 ? id - width * height : id + (stack - 1) * width * height;

        d_Dst[id] = stack / (PI * 2) * (d_Src[idtF] - d_Src[idtB]);
    }
}

__global__ void cuSE2FiniteDerivativeWW(
    double *d_Dst,
    double *d_Src,
    int width,
    int height,
    int stack,
    int k
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;
    const int id = idx + idy * width + idz * width * height;

    if (idx < width && idy < height) 
    {
        double c = cos(idz * PI / stack);
        double s = sin(idz * PI / stack);

        Spline_ptr SplineFunction = set_splineFunction(k);

        double sum = 0;
        int i, j;
        double val;

        for (i = -2; i <= 2; i++) {
            for (j = -2; j <= 2; j++) {
                val = (idx + j >= 0 && idx + j < width && idy + i >= 0 && idy + i < height) ? d_Src[id + j + i * width] : 0;
                sum += val * SplineFunction(c - j) * SplineFunction(s - i);
                sum += val * SplineFunction(-c - j) * SplineFunction(-s - i);
            }
        }

        d_Dst[id] = sum - 2 * d_Src[id];
    }
}

__global__ void cuSE2FiniteDerivativeW(
    double *d_Dst,
    double *d_Src,
    int width,
    int height,
    int stack,
    int k
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;
    const int id = idx + idy * width + idz * width * height;

    if (idx < width && idy < height) 
    {
        double c = cos(idz * PI / stack);
        double s = sin(idz * PI / stack);

        Spline_ptr SplineFunction = set_splineFunction(k);

        double sum = 0;
        int i, j;
        double val;

        for (i = -2; i <= 2; i++) {
            for (j = -2; j <= 2; j++) {
                val = (idx + j >= 0 && idx + j < width && idy + i >= 0 && idy + i < height) ? d_Src[id + j + i * width] : 0;
                sum += val * SplineFunction(c - j) * SplineFunction(s - i);
                sum -= val * SplineFunction(-c - j) * SplineFunction(-s - i);
            }
        }

        d_Dst[id] = sum / 2;
    }
}

__global__ void cuSE2FiniteDerivativeVV(
    double *d_Dst,
    double *d_Src,
    int width,
    int height,
    int stack,
    int k
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;
    const int id = idx + idy * width + idz * width * height;

    if (idx < width && idy < height) 
    {
        double c = cos(idz * PI / stack);
        double s = sin(idz * PI / stack);

        Spline_ptr SplineFunction = set_splineFunction(k);

        double sum = 0;
        int i, j;
        double val;

        for (i = -2; i <= 2; i++) {
            for (j = -2; j <= 2; j++) {
                val = (idx + j >= 0 && idx + j < width && idy + i >= 0 && idy + i < height) ? d_Src[id + j + i * width] : 0;
                sum += val * SplineFunction(-s - j) * SplineFunction(c - i);
                sum += val * SplineFunction(s - j) * SplineFunction(-c - i);
            }
        }

        d_Dst[id] = sum - 2 * d_Src[id];
    }
}

__global__ void cuSE2FiniteDerivativeV(
    double *d_Dst,
    double *d_Src,
    int width,
    int height,
    int stack,
    int k
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;
    const int id = idx + idy * width + idz * width * height;

    if (idx < width && idy < height) 
    {
        double c = cos(idz * PI / stack);
        double s = sin(idz * PI / stack);

        Spline_ptr SplineFunction = set_splineFunction(k);

        double sum = 0;
        int i, j;
        double val;

        for (i = -2; i <= 2; i++) {
            for (j = -2; j <= 2; j++) {
                val = (idx + j >= 0 && idx + j < width && idy + i >= 0 && idy + i < height) ? d_Src[id + j + i * width] : 0;
                sum += val * SplineFunction(-s - j) * SplineFunction(c - i);
                sum -= val * SplineFunction(s - j) * SplineFunction(-c - i);
            }
        }

        d_Dst[id] = sum / 2;
    }
}

__global__ void cuSE2FiniteDerivativeWV(
    double *d_Dst,
    double *d_Src,
    int width,
    int height,
    int stack,
    int k
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;
    const int id = idx + idy * width + idz * width * height;

    if (idx < width && idy < height) 
    {
        double c = cos(idz * PI / stack);
        double s = sin(idz * PI / stack);

        Spline_ptr SplineFunction = set_splineFunction(k);

        double sum = 0;
        int i, j;
        double val;

        for (i = -3; i <= 3; i++) {
            for (j = -3; j <= 3; j++) {
                val = (idx + j >= 0 && idx + j < width && idy + i >= 0 && idy + i < height) ? d_Src[id + j + i * width] : 0;
                sum += val * (SplineFunction(c - s - j) * SplineFunction(s + c - i) - SplineFunction(-c - s - j) * SplineFunction(-s + c - i) - SplineFunction(c + s - j) * SplineFunction(s - c - i) + SplineFunction(-c + s - j) * SplineFunction(-s - c - i));
            }
        }

        d_Dst[id] = sum / 4;
    }
}

extern "C" void SE2FiniteDerivative(
    double *d_Dst,
    double *d_Src,
    int width,
    int height,
    int stack,
    int order,
    int k
)
{
    dim3 blocks( iDivUp(width, DEFAULT_BLOCKDIM_X), iDivUp(height, DEFAULT_BLOCKDIM_Y), iDivUp(stack,  DEFAULT_BLOCKDIM_Z));
    dim3 threads( DEFAULT_BLOCKDIM_X, DEFAULT_BLOCKDIM_Y, DEFAULT_BLOCKDIM_Z);

    switch (order)
    {
        case 11: cuSE2FiniteDerivativeTT<<<blocks, threads>>>( d_Dst, d_Src, width, height, stack); break;
        case 22: cuSE2FiniteDerivativeWW<<<blocks, threads>>>( d_Dst, d_Src, width, height, stack, k); break;
        case 23: cuSE2FiniteDerivativeWV<<<blocks, threads>>>( d_Dst, d_Src, width, height, stack, k); break;
        case 32: cuSE2FiniteDerivativeWV<<<blocks, threads>>>( d_Dst, d_Src, width, height, stack, k); break;
        case 33: cuSE2FiniteDerivativeVV<<<blocks, threads>>>( d_Dst, d_Src, width, height, stack, k); break;
        default: 
	    double *d_Buf;
	    cudaMalloc((void **)&d_Buf, width * height * stack * sizeof(double));

	    switch (order / 10)
	    {
		    case 1: cuSE2FiniteDerivativeT<<<blocks, threads>>>( d_Buf, d_Src, width, height, stack); break;
		    case 2: cuSE2FiniteDerivativeW<<<blocks, threads>>>( d_Buf, d_Src, width, height, stack, k); break;
		    case 3: cuSE2FiniteDerivativeV<<<blocks, threads>>>( d_Buf, d_Src, width, height, stack, k); break;
	    }

	    switch (order % 10)
	    {
		    case 1: cuSE2FiniteDerivativeT<<<blocks, threads>>>( d_Dst, d_Buf, width, height, stack); break;
		    case 2: cuSE2FiniteDerivativeW<<<blocks, threads>>>( d_Dst, d_Buf, width, height, stack, k); break;
		    case 3: cuSE2FiniteDerivativeV<<<blocks, threads>>>( d_Dst, d_Buf, width, height, stack, k); break;
	    }

	    cudaFree(d_Buf);
    }
}

__global__ void cuSE2FiniteDerivativeStep(
    double *d_LwForward,
    double *d_LwBackward,
    double *d_LvForward,
    double *d_LvBackward,
    double *d_Lwt,
    double *d_Lvt,
    double *d_Lwv,
    double *d_Src,
    double *d_Lt,
    int width,
    int height,
    int stack,
    int k
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;
    const int id = idx + idy * width + idz * width * height;

    if (idx < width && idy < height)
    {
        double c = cos(idz * PI / stack);
        double s = sin(idz * PI / stack);

        Spline_ptr SplineFunction = set_splineFunction(k);

        double sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
        int i, j;

        double val;

        for (i = -2; i <= 2; i++) {
            for (j = -2; j <= 2; j++) {
                val = (idx + j >= 0 && idx + j < width && idy + i >= 0 && idy + i < height) ? d_Src[id + j + i * width] : 0;
                sum1 += val * SplineFunction(c - j) * SplineFunction(s - i);
                sum2 += val * SplineFunction(-c - j) * SplineFunction(-s - i);
                sum3 += val * SplineFunction(-s - j) * SplineFunction(c - i);
                sum4 += val * SplineFunction(s - j) * SplineFunction(-c - i);
            }
        }

        d_LwForward[id] = sum1;
        d_LwBackward[id] = sum2;
        d_LvForward[id] = sum3;
        d_LvBackward[id] = sum4;

        sum1 = sum2 = 0;

        for (i = -2; i <= 2; i++) {
            for (j = -2; j <= 2; j++) {
                val = (idx + j >= 0 && idx + j < width && idy + i >= 0 && idy + i < height) ? d_Lt[id + j + i * width] : 0;
                sum1 += val * (SplineFunction(c - j) * SplineFunction(s - i) - SplineFunction(-c - j) * SplineFunction(-s - i));
                if (d_Lvt) sum2 += val * (SplineFunction(-s - j) * SplineFunction(c - i) - SplineFunction(s - j) * SplineFunction(-c - i));
            }
        }

        d_Lwt[id] = sum1 / 2;
        if (d_Lvt) d_Lvt[id] = sum2 / 2;

        if (d_Lwv)
        {
            sum3 = 0;
            for (i = -3; i <= 3; i++) {
                for (j = -3; j <= 3; j++) {
                    val = (idx + j >= 0 && idx + j < width && idy + i >= 0 && idy + i < height) ? d_Src[id + j + i * width] : 0;
                    sum3 += val * (SplineFunction(c - s - j) * SplineFunction(s + c - i) - SplineFunction(-c - s - j) * SplineFunction(-s + c - i) - SplineFunction(c + s - j) * SplineFunction(s - c - i) + SplineFunction(-c + s - j) * SplineFunction(-s - c - i));
                }
            }
            d_Lwv[id] = sum3 / 4;
        }
    }
}


extern "C" void SE2FiniteDerivativeStep(
    double *d_LwForward,
    double *d_LwBackward,
    double *d_LvForward,
    double *d_LvBackward,
    double *d_Lwt,
    double *d_Lvt,
    double *d_Lwv,
    double *d_Src,
    int width,
    int height,
    int stack,
    int k
)
{
    double *d_Lt;
    cudaMalloc((void **)&d_Lt, width * height * stack * sizeof(double));

    dim3 blocks( iDivUp(width, DEFAULT_BLOCKDIM_X), iDivUp(height, DEFAULT_BLOCKDIM_Y), iDivUp(stack,  DEFAULT_BLOCKDIM_Z));
    dim3 threads( DEFAULT_BLOCKDIM_X, DEFAULT_BLOCKDIM_Y, DEFAULT_BLOCKDIM_Z);

    cuSE2FiniteDerivativeT<<<blocks, threads>>>( d_Lt, d_Src, width, height, stack);

    cuSE2FiniteDerivativeStep<<<blocks, threads>>>( d_LwForward, d_LwBackward, d_LvForward, d_LvBackward, d_Lwt, d_Lvt, d_Lwv, d_Src, d_Lt, width, height, stack, k);

    cudaFree(d_Lt);
}

__global__ void cuSE2FiniteDerivativeHessian(
    matrix3by3 *d_Dst,
    double *d_LwForward,
    double *d_LwBackward,
    double *d_LvForward,
    double *d_LvBackward,
    double *d_Lwt,
    double *d_Lvt,
    double *d_Lwv,
    double *d_Src,
    int width,
    int height,
    int stack
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;
    const int id = idx + idy * width + idz * width * height;

	if (idx < width && idy < height)
	{
	    double local = d_Src[id];

	    int idtF = idz < stack - 1 ? id + width * height : idx + idy * width;
	    int idtB = idz > 0 ? id - width * height : id + (stack - 1) * width * height;

	    double Ltt = pow(stack / PI, 2) * (d_Src[idtF] - 2 * local + d_Src[idtB]);

	    double Lww = d_LwForward[id] - 2 * local + d_LwBackward[id];

	    double Lvv = d_LvForward[id] - 2 * local + d_LvBackward[id];

	    double Ltw = stack / (PI * 4) * (d_LwForward[idtF] - d_LwBackward[idtF] - d_LwForward[idtB] + d_LwBackward[idtB]);

	    double Ltv = stack / (PI * 4) * (d_LvForward[idtF] - d_LvBackward[idtF] - d_LvForward[idtB] + d_LvBackward[idtB]);

	    matrix3by3 hes = {Ltt, d_Lwt[id], d_Lvt[id],
			      Ltw, Lww, d_Lwv[id],
			      Ltv, d_Lwv[id], Lvv};

	    d_Dst[id] = hes;
	} 
}

extern "C" void SE2FiniteDerivativeHessian(
    matrix3by3 *d_Dst,
    double *d_Src,
    int width,
    int height,
    int stack,
    int k
)
{
    int blob = width * height * stack;

    double *d_LwForward, *d_LwBackward, *d_LvForward, *d_LvBackward, *d_Lwt, *d_Lvt, *d_Lwv;

    cudaMalloc((void **)&d_LwForward, blob * sizeof(double));
    cudaMalloc((void **)&d_LwBackward, blob * sizeof(double));
    cudaMalloc((void **)&d_LvForward, blob * sizeof(double));
    cudaMalloc((void **)&d_LvBackward, blob * sizeof(double));
    cudaMalloc((void **)&d_Lwt, blob * sizeof(double));
    cudaMalloc((void **)&d_Lvt, blob * sizeof(double));
    cudaMalloc((void **)&d_Lwv, blob * sizeof(double));

    SE2FiniteDerivativeStep( d_LwForward, d_LwBackward, d_LvForward, d_LvBackward, d_Lwt, d_Lvt, d_Lwv, d_Src, width, height, stack, k);

    dim3 blocks( iDivUp(width, DEFAULT_BLOCKDIM_X), iDivUp(height, DEFAULT_BLOCKDIM_Y), iDivUp(stack,  DEFAULT_BLOCKDIM_Z));
    dim3 threads( DEFAULT_BLOCKDIM_X, DEFAULT_BLOCKDIM_Y, DEFAULT_BLOCKDIM_Z);

    cuSE2FiniteDerivativeHessian<<<blocks, threads>>>( d_Dst, d_LwForward, d_LwBackward, d_LvForward, d_LvBackward, d_Lwt, d_Lvt, d_Lwv, d_Src, width, height, stack);

    cudaFree(d_LwForward);
    cudaFree(d_LwBackward);
    cudaFree(d_LvForward);
    cudaFree(d_LvBackward);
    cudaFree(d_Lwt);
    cudaFree(d_Lvt);
    cudaFree(d_Lwv);

}

