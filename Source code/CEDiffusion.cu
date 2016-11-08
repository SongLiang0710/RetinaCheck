//
// @Author	Song Liang, Northeastern University ,CN 
//
// The code and binary file are attributed MIT license. See the detail and latest update in 
// https://github.com/SongLiang0710/RetinaCheck
//
// This copy of program is developed in Windows 7, Mathematica 11.0, CUDA Toolkit 7.5
//
// Functions for coherence-enhancing nonlinear diffusion algorithms.

#include <math.h>
#include "RcMathVisionParallel.h"

#include "matrix3by3.cuh"
#include "BSpline.cuh"


__global__ void cuNonlinearCoefficient(
    double *d_Dst,
    double *d_Src,
    int width,
    int height,
    double c
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;
    const int id = idx + idy * width + idz * width * height;

    if (idx < width && idy < height)
    {
        double conf = d_Src[id];

        if (conf <= 0) {
            d_Dst[id] = 1;
        }

        else {
            d_Dst[id] = exp(-conf / c); 
        }
    }
}

extern "C" void nonlinearDiffusionFunction(
    double *d_Dst,
    double *d_Src,
    int width,
    int height,
    int stack,
    double c
)
{
    dim3 blocks( iDivUp(width, DEFAULT_BLOCKDIM_X), iDivUp(height, DEFAULT_BLOCKDIM_Y), iDivUp(stack,  DEFAULT_BLOCKDIM_Z));
    dim3 threads( DEFAULT_BLOCKDIM_X, DEFAULT_BLOCKDIM_Y, DEFAULT_BLOCKDIM_Z);

    cuNonlinearCoefficient<<<blocks, threads>>>( d_Dst, d_Src, width, height, c);
}

__global__ void cuNonlinearDiffusionExplicit(
    double *d_Dst,
    double *d_LwForward,
    double *d_LwBackward,
    double *d_LvForward,
    double *d_LvBackward,
    double *d_Lwt,
    double *d_Dtt,
    double *d_Dww,
    double *d_Dvv,
    double *d_Dwt,
    double *d_Src,
    int width,
    int height,
    int stack,
    double tau
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;
    const int id = idx + idy * width + idz * width * height;

    if (idx < width && idy < height)
    {
        double Ltt, Lww, Lvv, Lwt, Ltw;
        double local = d_Src[id];

        int idtF = idz < stack - 1 ? id + width * height : idx + idy * width;
        int idtB = idz > 0 ? id - width * height : id + (stack - 1) * width * height;

        Ltt = pow(stack / PI, 2) * (d_Src[idtF] - 2 * local + d_Src[idtB]);

        Lww = d_LwForward[id] - 2 * local + d_LwBackward[id];

        Lvv = d_LvForward[id] - 2 * local + d_LvBackward[id];

        Ltw = stack / (PI * 4) * (d_LwForward[idtF] - d_LwBackward[idtF] - d_LwForward[idtB] + d_LwBackward[idtB]);

        Lwt = stack / (PI * 4) * d_Lwt[id];

        d_Dst[id] = local + tau * (Ltt * d_Dtt[id] + Lww * d_Dww[id] + Lvv * d_Dvv[id] + (Ltw + Lwt) * d_Dwt[id]);
    }
}

extern "C" void OS2DDiffusionStepExplicit(
    double *d_Dst,
    double *d_Src,
    int width,
    int height,
    int stack,
    int k,
    int steps,
    double tau
)
{
    int blob = width * height * stack;

    double *d_LwForward, *d_LwBackward, *d_LvForward, *d_LvBackward, *d_Lwt, *d_Buf;
    cudaMalloc((void **)&d_LwForward, blob * sizeof(double));
    cudaMalloc((void **)&d_LwBackward, blob * sizeof(double));
    cudaMalloc((void **)&d_LvForward, blob * sizeof(double));
    cudaMalloc((void **)&d_LvBackward, blob * sizeof(double));
    cudaMalloc((void **)&d_Lwt, blob * sizeof(double));
    cudaMalloc((void **)&d_Buf, blob * sizeof(double));

    dim3 blocks( iDivUp(width, DEFAULT_BLOCKDIM_X), iDivUp(height, DEFAULT_BLOCKDIM_Y), iDivUp(stack,  DEFAULT_BLOCKDIM_Z));
    dim3 threads( DEFAULT_BLOCKDIM_X, DEFAULT_BLOCKDIM_Y, DEFAULT_BLOCKDIM_Z);

    if (steps % 2 == 0)
        cudaMemcpy(d_Dst, d_Src, blob * sizeof(double), cudaMemcpyDeviceToDevice);
    else
        cudaMemcpy(d_Buf, d_Src, blob * sizeof(double), cudaMemcpyDeviceToDevice);

    while (steps-- > 0)
    {
        if (steps % 2 == 0)
        {
            SE2FiniteDerivativeStep( d_LwForward, d_LwBackward, d_LvForward, d_LvBackward, d_Lwt, NULL, NULL, d_Buf, width, height, stack, k);

            cuNonlinearDiffusionExplicit<<<blocks, threads>>>( d_Dst, d_LwForward, d_LwBackward, d_LvForward, d_LvBackward, d_Lwt, d_Src + blob, d_Src + 2 * blob, d_Src + 3 * blob, d_Src + 4 * blob, d_Buf, width, height, stack, tau);
        }

        else
        {
            SE2FiniteDerivativeStep( d_LwForward, d_LwBackward, d_LvForward, d_LvBackward, d_Lwt, NULL, NULL, d_Dst, width, height, stack, k);

            cuNonlinearDiffusionExplicit<<<blocks, threads>>>( d_Buf, d_LwForward, d_LwBackward, d_LvForward, d_LvBackward, d_Lwt, d_Src + blob, d_Src + 2 * blob, d_Src + 3 * blob, d_Src + 4 * blob, d_Dst, width, height, stack, tau);
        }
    }

    cudaFree(d_LwForward);
    cudaFree(d_LwBackward);
    cudaFree(d_LvForward);
    cudaFree(d_LvBackward);
    cudaFree(d_Lwt);
    cudaFree(d_Buf);
}

__global__ void cuCoherenceEnhancingDiffusion(
    double *d_Dst,
    matrix3by3 *d_Hes,
    matrix3by3 *d_Tensor,
    double *d_Src,
    int width,
    int height,
    int stack,
    double tau,
    double mu
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;
    const int id = idx + idy * width + idz * width * height;

    if (idx < width && idy < height)
    {
        matrix3by3 hes = d_Hes[id];
        matrix3by3 tensor = d_Tensor[id];

        d_Dst[id] = d_Src[id] + tau * (mu*mu * hes.D00 * tensor.D00 + mu * hes.D01 * tensor.D01 + mu * hes.D02 * tensor.D02 + mu * hes.D10 * tensor.D10 + hes.D11 * tensor.D11 + hes.D12 * tensor.D12 + mu * hes.D20 * tensor.D20 + hes.D21 * tensor.D21 + hes.D22 * tensor.D22);
    }
}

__global__ void cuCoherenceEnhancingDiffusionTensor(
    matrix3by3 *d_Tensor,
    double *d_nonlinearcoef,
    double *d_curvature,
    double *d_deviation,
    int width,
    int height,
    int stack,
    double mu
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;
    const int id = idx + idy * width + idz * width * height;

    if (idx < width && idy < height)
    {
        double dfh = d_deviation[id];
        matrix3by3 RdH = make_matrix3by3(1, 0, 0, 0, cos(dfh), -sin(dfh), 0, sin(dfh), cos(dfh));

        double alpha = atan(d_curvature[id] / mu);
        matrix3by3 Qku = make_matrix3by3(cos(alpha), -sin(alpha), 0, sin(alpha), cos(alpha), 0, 0, 0, 1);

        d_Tensor[id] = matrixMultiply(matrixMultiply(RdH, matrixMultiply(Qku, make_diagonal3by3(d_nonlinearcoef[id], 1, d_nonlinearcoef[id]))), transpose_matrix3by3(matrixMultiply(RdH, Qku)));
    }
}


extern "C" void OS2DCoherenceEnhancingDiffusionStep(
    double *d_Dst,
    double *d_Src,
    int width,
    int height,
    int stack,
    int k,
    int steps,
    double tau,
    double mu
)
{
    int blob = width * height * stack;

    matrix3by3 *d_Tensor, *d_Hes;

    double *d_Buf;

    cudaMalloc((void **)&d_Tensor, blob * sizeof(matrix3by3));
    cudaMalloc((void **)&d_Hes, blob * sizeof(matrix3by3));
    cudaMalloc((void **)&d_Buf, blob * sizeof(double));

    dim3 blocks( iDivUp(width, DEFAULT_BLOCKDIM_X), iDivUp(height, DEFAULT_BLOCKDIM_Y), iDivUp(stack,  DEFAULT_BLOCKDIM_Z));
    dim3 threads( DEFAULT_BLOCKDIM_X, DEFAULT_BLOCKDIM_Y, DEFAULT_BLOCKDIM_Z);

    cuCoherenceEnhancingDiffusionTensor<<<blocks, threads>>>( d_Tensor, d_Src + blob, d_Src + 2 * blob, d_Src + 3 * blob, width, height, stack, mu);

    if (steps % 2 == 0)
        cudaMemcpy(d_Dst, d_Src, blob * sizeof(double), cudaMemcpyDeviceToDevice);
    else
        cudaMemcpy(d_Buf, d_Src, blob * sizeof(double), cudaMemcpyDeviceToDevice);

    while (steps-- > 0)
    {
        if (steps % 2 == 0)
        {
            SE2FiniteDerivativeHessian( d_Hes, d_Buf, width, height, stack, k);

            cuCoherenceEnhancingDiffusion<<<blocks, threads>>>( d_Dst, d_Hes, d_Tensor, d_Buf, width, height, stack, tau, mu);
        }

        else
        {
            SE2FiniteDerivativeHessian( d_Hes, d_Dst, width, height, stack, k);

            cuCoherenceEnhancingDiffusion<<<blocks, threads>>>( d_Buf, d_Hes, d_Tensor, d_Dst, width, height, stack, tau, mu);
        }
    }

    cudaFree(d_Tensor);
    cudaFree(d_Hes);
    cudaFree(d_Buf);
}


