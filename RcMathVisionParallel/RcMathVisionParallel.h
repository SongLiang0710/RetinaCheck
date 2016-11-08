//
// @Author	Song Liang, Northeastern University ,CN 
//
// The code and binary file are attributed MIT license. See the detail and latest update in 
// https://github.com/SongLiang0710/RetinaCheck
//
// This copy of program is developed in Windows 7, Mathematica 11.0, CUDA Toolkit 7.5
//

#ifndef RCMATHVISIONPARALLEL_H
#define RCMATHVISIONPARALLEL_H

#include "wstp.h"
#include <cuda_runtime.h>
#include <cuComplex.h>

#define PI 3.14159265359

#define   DEFAULT_BLOCKDIM_X 32
#define   DEFAULT_BLOCKDIM_Y 8
#define   DEFAULT_BLOCKDIM_Z 1

typedef struct theMatrix3{
	double D00, D01, D02,
	       D10, D11, D12,
	       D20, D21, D22;
} matrix3by3;


__inline__  int iDivUp (int num, int den){
	return num % den == 0 ? num / den : num / den + 1;
}

#if defined(__cplusplus)
extern "C" {
#endif

int setGaussianKernel(double sigma, double stride);

void gaussianDerivative2D(
    double *d_Dst,
    double *d_Src,
    int width,
    int height,
    double sigma,
    int nx,
    int ny
);

void gaugeDerivative2D(
    double *d_Dst,
    double *d_Src,
    int width,
    int height,
    double sigma,
    int nw,
    int nv
);

void cakeWaveletStackFourier(
    cuDoubleComplex *d_Dst,
    int size,
    int pieces,
    int k,
    double t,
    int q,
    int overlap,
    int periodicity
);

void cakeWaveletStack(
    cuDoubleComplex *d_Dst,
    int size,
    int pieces,
    int k,
    double t,
    int q,
    int s,
    int overlap,
    int periodicity
);

void OS2DCakeTransform(
    cuDoubleComplex *d_Dst,
    double *d_Image,
    int imagewidth,
    int imageheight,
    int kernelsize,
    int pieces,
    int k,
    double t,
    int q,
    int s,
    int overlap,
    int periodicity,
    int method
);

void GaussianDerivativeSE2(
    double *d_Dst,
    double *d_Src,
    int width,
    int height,
    int stack,
    int radiusS,
    int radiusO,
    int nx,
    int ny,
    int nt
);

void OS2DGaussianDerivative(
    double *d_Dst,
    double *d_Src,
    int width,
    int height,
    int stack,
    double sigmaS,
    double sigmaO,
    double mu,
    int order
);

void OS2DGaussianHessian(
    double *d_Dst,
    double *d_Src,
    int width,
    int height,
    int stack,
    double sigmaS,
    double sigmaO,
    double mu
);

void OS2DHessianFeatures(
    double3 *d_Dst,
    matrix3by3 *d_Src,
    int width,
    int height,
    int stack,
    double mu
);

void nonlinearDiffusionFunction(
    double *d_Dst,
    double *d_Src,
    int width,
    int height,
    int stack,
    double c
);

void OS2DDiffusionStepExplicit(
    double *d_Dst,
    double *d_Src,
    int width,
    int height,
    int stack,
    int k,
    int steps,
    double tau
);

void OS2DCoherenceEnhancingDiffusionStep(
    double *d_Dst,
    double *d_Src,
    int width,
    int height,
    int stack,
    int k,
    int steps,
    double tau,
    double mu
);

void SE2FiniteDerivative(
    double *d_Dst,
    double *d_Src,
    int width,
    int height,
    int stack,
    int order,
    int k
);

void SE2FiniteDerivativeStep(
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
);

void SE2FiniteDerivativeHessian(
    matrix3by3 *d_Dst,
    double *d_Src,
    int width,
    int height,
    int stack,
    int k
);

#if defined(__cplusplus)
}
#endif


#endif
