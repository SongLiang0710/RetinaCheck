//
// @Author	Song Liang, Northeastern University ,CN 
//
// The code and binary file are attributed MIT license. See the detail and latest update in 
// https://github.com/SongLiang0710/RetinaCheck
//
// This copy of program is developed in Windows 7, Mathematica 11.0, CUDA Toolkit 7.5
//
// Orientation score Fourier domain prototype, spacial domain kernel, image OS transform

#include <math.h>
#include <cufft.h>
#include "RcMathVisionParallel.h"

#include "BSpline.cuh"


// Radial decay coeffcient

__device__ long factorial(int n){
	long prod = 1;
	for (int m = 2; m <= n; m++){
		prod *= m;
	}
	return prod;
}

__device__ double deMrho(int x, int y, double t, int q){
	double tsum = 1;
	double tmid = (x*x + y*y) / (4 * t);
	for (int k = 1; k <= q; k++){
		tsum += pow(tmid, k) / factorial(k);
	}
	return tsum * exp(-tmid);
}

// Cake wavelet stack generation in Fourier domain

__global__ void cuCakeWaveletStackFourier(
    void *d_Dst,
    int size,
    int pieces,
    int k,
    double t,
    int q,
    int overlap,
    int periodicity,
    int complexQ
)
{
    const int idX = blockIdx.x * blockDim.x + threadIdx.x;
    const int idY = blockIdx.y * blockDim.y + threadIdx.y;
    const int idZ = blockIdx.z * blockDim.z + threadIdx.z;

    int varX = (idX + size / 2) % size - size / 2;
    int varY = (idY + size / 2) % size - size / 2;

    // Averaging across the stack around origin
    if (varX <= 2 && varX >= -2 && varY <= 2 && varY >= -2){
	for (double ov = 0; ov < overlap; ov++) {

	    // Determining the output datatype by complexQ
	    if (complexQ == 0) {
	        double *ptr = (double *)d_Dst;
	        ptr[idX + idY * size + (idZ * overlap + (int)ov) * size * size] = 1. / pieces;
	    }
	    else {
	        cuDoubleComplex *ptr = (cuDoubleComplex *)d_Dst;
	        ptr[idX + idY * size + (idZ * overlap + (int)ov) * size * size] = make_cuDoubleComplex(1. / pieces, 0);
	    }
        }
    }

    // Normal calculation
    else if (idX < size && idY < size){
    
        Spline_ptr SplineFunction = set_splineFunction(k);
        
        double varBS, res;
	for (double ov = 0; ov < overlap; ov++) {

	    varBS = remainder((2 / periodicity) * pieces * (atan2((double)varY, (double)varX) / (2 * PI)) + idZ + ov / overlap,  (2 / periodicity) * pieces);
            res = SplineFunction(varBS) * deMrho(varX, varY, t, q) / overlap;

	    if (complexQ == 0) {
	        double *ptr = (double *)d_Dst;
	        ptr[idX + idY * size + (idZ * overlap + (int)ov) * size * size] = res;
	    }
	    else {
	        cuDoubleComplex *ptr = (cuDoubleComplex *)d_Dst;
	        ptr[idX + idY * size + (idZ * overlap + (int)ov) * size * size] = make_cuDoubleComplex(res, 0);
	    }
        }
    }
}

#define   OS_BLOCKDIM_X 32
#define   OS_BLOCKDIM_Y 8
#define   OS_BLOCKDIM_Z 1

extern "C" void cakeWaveletStackFourier(
    cuDoubleComplex *d_Dst,
    int size,
    int pieces,
    int k,
    double t,
    int q,
    int overlap,
    int periodicity
)
{
    dim3 blocks( iDivUp(size, OS_BLOCKDIM_X), iDivUp(size, OS_BLOCKDIM_Y), pieces / OS_BLOCKDIM_Z);
    dim3 threads( OS_BLOCKDIM_X, OS_BLOCKDIM_Y, OS_BLOCKDIM_Z);

    cuCakeWaveletStackFourier<<<blocks, threads>>>( d_Dst, size, pieces, k, t, q, overlap, periodicity, 1);
}

// Radial decay in spacial domain by Gaussian window

__global__ void cuSpacialGaussianWindow(
    cuDoubleComplex *d_Dst,
    int width,
    int height,
    int s
)
{
    const int idX = blockIdx.x * blockDim.x + threadIdx.x;
    const int idY = blockIdx.y * blockDim.y + threadIdx.y;
    const int idZ = blockIdx.z * blockDim.z + threadIdx.z;
    const int id = idX + idY * width + idZ * width * height;

    int varX = (idX + width / 2) % width - width / 2;
    int varY = (idY + height / 2) % height - height / 2;

    if (idX < width && idY < height)
    {
//        double varBL = exp(-(varX*varX + varY*varY) / (4. * s)) / (4 * PI * s);
        double varBL = exp(-(varX*varX + varY*varY) / (2. * s * s)) / sqrt((double)width * height);
        d_Dst[id] = make_cuDoubleComplex(cuCreal(d_Dst[id]) * varBL, cuCimag(d_Dst[id]) * varBL);	
    }
}

extern "C" void cakeWaveletStack(
    cuDoubleComplex *d_Dst,
    int size,
    int pieces,
    int k,
    double t,
    int q,
    int s,
    int overlap,
    int periodicity
)
{
    // Generate cake kernel in Fourier domain at first

    cakeWaveletStackFourier( d_Dst, size, pieces, k, t, q, overlap, periodicity);

    // Transform into space domain using cufft library
    cufftHandle plan;
    int n[2] = {size, size};
    
    cufftPlanMany(&plan, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, pieces * overlap);
    cufftExecZ2Z(plan, d_Dst, d_Dst, CUFFT_INVERSE);

    cudaDeviceSynchronize();

    cufftDestroy(plan);

    // Radial decay by Gaussian window
    dim3 blocks( iDivUp(size, OS_BLOCKDIM_X), iDivUp(size, OS_BLOCKDIM_Y), pieces * overlap / OS_BLOCKDIM_Z);
    dim3 threads( OS_BLOCKDIM_X, OS_BLOCKDIM_Y, OS_BLOCKDIM_Z);

    cuSpacialGaussianWindow<<<blocks, threads>>>( d_Dst, size, size, s);
}

__global__ void cuImageReal2Complex(
    cuDoubleComplex *d_ImageComplex,
    double *d_Image,
    int imagewidth,
    int imageheight,
    int kernelsize
)
{
    const int idX = blockIdx.x * blockDim.x + threadIdx.x;
    const int idY = blockIdx.y * blockDim.y + threadIdx.y;
    const int id = idX + idY * imagewidth;

    if (idX < imagewidth && idY < imageheight){
        d_ImageComplex[id] = make_cuDoubleComplex(d_Image[id] / (imagewidth * imageheight), 0);
        // Notice : the magnitude compensated for the cufft transformation is put here temporarily 
    }
}

// Method 0 : layerwise list correlating

__global__ void cuOS2DCakeTransformCorrelation(
    cuDoubleComplex *d_Dst,
    cuDoubleComplex *d_Kernel,
    double *d_Image,
    int kernelsize,
    int imagewidth,
    int imageheight
)
{
    const int idX = blockIdx.x * blockDim.x + threadIdx.x;
    const int idY = blockIdx.y * blockDim.y + threadIdx.y;
    const int idZ = blockIdx.z * blockDim.z + threadIdx.z;

    int i, j;
    int kerX, kerY;
    cuDoubleComplex sum;

    if (idX < imagewidth && idY < imageheight) 
    {
        sum = make_cuDoubleComplex(0, 0);

        for (i = 0; i < kernelsize; i++)
            for (j = 0; j < kernelsize; j++)
            {
                kerX = j < (kernelsize + 1) / 2 ? j : j - kernelsize;
                kerY = i < (kernelsize + 1) / 2 ? i : i - kernelsize;

                if (idX + kerX >= 0 && idX + kerX < imagewidth && idY + kerY >= 0 && idY + kerY < imageheight)
                    sum = cuCfma(d_Kernel[j + i * kernelsize + idZ * kernelsize * kernelsize], make_cuDoubleComplex(d_Image[idX + kerX + (idY + kerY) * imagewidth], 0), sum);
            }

        d_Dst[idX + idY * imagewidth + idZ * imagewidth * imageheight] = sum;
    }
}

// Method 1 : fourier dot product by resampled kernel regarded as periodic

// Resample the kernel from prototype to imagesize
__global__ void cuArrayResample(
    cuDoubleComplex *d_Dst,
    double *d_Src,
    int sizeSrc,
    int widthDst,
    int heightDst
)
{
    const int idX = blockIdx.x * blockDim.x + threadIdx.x;
    const int idY = blockIdx.y * blockDim.y + threadIdx.y;
    const int idZ = blockIdx.z * blockDim.z + threadIdx.z;

    if (idX < widthDst && idY < heightDst)
    {
        double px = (double)idX * sizeSrc / widthDst;
        double py = (double)idY * sizeSrc / heightDst;

        int px1 = (int)floor(px);
        int px2 = (int)floor(px + 1);
        int py1 = (int)floor(py);
        int py2 = (int)floor(py + 1);
 
        d_Dst[idX + idY * widthDst + idZ * widthDst * heightDst] = make_cuDoubleComplex((d_Src[px1 + py1 * sizeSrc + idZ * sizeSrc * sizeSrc] * (py2 - py) + d_Src[px1 + (py2 % sizeSrc) * sizeSrc + idZ * sizeSrc * sizeSrc] * (py - py1)) * (px2 - px) + (d_Src[(px2 % sizeSrc) + py1 * sizeSrc + idZ * sizeSrc * sizeSrc] * (py2 - py) + d_Src[(px2 % sizeSrc) + (py2 % sizeSrc) * sizeSrc + idZ * sizeSrc * sizeSrc] * (py - py1)) * (px - px1), 0);
    }
}

extern "C" void arrayResample(
    cuDoubleComplex *d_Dst,
    double *d_Src,
    int sizeSrc,
    int widthDst,
    int heightDst,
    int stack
)
{
    dim3 blocks( iDivUp(widthDst, OS_BLOCKDIM_X), iDivUp(heightDst, OS_BLOCKDIM_Y), stack / OS_BLOCKDIM_Z);
    dim3 threads( OS_BLOCKDIM_X, OS_BLOCKDIM_Y, OS_BLOCKDIM_Z);

    cuArrayResample<<<blocks, threads>>>( d_Dst, d_Src, sizeSrc, widthDst, heightDst);
}

// Dot product in Fourier domain. This is used after spacial Gaussian filtering
__global__ void cuDotProduct(
    cuDoubleComplex *d_Dst,
    cuDoubleComplex *d_Kernel,
    cuDoubleComplex *d_Image,
    int width,
    int height
)
{
    const int idX = blockIdx.x * blockDim.x + threadIdx.x;
    const int idY = blockIdx.y * blockDim.y + threadIdx.y;
    const int idZ = blockIdx.z * blockDim.z + threadIdx.z;
    const int id = idX + idY * width + idZ * width * height;

    if (idX < width && idY < height)
    {
        d_Dst[id] = cuCmul(d_Kernel[id], d_Image[idX + idY * width]);
    }
}

// Method 2 : truncated product regarding kernel as resized

__global__ void cuOS2DCakeTransformResize(
    cuDoubleComplex *d_Dst,
    cuDoubleComplex *d_Kernel,
    cuDoubleComplex *d_Image,
    int kernelsize,
    int imagewidth,
    int imageheight
)
{
    const int idX = blockIdx.x * blockDim.x + threadIdx.x;
    const int idY = blockIdx.y * blockDim.y + threadIdx.y;
    const int idZ = blockIdx.z * blockDim.z + threadIdx.z;
    const int id = idX + idY * imagewidth + idZ * imagewidth * imageheight;

    int kerX = idX < (imagewidth + 1) / 2 ? idX : idX + kernelsize - imagewidth;
    int kerY = idY < (imageheight + 1) / 2 ? idY : idY + kernelsize - imageheight;

    if (idX < imagewidth && idY < imageheight) 
    {
        if ((idX < (kernelsize + 1) / 2 || idX >= imagewidth - kernelsize / 2) && (idY < (kernelsize + 1) / 2 || idY >= imageheight - kernelsize / 2))
            d_Dst[id] = cuCmul(d_Image[idX + idY * imagewidth], d_Kernel[kerX + kerY * kernelsize + idZ * kernelsize * kernelsize]);
        else
            d_Dst[id] = make_cuDoubleComplex(0, 0);
    }
}

extern "C" void OS2DCakeTransform(
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
)
{
    // Prepare the blocks and threads repeatedly used

    dim3 blocks( iDivUp(imagewidth, OS_BLOCKDIM_X), iDivUp(imageheight, OS_BLOCKDIM_Y), pieces * overlap / OS_BLOCKDIM_Z);
    dim3 threads( OS_BLOCKDIM_X, OS_BLOCKDIM_Y, OS_BLOCKDIM_Z);

    if (method == 1 || method == 2)
    {
        // Prepare the image in spacial, complex domain
        cuDoubleComplex *d_ImageComplex;
        cudaMalloc((void **)&d_ImageComplex, imagewidth * imageheight * sizeof(cuDoubleComplex));

        dim3 blockImage( iDivUp(imagewidth, 32), iDivUp(imageheight, 32));
        dim3 threadImage( 32, 32);

        cuImageReal2Complex<<<blockImage, threadImage>>>( d_ImageComplex, d_Image, imagewidth, imageheight, kernelsize);

        if (method == 1) // Resample
        {
            double *d_Kernel;
            cuDoubleComplex *d_KernelResample;

            cudaMalloc((void **)&d_Kernel, kernelsize * kernelsize * pieces * overlap * sizeof(double));
            cudaMalloc((void **)&d_KernelResample, imagewidth * imageheight * pieces * overlap * sizeof(cuDoubleComplex));

            cuCakeWaveletStackFourier<<<blocks, threads>>>( d_Kernel, kernelsize, pieces, k, t, q, overlap, periodicity, 0);

            arrayResample( d_KernelResample, d_Kernel, kernelsize, imagewidth, imageheight, pieces * overlap);

            // Transform into space domain and filter by Gaussian window

            int n[2] = {imagewidth, imageheight};
            cufftHandle plan1;
    
            cufftPlanMany(&plan1, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, pieces * overlap);
            cufftExecZ2Z(plan1, d_KernelResample, d_KernelResample, CUFFT_INVERSE);

            cudaDeviceSynchronize();
            cufftDestroy(plan1);

            cuSpacialGaussianWindow<<<blocks, threads>>>( d_KernelResample, imagewidth, imageheight, s);

            // Transform the image and kernel from spacial domain to Fourier domain respectively

            cufftHandle planImage, planKernel;
    
            cufftPlanMany(&planImage, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, 1);
            cufftExecZ2Z(planImage, d_ImageComplex, d_ImageComplex, CUFFT_FORWARD);
            cudaDeviceSynchronize();

            cufftPlanMany(&planKernel, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, pieces * overlap);
            cufftExecZ2Z(planKernel, d_KernelResample, d_KernelResample, CUFFT_FORWARD);
            cudaDeviceSynchronize();

            cufftDestroy(planImage);
            cufftDestroy(planKernel);

            // Dot product in Fourier domain
            cuDotProduct<<<blocks, threads>>>( d_Dst, d_KernelResample, d_ImageComplex, imagewidth, imageheight);

            // Transform back to spacial domain
            cufftHandle planDst;

            cufftPlanMany(&planDst, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, pieces * overlap);
            cufftExecZ2Z(planDst, d_Dst, d_Dst, CUFFT_INVERSE);

            cudaDeviceSynchronize();
            cufftDestroy(planDst);

            cudaFree(d_Kernel);
            cudaFree(d_KernelResample);
        }

        else if (method == 2) // Resize
        {
            cuDoubleComplex *d_Kernel;
            cudaMalloc((void **)&d_Kernel, kernelsize * kernelsize * pieces * overlap * sizeof(cuDoubleComplex));

            cakeWaveletStack( d_Kernel, kernelsize, pieces, k, t, q, s, overlap, periodicity);

            // Transform the image and stack of kernels from spacial domain to Fourier domain respectively

            cufftHandle planImage, planKernel;
            int imageDim[2] = {imagewidth, imageheight};
            int kernelDim[2] = {kernelsize, kernelsize};
    
            cufftPlanMany(&planImage, 2, imageDim, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, 1);
            cufftExecZ2Z(planImage, d_ImageComplex, d_ImageComplex, CUFFT_FORWARD);
            cudaDeviceSynchronize();
 
            cufftPlanMany(&planKernel, 2, kernelDim, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, pieces * overlap);
            cufftExecZ2Z(planKernel, d_Kernel, d_Kernel, CUFFT_FORWARD);
            cudaDeviceSynchronize();

            cufftDestroy(planImage);
            cufftDestroy(planKernel);

            // Dot product in Fourier domain
            cuOS2DCakeTransformResize<<<blocks, threads>>>( d_Dst, d_Kernel, d_ImageComplex, kernelsize, imagewidth, imageheight);

            // Transform back to spacial domain

            cufftHandle plan;
            int n[2] = {imagewidth, imageheight};
    
            cufftPlanMany(&plan, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, pieces * overlap);
            cufftExecZ2Z(plan, d_Dst, d_Dst, CUFFT_INVERSE);

            cudaDeviceSynchronize();
            cufftDestroy(plan);

            cudaFree(d_Kernel);        
        }

    cudaFree(d_ImageComplex);
    }

    else if (method == 0) // Spacial correlation
    {
        cuDoubleComplex *d_Kernel;
        cudaMalloc((void **)&d_Kernel, kernelsize * kernelsize * pieces * overlap * sizeof(cuDoubleComplex));

        cakeWaveletStack( d_Kernel, kernelsize, pieces, k, t, q, s, overlap, periodicity);

        cuOS2DCakeTransformCorrelation<<<blocks, threads>>>( d_Dst, d_Kernel, d_Image, kernelsize, imagewidth, imageheight);

        cudaFree(d_Kernel);
    }
}





