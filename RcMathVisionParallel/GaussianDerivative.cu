//
// @Author	Song Liang, Northeastern University ,CN 
//
// The code and binary file are attributed MIT license. See the detail and latest update in 
// https://github.com/SongLiang0710/RetinaCheck
//
// This copy of program is developed in Windows 7, Mathematica 11.0, CUDA Toolkit 7.5
//
// Gaussian derivative, gauge derivative, left-invariant derivative, SE(2) Gaussian Hessian derivative

#include <math.h>
#include "RcMathVisionParallel.h"

#define LENGTH_GKERNEL 129
// Kernel storage. It seems the dimension of the constant memory cannot be a variable

__constant__ double c_k0[LENGTH_GKERNEL];
__constant__ double c_k1[LENGTH_GKERNEL];
__constant__ double c_k2[LENGTH_GKERNEL];

__constant__ double c_theta0[LENGTH_GKERNEL];
__constant__ double c_theta1[LENGTH_GKERNEL];
__constant__ double c_theta2[LENGTH_GKERNEL];

extern "C" int setGaussianKernel(double sigma, double stride)
{
	int flag = 0; // distinguish XOY and theta
	if (stride == 0) {
		stride = 1;
		flag = 1;
	}

    int radius = (int)ceil(sigma * 3 / stride);
    if (radius < 6) radius = 6;

    double *g_kernel;
    g_kernel = (double *)malloc((radius * 2 + 1) * sizeof(double));

    int i;

    for (i = -radius; i <= radius; i++)
    {
        g_kernel[i + radius] = exp(-pow(i * stride, 2) / (2*sigma*sigma)) / (sigma * sqrt(2 * PI));
    }
    if (flag == 1) cudaMemcpyToSymbol(c_k0, g_kernel, (radius * 2 + 1) * sizeof(double));
        else cudaMemcpyToSymbol(c_theta0, g_kernel, (radius * 2 + 1) * sizeof(double));

    for (i = -radius; i <= radius; i++)
    {
        g_kernel[i + radius] = -exp(-pow(i * stride, 2) / (2*sigma*sigma)) * i * stride / (pow(sigma, 3) * sqrt(2 * PI));
    }
    if (flag == 1) cudaMemcpyToSymbol(c_k1, g_kernel, (radius * 2 + 1) * sizeof(double));
        else cudaMemcpyToSymbol(c_theta1, g_kernel, (radius * 2 + 1) * sizeof(double));

    for (i = -radius; i <= radius; i++)
    {
        g_kernel[i + radius] = exp(-pow(i * stride, 2) / (2*sigma*sigma)) * (pow(i * stride, 2) - sigma*sigma) / (pow(sigma, 5) * sqrt(2 * PI));
    }
    if (flag == 1) cudaMemcpyToSymbol(c_k2, g_kernel, (radius * 2 + 1) * sizeof(double));
        else cudaMemcpyToSymbol(c_theta2, g_kernel, (radius * 2 + 1) * sizeof(double));

    free(g_kernel);

    return radius;
}

// Assuming radii of both row and column convolution are no larger than 64
// Row convolution 

#define   ROWS_BLOCKDIM_X 64
#define   ROWS_BLOCKDIM_Y 4
#define   ROWS_TILE_STEPS 4

__global__ void cuRowConvolution(
    double *d_Dst,
    double *d_Src,
    int width,
    int height,
    int radius,
    int nx
)
{
    __shared__ double s_Data[ROWS_BLOCKDIM_Y][(ROWS_TILE_STEPS + 2) * ROWS_BLOCKDIM_X];

    //Offset to the left apron edge
    const int baseX = (blockIdx.x * ROWS_TILE_STEPS - 1) * ROWS_BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

    int idz = 0;
    if (blockDim.z) idz = blockIdx.z * blockDim.z + threadIdx.z;

    d_Src += baseY * width + baseX + idz * width * height;
    d_Dst += baseY * width + baseX + idz * width * height;

    if (baseY < height)
    {
        int i;

        //Load left apron
        s_Data[threadIdx.y][threadIdx.x] = (baseX >= 0) ? d_Src[0] : 0;

        //Load main data
        for (i = 1; i < 1 + ROWS_TILE_STEPS; i++)
        {        
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX + i * ROWS_BLOCKDIM_X < width) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
        }

        //Load right apron
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX + i * ROWS_BLOCKDIM_X < width) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
    }
    __syncthreads();

    if (baseY < height)
    {
        //Locate convolution kernel
        double *c_kernel;
        if (nx == 0) c_kernel = c_k0;
        else if (nx == 1) c_kernel = c_k1;
        else if (nx == 2) c_kernel = c_k2;

        //Compute and store results
        for (int i = 1; i < 1 + ROWS_TILE_STEPS; i++)
        {
            if (baseX + i * ROWS_BLOCKDIM_X < width)
            {
                double sum = 0;

                for (int j = -radius; j <= radius; j++)
                {
                    sum += c_kernel[radius - j] * s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
                }

                d_Dst[i * ROWS_BLOCKDIM_X] = sum;
            }
        }
    }
}

// Column convolution 

#define   COLUMNS_BLOCKDIM_X 16
#define   COLUMNS_BLOCKDIM_Y 16
#define   COLUMNS_TILE_STEPS 8

__global__ void cuColumnConvolution(
    double *d_Dst,
    double *d_Src,
    int width,
    int height,
    int radius,
    int ny
)
{
    __shared__ double s_Data[COLUMNS_BLOCKDIM_X][COLUMNS_TILE_STEPS * COLUMNS_BLOCKDIM_Y + LENGTH_GKERNEL];

    //Offset to the upper apron edge
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y * COLUMNS_TILE_STEPS * COLUMNS_BLOCKDIM_Y - 64 + threadIdx.y;

    int idz = 0;
    if (blockDim.z) idz = blockIdx.z * blockDim.z + threadIdx.z;

    d_Src += baseY * width + baseX + idz * width * height;
    d_Dst += baseY * width + baseX + idz * width * height;

    if (baseX < width)
    {
        int halos = (radius + COLUMNS_BLOCKDIM_Y - 1) / COLUMNS_BLOCKDIM_Y;
        int i;

        //Load upper apron
        for (i = 4 - halos; i < 4; i++)
        {
            s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY + i * COLUMNS_BLOCKDIM_Y >= 0) ? d_Src[i * COLUMNS_BLOCKDIM_Y * width] : 0;
        }

        //Load main data
        for (i = 4; i < 4 + COLUMNS_TILE_STEPS; i++)
        {
            s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY + i * COLUMNS_BLOCKDIM_Y < height) ? d_Src[i * COLUMNS_BLOCKDIM_Y * width] : 0;
        }

        //Load lower apron
        for (i = 4 + COLUMNS_TILE_STEPS; i < 4 + COLUMNS_TILE_STEPS + halos; i++)
        {
            s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY + i * COLUMNS_BLOCKDIM_Y < height) ? d_Src[i * COLUMNS_BLOCKDIM_Y * width] : 0;
        }
    }
    __syncthreads();

    if (baseX < width)
    {
        //Locate convolution kernel
        double *c_kernel;
        if (ny == 0) c_kernel = c_k0;
        else if (ny == 1) c_kernel = c_k1;
        else if (ny == 2) c_kernel = c_k2;

        //Compute and store results
        for (int i = 4; i < 4 + COLUMNS_TILE_STEPS; i++)
        {
            if (baseY + i * COLUMNS_BLOCKDIM_Y < height)
            {
                double sum = 0;

                for (int j = -radius; j <= radius; j++)
                {
                    sum += c_kernel[radius - j] * s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
                }

                d_Dst[i * COLUMNS_BLOCKDIM_Y * width] = sum;
            }
        }
    }
}


extern "C" void gaussianDerivativeXOY(
    double *d_Dst,
    double *d_Src,
    int width,
    int height,
    int stack,
    int radius,
    int nx,
    int ny
)
{
    double *d_Buf;
    cudaMalloc((void **)&d_Buf, stack * height * width * sizeof(double));

    dim3 blockDx( iDivUp(width, ROWS_TILE_STEPS * ROWS_BLOCKDIM_X), iDivUp(height, ROWS_BLOCKDIM_Y), stack);
    dim3 threadDx( ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y, 1);

    cuRowConvolution<<<blockDx, threadDx>>>( d_Buf, d_Src, width, height, radius, nx);

    dim3 blockDy( iDivUp(width, COLUMNS_BLOCKDIM_X), iDivUp(height, COLUMNS_TILE_STEPS * COLUMNS_BLOCKDIM_Y), stack);
    dim3 threadDy( COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y, 1);

    cuColumnConvolution<<<blockDy, threadDy>>>( d_Dst, d_Buf, width, height, radius, ny);

    cudaFree(d_Buf);
}

extern "C" void gaussianDerivative2D(
    double *d_Dst,
    double *d_Src,
    int width,
    int height,
    double sigma,
    int nx,
    int ny
)
{
    int radius;

    if (nx <= 2 && ny <= 2) 
    {
        radius = setGaussianKernel(sigma, 0);
        gaussianDerivativeXOY( d_Dst, d_Src, width, height, 1, radius, nx, ny);
    }
    else  // calculate iteratively
    {
        int iter, iternx, iterny;
		iter = nx > ny ? (nx + 1) / 2 : (ny + 1) / 2;

        sigma = sqrt(sigma*sigma / iter);
        radius = setGaussianKernel(sigma, 0);

        while (iter-- > 0)
        {
	    if (nx > 2) 
	    {
	        nx -= 2;
	        iternx = 2;
	    }
	    else iternx = (iter == 0) ? nx : 0;

	    if (ny > 2) 
	    {
	        ny -= 2;
	        iterny = 2;
	    }
	    else iterny = (iter == 0) ? ny : 0;

	    gaussianDerivativeXOY( d_Src, d_Src, width, height, 1, radius, iternx, iterny);
        }
    cudaMemcpy(d_Dst, d_Src, width * height * sizeof(double), cudaMemcpyDeviceToDevice);
    }
}

#define   GAUGE_BLOCKDIM_X 32
#define   GAUGE_BLOCKDIM_Y 32

//Gauge derivative W1V0

__global__ void cuGaugeDerivativeW1V0(
    double *d_Dst,
    double *d_Dx1y0,
    double *d_Dx0y1,
    int width,
    int height
)
{
    const int idX = blockIdx.x * GAUGE_BLOCKDIM_X + threadIdx.x;
    const int idY = blockIdx.y * GAUGE_BLOCKDIM_Y + threadIdx.y;
    int id = idX + idY * width;
    if (blockIdx.z) id += blockIdx.z * width * height;

    if (idX < width && idY < height){
        d_Dst[id] = hypot(d_Dx1y0[id], d_Dx0y1[id]);
    }
}

extern "C" void gaugeDerivativeW1V0(
    double *d_Dst,
    double *d_Src,
    int width,
    int height,
    int stack,
    int radius
)
{
    double *d_Dx1y0;
    double *d_Dx0y1;

    cudaMalloc((void **)&d_Dx1y0, stack * height * width * sizeof(double));
    cudaMalloc((void **)&d_Dx0y1, stack * height * width * sizeof(double));

    gaussianDerivativeXOY( d_Dx1y0, d_Src, width, height, stack, radius, 1, 0);
    gaussianDerivativeXOY( d_Dx0y1, d_Src, width, height, stack, radius, 0, 1);

    dim3 blocks( iDivUp(width, GAUGE_BLOCKDIM_X), iDivUp(height, GAUGE_BLOCKDIM_Y), stack);
    dim3 threads( GAUGE_BLOCKDIM_X, GAUGE_BLOCKDIM_Y, 1);

    cuGaugeDerivativeW1V0<<<blocks, threads>>>( d_Dst, d_Dx1y0, d_Dx0y1, width, height);

    cudaFree(d_Dx1y0);
    cudaFree(d_Dx0y1);
}

//Gauge derivative in XOY plane

__device__ double deGaugeDerivativeW2V0(double Lx, double Ly, double Lxx, double Lxy, double Lyy){
	return (Lx * Lx * Lxx + 2 * Lx * Lxy * Ly + Ly * Ly * Lyy) / (Lx * Lx + Ly * Ly);
}

__device__ double deGaugeDerivativeW0V2(double Lx, double Ly, double Lxx, double Lxy, double Lyy){
	return (Lx * Lx * Lyy - 2 * Lx * Lxy * Ly + Ly * Ly * Lxx) / (Lx * Lx + Ly * Ly);
}

__device__ double deGaugeDerivativeW1V1(double Lx, double Ly, double Lxx, double Lxy, double Lyy){
	return (Ly * Ly * Lxy - Lx * Lx * Lxy + Lx * Ly * Lxx - Lx * Ly * Lyy) / (Lx * Lx + Ly * Ly);
}

__global__ void cuGaugeDerivativeXOY(
    double *d_Dst,
    double *d_Dx1y0,
    double *d_Dx0y1,
    double *d_Dx2y0,
    double *d_Dx1y1,
    double *d_Dx0y2,
    int width,
    int height,
    int dWdV
)
{
    const int idX = blockIdx.x * GAUGE_BLOCKDIM_X + threadIdx.x;
    const int idY = blockIdx.y * GAUGE_BLOCKDIM_Y + threadIdx.y;
    int id = idX + idY * width;
    if (blockIdx.z) id += blockIdx.z * width * height;

    if (idX < width && idY < height)
    {
        double (*deDerivativeXOY)(double Lx, double Ly, double Lxx, double Lxy, double Lyy);

	if (dWdV == 2) deDerivativeXOY = deGaugeDerivativeW0V2;
	else if (dWdV == 20) deDerivativeXOY = deGaugeDerivativeW2V0;
	else if (dWdV == 11) deDerivativeXOY = deGaugeDerivativeW1V1;

        d_Dst[id] = deDerivativeXOY(d_Dx1y0[id], d_Dx0y1[id], d_Dx2y0[id], d_Dx1y1[id], d_Dx0y2[id]);
    }
}

extern "C" void gaugeDerivativeXOY(
    double *d_Dst,
    double *d_Src,
    int width,
    int height,
    int stack,
    int radius,
    int dWdV
)
{
    double *d_Dx1y0, *d_Dx0y1, *d_Dx2y0, *d_Dx1y1, *d_Dx0y2;

    cudaMalloc((void **)&d_Dx1y0, stack * height * width * sizeof(double));
    cudaMalloc((void **)&d_Dx0y1, stack * height * width * sizeof(double));
    cudaMalloc((void **)&d_Dx2y0, stack * height * width * sizeof(double));
    cudaMalloc((void **)&d_Dx1y1, stack * height * width * sizeof(double));
    cudaMalloc((void **)&d_Dx0y2, stack * height * width * sizeof(double));

    gaussianDerivativeXOY( d_Dx1y0, d_Src, width, height, stack, radius, 1, 0);
    gaussianDerivativeXOY( d_Dx0y1, d_Src, width, height, stack, radius, 0, 1);
    gaussianDerivativeXOY( d_Dx2y0, d_Src, width, height, stack, radius, 2, 0);
    gaussianDerivativeXOY( d_Dx1y1, d_Src, width, height, stack, radius, 1, 1);
    gaussianDerivativeXOY( d_Dx0y2, d_Src, width, height, stack, radius, 0, 2);

    dim3 blocks( iDivUp(width, GAUGE_BLOCKDIM_X), iDivUp(height, GAUGE_BLOCKDIM_Y), stack);
    dim3 threads( GAUGE_BLOCKDIM_X, GAUGE_BLOCKDIM_Y, 1);

    cuGaugeDerivativeXOY<<<blocks, threads>>>( d_Dst, d_Dx1y0, d_Dx0y1, d_Dx2y0, d_Dx1y1, d_Dx0y2,  width, height, dWdV);

    cudaFree(d_Dx1y0);
    cudaFree(d_Dx0y1);
    cudaFree(d_Dx2y0);
    cudaFree(d_Dx1y1);
    cudaFree(d_Dx0y2);
}

extern "C" void gaugeDerivative2D(
    double *d_Dst,
    double *d_Src,
    int width,
    int height,
    double sigma,
    int nw,
    int nv
)
{
    int radius = setGaussianKernel(sigma, 0);

    if (nw == 0 && nv == 0)
        gaussianDerivativeXOY( d_Dst, d_Src, width, height, 1, radius, 0, 0);

    else if (nw == 1 && nv == 0)
        gaugeDerivativeW1V0( d_Dst, d_Src, width, height, 1, radius);

    else 
    {
        int dWdV = nw * 10 + nv;
        gaugeDerivativeXOY( d_Dst, d_Src, width, height, 1, radius, dWdV);
    }
}

// Orientation convolution 

#define   ORIENTATIONS_BLOCKDIM_X 16
//#define   ORIENTATIONS_BLOCKDIM_Y 1
//#define   ORIENTATIONS_BLOCKDIM_Z 32

__global__ void cuOrientationConvolution(
    double *d_Dst,
    double *d_Src,
    int width,
    int height,
    int stack,
    int radius,
    int nt
)
{
    __shared__ double s_Data[ORIENTATIONS_BLOCKDIM_X][LENGTH_GKERNEL];

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;
    const int id = idx + idy * width + idz * width * height;

    //Assuming blockDim.z == stack >= radius, stack < LENGTH_GKERNEL / 3

    if (idx < width)
    {
        s_Data[threadIdx.x][threadIdx.z] = (idz >= blockDim.z - radius) ? d_Src[id] : 0;

        s_Data[threadIdx.x][threadIdx.z + blockDim.z] = d_Src[id];

        s_Data[threadIdx.x][threadIdx.z + 2 * blockDim.z] = (threadIdx.z < radius) ? d_Src[id] : 0;
    }
    __syncthreads();

    if (idx < width)
    {
        //Locate convolution kernel
        double *c_kernel;
        if (nt == 0) c_kernel = c_theta0;
    	else if (nt == 1) c_kernel = c_theta1;
        else if (nt == 2) c_kernel = c_theta2;

        //Compute and store results
        if (idz < stack)
        {
            double sum = 0;
            for (int j = -radius; j <= radius; j++)
            {
                sum += c_kernel[radius - j] * s_Data[threadIdx.x][threadIdx.z + blockDim.z + j];
            }

            d_Dst[id] = sum * PI / stack; // (when overlap = 1)
        }
    }
}

// Left-invariant derivatives, implemented by Gaussian derivatives 

// Basic Gaussian derivative, meanwhile taken up immediately as -Type1: orientational derivative only

extern "C" void GaussianDerivativeSE2(
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
)
{
    double *d_Buf;
    cudaMalloc((void **)&d_Buf, stack * height * width * sizeof(double));

    gaussianDerivativeXOY( d_Buf, d_Src, width, height, stack, radiusS, nx, ny);

    dim3 blocks( iDivUp(width, ORIENTATIONS_BLOCKDIM_X), height, 1);
    dim3 threads( ORIENTATIONS_BLOCKDIM_X, 1, stack);

    cuOrientationConvolution<<<blocks, threads>>>( d_Dst, d_Buf, width, height, stack, radiusO, nt);

    cudaFree(d_Buf);
}

// -Type2: Lwt & Lvt --derivative with respect to orientation at first

__device__ double deGaussianLeftInvariantLwt(double Lxt, double Lyt, double theta){
	return Lxt * cos(theta) + Lyt * sin(theta);
}

__device__ double deGaussianLeftInvariantLvt(double Lxt, double Lyt, double theta){
	return Lyt * cos(theta) - Lxt * sin(theta);
}

__global__ void cuGaussianLeftInvariantXYOT(
    double *d_Dst,
    double *d_Dx1y0t1,
    double *d_Dx0y1t1,
    int width,
    int height,
    int stack,
    double mu,
    int dWdV
)
{
    const int idX = blockIdx.x * blockDim.x + threadIdx.x;
    const int idY = blockIdx.y * blockDim.y + threadIdx.y;
    const int idZ = blockIdx.z * blockDim.z + threadIdx.z;
    const int id = idX + idY * width + idZ * width * height;

    if (idX < width && idY < height)
    {
        double (*deDerivativeLeftI)(double Lxt, double Lyt, double theta);

	    if (dWdV == 10) deDerivativeLeftI = deGaussianLeftInvariantLwt;
	    else if (dWdV == 1) deDerivativeLeftI = deGaussianLeftInvariantLvt;

        d_Dst[id] = deDerivativeLeftI(d_Dx1y0t1[id], d_Dx0y1t1[id], idZ * PI / stack) / mu;
    }
}

extern "C" void GaussianLeftInvariantXYOT(
    double *d_Dst,
    double *d_Src,
    int width,
    int height,
    int stack,
    int radiusS,
    int radiusO,
    double mu,
    int dWdV
)
{
    double *d_Dx1y0t1, *d_Dx0y1t1;
    cudaMalloc((void **)&d_Dx1y0t1, stack * height * width * sizeof(double));
    cudaMalloc((void **)&d_Dx0y1t1, stack * height * width * sizeof(double));

	GaussianDerivativeSE2( d_Dx1y0t1, d_Src, width, height, stack, radiusS, radiusO, 1, 0, 1);
	GaussianDerivativeSE2( d_Dx0y1t1, d_Src, width, height, stack, radiusS, radiusO, 0, 1, 1);

    dim3 blocks( iDivUp(width, DEFAULT_BLOCKDIM_X), iDivUp(height, DEFAULT_BLOCKDIM_Y), iDivUp(stack, DEFAULT_BLOCKDIM_Z));
    dim3 threads( DEFAULT_BLOCKDIM_X, DEFAULT_BLOCKDIM_Y, DEFAULT_BLOCKDIM_Z);

    cuGaussianLeftInvariantXYOT<<<blocks, threads>>>( d_Dst, d_Dx1y0t1, d_Dx0y1t1, width, height, stack, mu, dWdV);

    cudaFree(d_Dx1y0t1);
    cudaFree(d_Dx0y1t1);
}

// -Type3: Ltw & Ltv --derivative with respect to orientation at last

__device__ double deGaussianLeftInvariantLtw(double Lxt, double Lyt, double Lx, double Ly, double theta){
	return (Lxt + Ly) * cos(theta) + (Lyt - Lx) * sin(theta);
}

__device__ double deGaussianLeftInvariantLtv(double Lxt, double Lyt, double Lx, double Ly, double theta){
	return (Lyt - Lx) * cos(theta) - (Lxt + Ly) * sin(theta);
}

__global__ void cuGaussianLeftInvariantTOXY(
    double *d_Dst,
    double *d_Dx1y0t1,
    double *d_Dx0y1t1,
    double *d_Dx1y0t0,
    double *d_Dx0y1t0,
    int width,
    int height,
    int stack,
    double mu,
    int dWdV
)
{
    const int idX = blockIdx.x * blockDim.x + threadIdx.x;
    const int idY = blockIdx.y * blockDim.y + threadIdx.y;
    const int idZ = blockIdx.z * blockDim.z + threadIdx.z;
    const int id = idX + idY * width + idZ * width * height;

    if (idX < width && idY < height)
    {
        double (*deDerivativeLeftI)(double Lxt, double Lyt, double Lx, double Ly, double theta);

	    if (dWdV == 10) deDerivativeLeftI = deGaussianLeftInvariantLtw;
	    else if (dWdV == 1) deDerivativeLeftI = deGaussianLeftInvariantLtv;

        d_Dst[id] = deDerivativeLeftI(d_Dx1y0t1[id], d_Dx0y1t1[id], d_Dx1y0t0[id], d_Dx0y1t0[id], idZ * PI / stack) / mu;
    }
}

extern "C" void GaussianLeftInvariantTOXY(
    double *d_Dst,
    double *d_Src,
    int width,
    int height,
    int stack,
    int radiusS,
    int radiusO,
    double mu,
    int dWdV
)
{
    double *d_Dx1y0t1, *d_Dx0y1t1, *d_Dx1y0t0, *d_Dx0y1t0;

    cudaMalloc((void **)&d_Dx1y0t1, stack * height * width * sizeof(double));
    cudaMalloc((void **)&d_Dx0y1t1, stack * height * width * sizeof(double));
    cudaMalloc((void **)&d_Dx1y0t0, stack * height * width * sizeof(double));
    cudaMalloc((void **)&d_Dx0y1t0, stack * height * width * sizeof(double));

	GaussianDerivativeSE2( d_Dx1y0t1, d_Src, width, height, stack, radiusS, radiusO, 1, 0, 1);
	GaussianDerivativeSE2( d_Dx0y1t1, d_Src, width, height, stack, radiusS, radiusO, 0, 1, 1);
	GaussianDerivativeSE2( d_Dx1y0t0, d_Src, width, height, stack, radiusS, radiusO, 1, 0, 0);
	GaussianDerivativeSE2( d_Dx0y1t0, d_Src, width, height, stack, radiusS, radiusO, 0, 1, 0);

    dim3 blocks( iDivUp(width, DEFAULT_BLOCKDIM_X), iDivUp(height, DEFAULT_BLOCKDIM_Y), iDivUp(stack, DEFAULT_BLOCKDIM_Z));
    dim3 threads( DEFAULT_BLOCKDIM_X, DEFAULT_BLOCKDIM_Y, DEFAULT_BLOCKDIM_Z);

    cuGaussianLeftInvariantTOXY<<<blocks, threads>>>( d_Dst, d_Dx1y0t1, d_Dx0y1t1, d_Dx1y0t0, d_Dx0y1t0, width, height, stack, mu, dWdV);

    cudaFree(d_Dx1y0t1);
    cudaFree(d_Dx0y1t1);
    cudaFree(d_Dx1y0t0);
    cudaFree(d_Dx0y1t0);
}

// -Type4: Lww & Lwv & Lvv --no orientationals

__device__ double deGaussianLeftInvariantW2V0(double Lxx, double Lxy, double Lyy, double theta){
	double c, s;
	c = cos(theta);
	s = sin(theta);
	 
	return c * c * Lxx + 2 * c * s * Lxy + s * s * Lyy;
}

__device__ double deGaussianLeftInvariantW0V2(double Lxx, double Lxy, double Lyy, double theta){
	double c, s;
	c = cos(theta);
	s = sin(theta);
	 
	return s * s * Lxx - 2 * c * s * Lxy + c * c * Lyy;
}

__device__ double deGaussianLeftInvariantW1V1(double Lxx, double Lxy, double Lyy, double theta){
	double c, s;
	c = cos(theta);
	s = sin(theta);
	 
	return -c * s * Lxx + c * c * Lxy - s * s * Lxy + c * s * Lyy;
}

__global__ void cuGaussianLeftInvariantXOY(
    double *d_Dst,
    double *d_Dx2y0t0,
    double *d_Dx1y1t0,
    double *d_Dx0y2t0,
    int width,
    int height,
    int stack,
    double mu,
    int dWdV
)
{
    const int idX = blockIdx.x * blockDim.x + threadIdx.x;
    const int idY = blockIdx.y * blockDim.y + threadIdx.y;
    const int idZ = blockIdx.z * blockDim.z + threadIdx.z;
    const int id = idX + idY * width + idZ * width * height;

    if (idX < width && idY < height)
    {
        double (*deDerivativeLeftI)(double Lxx, double Lxy, double Lyy, double theta);

	    if (dWdV == 11) deDerivativeLeftI = deGaussianLeftInvariantW1V1;
	    else if (dWdV == 20) deDerivativeLeftI = deGaussianLeftInvariantW2V0;
	    else if (dWdV == 2) deDerivativeLeftI = deGaussianLeftInvariantW0V2;

        d_Dst[id] = deDerivativeLeftI(d_Dx2y0t0[id], d_Dx1y1t0[id], d_Dx0y2t0[id], idZ * PI / stack) / (mu*mu);
    }
}

extern "C" void GaussianLeftInvariantXOY(
    double *d_Dst,
    double *d_Src,
    int width,
    int height,
    int stack,
    int radiusS,
    int radiusO,
    double mu,
    int dWdV
)
{
    double *d_Dx2y0t0, *d_Dx1y1t0, *d_Dx0y2t0;

    cudaMalloc((void **)&d_Dx2y0t0, stack * height * width * sizeof(double));
    cudaMalloc((void **)&d_Dx1y1t0, stack * height * width * sizeof(double));
    cudaMalloc((void **)&d_Dx0y2t0, stack * height * width * sizeof(double));

	GaussianDerivativeSE2( d_Dx2y0t0, d_Src, width, height, stack, radiusS, radiusO, 2, 0, 0);
	GaussianDerivativeSE2( d_Dx1y1t0, d_Src, width, height, stack, radiusS, radiusO, 1, 1, 0);
	GaussianDerivativeSE2( d_Dx0y2t0, d_Src, width, height, stack, radiusS, radiusO, 0, 2, 0);

    dim3 blocks( iDivUp(width, DEFAULT_BLOCKDIM_X), iDivUp(height, DEFAULT_BLOCKDIM_Y), iDivUp(stack, DEFAULT_BLOCKDIM_Z));
    dim3 threads( DEFAULT_BLOCKDIM_X, DEFAULT_BLOCKDIM_Y, DEFAULT_BLOCKDIM_Z);

    cuGaussianLeftInvariantXOY<<<blocks, threads>>>( d_Dst, d_Dx2y0t0, d_Dx1y1t0, d_Dx0y2t0, width, height, stack, mu, dWdV);

    cudaFree(d_Dx2y0t0);
    cudaFree(d_Dx1y1t0);
    cudaFree(d_Dx0y2t0);
}

extern "C" void OS2DGaussianDerivative(
    double *d_Dst,
    double *d_Src,
    int width,
    int height,
    int stack,
    double sigmaS,
    double sigmaO,
    double mu,
    int order
)
{
    int radiusS = setGaussianKernel(sigmaS, 0); 

    // Assuming angular overlap equals 1 everywhere below
    int radiusO = setGaussianKernel(sigmaO, PI / stack); 

    switch (order)
    {
        case 11: GaussianDerivativeSE2( d_Dst, d_Src, width, height, stack, radiusS, radiusO, 0, 0, 2); break;
        case 12: GaussianLeftInvariantXYOT( d_Dst, d_Src, width, height, stack, radiusS, radiusO, mu, 10); break;
        case 13: GaussianLeftInvariantXYOT( d_Dst, d_Src, width, height, stack, radiusS, radiusO, mu, 1); break;
        case 21: GaussianLeftInvariantTOXY( d_Dst, d_Src, width, height, stack, radiusS, radiusO, mu, 10); break;
        case 22: GaussianLeftInvariantXOY( d_Dst, d_Src, width, height, stack, radiusS, radiusO, mu, 20); break;
        case 23: GaussianLeftInvariantXOY( d_Dst, d_Src, width, height, stack, radiusS, radiusO, mu, 11); break;
        case 31: GaussianLeftInvariantTOXY( d_Dst, d_Src, width, height, stack, radiusS, radiusO, mu, 1); break;
        case 32: GaussianLeftInvariantXOY( d_Dst, d_Src, width, height, stack, radiusS, radiusO, mu, 11); break;
        case 33: GaussianLeftInvariantXOY( d_Dst, d_Src, width, height, stack, radiusS, radiusO, mu, 2); break;
    }
}

// SE2 Hessian matrix. Get each blob consequently. Then transpose to proper form

__global__ void cuOS2DGaussianHessianTranspose(
    matrix3by3 *d_Dst,
    double *d_Src,
    int width,
    int height,
    int stack
)
{
    const int idX = blockIdx.x * blockDim.x + threadIdx.x;
    const int idY = blockIdx.y * blockDim.y + threadIdx.y;
    const int idZ = blockIdx.z * blockDim.z + threadIdx.z;
    const int id = idX + idY * width + idZ * width * height;
    const int blob = stack * height * width;

	if (idX < width && idY < height)
	{
	    matrix3by3 res = {d_Src[id], d_Src[id + blob], d_Src[id + 2 * blob], d_Src[id + 3 * blob], d_Src[id + 4 * blob], d_Src[id + 5 * blob], d_Src[id + 6 * blob], d_Src[id + 7 * blob], d_Src[id + 8 * blob]};
	    d_Dst[id] = res;
	} 
}

extern "C" void OS2DGaussianHessian(
    double *d_Dst,
    double *d_Src,
    int width,
    int height,
    int stack,
    double sigmaS,
    double sigmaO,
    double mu
)
{
    int radiusS = setGaussianKernel(sigmaS, 0); 

    // Assuming angular overlap equals 1 everywhere below
    int radiusO = setGaussianKernel(sigmaO, PI / stack); 
	
    int blob = stack * height * width;

	GaussianDerivativeSE2( d_Dst, d_Src, width, height, stack, radiusS, radiusO, 0, 0, 2);
	GaussianLeftInvariantXYOT( d_Dst + blob, d_Src, width, height, stack, radiusS, radiusO, mu, 10);
	GaussianLeftInvariantXYOT( d_Dst + 2 * blob, d_Src, width, height, stack, radiusS, radiusO, mu, 1);
	GaussianLeftInvariantTOXY( d_Dst + 3 * blob, d_Src, width, height, stack, radiusS, radiusO, mu, 10);
	GaussianLeftInvariantXOY( d_Dst + 4 * blob, d_Src, width, height, stack, radiusS, radiusO, mu, 20);
	GaussianLeftInvariantXOY( d_Dst + 5 * blob, d_Src, width, height, stack, radiusS, radiusO, mu, 11);
	GaussianLeftInvariantTOXY( d_Dst + 6 * blob, d_Src, width, height, stack, radiusS, radiusO, mu, 1);
	cudaMemcpy(d_Dst + 7 * blob, d_Dst + 5 * blob, blob * sizeof(double), cudaMemcpyDeviceToDevice);
	GaussianLeftInvariantXOY( d_Dst + 8 * blob, d_Src, width, height, stack, radiusS, radiusO, mu, 2);

    cudaDeviceSynchronize();

    // The postpositional memory operation is aimed to save the occupation of memory, at the price of one additional operation. This has yet to be optimized
    matrix3by3 *d_Buf;
    cudaMalloc((void **)&d_Buf, blob * sizeof(matrix3by3));

    dim3 blocks( iDivUp(width, DEFAULT_BLOCKDIM_X), iDivUp(height, DEFAULT_BLOCKDIM_Y), iDivUp(stack, DEFAULT_BLOCKDIM_Z));
    dim3 threads( DEFAULT_BLOCKDIM_X, DEFAULT_BLOCKDIM_Y, DEFAULT_BLOCKDIM_Z);

    cuOS2DGaussianHessianTranspose<<<blocks, threads>>>( d_Buf, d_Dst, width, height, stack);

    cudaMemcpy(d_Dst, d_Buf, blob * sizeof(matrix3by3), cudaMemcpyDeviceToDevice);

    cudaFree(d_Buf);
}


