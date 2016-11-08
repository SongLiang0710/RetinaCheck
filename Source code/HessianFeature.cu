//
// @Author	Song Liang, Northeastern University ,CN 
//
// The code and binary file are attributed MIT license. See the detail and latest update in 
// https://github.com/SongLiang0710/RetinaCheck
//
// This copy of program is developed in Windows 7, Mathematica 11.0, CUDA Toolkit 7.5
//
// Hessian features including curvature, confidence and deviation from horizontality.

#include <math.h>
#include "RcMathVisionParallel.h"

#include "matrix3by3.cuh"

// Jacobian iterative method --obtaining eigenvectors of symmetric 3 by 3 matrix

//#define EPSILON 0.000000000001  (the early-stopping strategy is prone to make errors and provide little improvement in speed)

__device__ void jacobianEigenvector (double3 *eVector, matrix3by3 sHessian)
{
	matrix3by3 mEigen, mInner;

	double tmp, phi, c, s; 
	int i, j, row = 0, col = 0, iter = 0;

	mEigen = make_identity3by3 ();

	while (iter++ < 100) 
	{
		tmp = 0;
		for (i = 0; i <= 1; i++) {
			for (j = i + 1; j <= 2; j++) {
				if (fabs(getEntry_matrix3by3Index(sHessian, 3 * i + j)) > tmp) {
					row = i; 
					col = j;
					tmp = fabs(getEntry_matrix3by3Index(sHessian, 3 * i + j));
				}
			}
		}
		if (tmp == 0) break;

		phi = -atan2(2 * getEntry_matrix3by3Index(sHessian, 3 * row + col), getEntry_matrix3by3Index(sHessian, 3 * col + col) - getEntry_matrix3by3Index(sHessian, 3 * row + row)) / 2;
		c = cos(phi);
		s = sin(phi);

		mInner = make_identity3by3 ();
		setEntry_matrix3by3Index(mInner, 3 * row + row, c);
		setEntry_matrix3by3Index(mInner, 3 * row + col, s);
		setEntry_matrix3by3Index(mInner, 3 * col + row, -s);
		setEntry_matrix3by3Index(mInner, 3 * col + col, c);

		sHessian = matrixMultiply(mInner, sHessian);

		setEntry_matrix3by3Index(mInner, 3 * row + col, -s);
		setEntry_matrix3by3Index(mInner, 3 * col + row, s);

		sHessian = matrixMultiply(sHessian, mInner);
		mEigen = matrixMultiply(mEigen, mInner);
	}

	// Arrange the eigenvector corresponding to the lowest eigenvalue at first
	tmp = 9999;
	for (i = 0; i < 3; i++)
	{ 
		if (fabs(getEntry_matrix3by3Index(sHessian, 3 * i + i)) < tmp)
		{
			col = i;
			tmp = fabs(getEntry_matrix3by3Index(sHessian, 3 * i + i));
		}
	}
	for (i = 0; i < 3; i++)
	{ 
//		tmp = rsqrt(pow(getEntry_matrix3by3Index(mEigen, col), 2) + pow(getEntry_matrix3by3Index(mEigen, col + 3), 2) + pow(getEntry_matrix3by3Index(mEigen, col + 6), 2));
		eVector[i] = make_double3(getEntry_matrix3by3Index(mEigen, col), getEntry_matrix3by3Index(mEigen, col + 3), getEntry_matrix3by3Index(mEigen, col + 6));
		col = (col + 1) % 3;
	}
}

__global__ void cuHessianFeatures(
    double3 *d_Dst,
    matrix3by3 *d_Src,
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
        double3 eigenVectors[3];

	matrix3by3 muMatrix = make_diagonal3by3(1, mu, mu);
        matrix3by3 Hessian = matrixMultiply(matrixMultiply(muMatrix, d_Src[id]), muMatrix);

        jacobianEigenvector ( eigenVectors, matrixMultiply(transpose_matrix3by3(Hessian), Hessian));

        // Curvature
        d_Dst[id].x = eigenVectors[0].x * copysign(1., eigenVectors[0].y) * rhypot(eigenVectors[0].y,eigenVectors[0].z);

        // Confidence
        d_Dst[id].y = -vectorMultiply(vectorMatrixMultiply(eigenVectors[1], Hessian), eigenVectors[1])-vectorMultiply(vectorMatrixMultiply(eigenVectors[2], Hessian), eigenVectors[2]); 

        // Deviation from horizontality
        d_Dst[id].z = atan(eigenVectors[0].z / eigenVectors[0].y);
    }
}

extern "C" void OS2DHessianFeatures(
    double3 *d_Dst,
    matrix3by3 *d_Src,
    int width,
    int height,
    int stack,
    double mu
)
{
    dim3 blocks( iDivUp(width, DEFAULT_BLOCKDIM_X), iDivUp(height, DEFAULT_BLOCKDIM_Y), iDivUp(stack,  DEFAULT_BLOCKDIM_Z));
    dim3 threads( DEFAULT_BLOCKDIM_X, DEFAULT_BLOCKDIM_Y, DEFAULT_BLOCKDIM_Z);

    cuHessianFeatures<<<blocks, threads>>>( d_Dst, d_Src, width, height, stack, mu);
}

// A deprecated test function
__global__ void cuJacobianEigenvector(
    double3 *d_Dst,
    matrix3by3 *d_Src
)
{
    jacobianEigenvector (d_Dst, *d_Src);
}


extern "C" void jEigenvector(
    matrix3by3 *d_Dst,
    matrix3by3 *d_Src
)
{
    dim3 blocks(1, 1);
    dim3 threads(1, 1);

    cuJacobianEigenvector<<<blocks, threads>>>( (double3 *)d_Dst, d_Src);
}

