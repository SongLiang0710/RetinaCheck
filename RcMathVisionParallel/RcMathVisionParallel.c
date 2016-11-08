//
// @Author	Song Liang, Northeastern University ,CN 
//
// The code and binary file are attributed MIT license. See the detail and latest update in 
// https://github.com/SongLiang0710/RetinaCheck
//
// This copy of program is developed in Windows 7, Mathematica 11.0, CUDA Toolkit 7.5
//
// The main file deals solely with data communication, on both Mathematica-external and CPU-GPU.
// Concrete algorithms should not appear here for conciseness sake.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "RcMathVisionParallel.h"


void cudaGaussianDerivative (int nx, int ny, double sigma)
{
	double *ptr;
	int *dim;
	char **head; 
	int d;

	double *d_Input, *d_Output;

	WSGetReal64Array(stdlink, &ptr, &dim, &head, &d);

	cudaMalloc((void **)&d_Input,dim[0] * dim[1] * sizeof(double));
	cudaMalloc((void **)&d_Output,dim[0] * dim[1] * sizeof(double));

	cudaMemcpy(d_Input, ptr, dim[0] * dim[1] * sizeof(double), cudaMemcpyHostToDevice);

	gaussianDerivative2D( d_Output, d_Input, dim[1], dim[0], sigma, nx, ny);
 	
	cudaMemcpy(ptr, d_Output, dim[0] * dim[1] * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_Output);
	cudaFree(d_Input);

	WSPutReal64Array(stdlink, ptr, dim, NULL, 2);

	WSReleaseReal64Array(stdlink, ptr, dim, head, d);
}


void cudaGaugeDerivative (int nv, int nw, double sigma)
{
	double *ptr;
	int *dim;
	char **head; 
	int d;

	double *d_Input, *d_Output;

	WSGetReal64Array(stdlink, &ptr, &dim, &head, &d);

	cudaMalloc((void **)&d_Input,dim[0] * dim[1] * sizeof(double));
	cudaMalloc((void **)&d_Output,dim[0] * dim[1] * sizeof(double));

	cudaMemcpy(d_Input, ptr, dim[0] * dim[1] * sizeof(double), cudaMemcpyHostToDevice);

	gaugeDerivative2D( d_Output, d_Input, dim[1], dim[0], sigma, nw, nv);
 	
	cudaMemcpy(ptr, d_Output, dim[0] * dim[1] * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_Output);
	cudaFree(d_Input);

	WSPutReal64Array(stdlink, ptr, dim, NULL, 2);

	WSReleaseReal64Array(stdlink, ptr, dim, head, d);
}


void cudaCakeWaveletStackFourier (int size, int nc, int k, double t, int q, int ov, int periodicity)
{
	cuDoubleComplex *h_Output, *h_Scan, *d_Output;
	int ii, jj, kk;

	cudaMalloc((void **)&d_Output, nc * ov * size * size * sizeof(cuDoubleComplex));

	cakeWaveletStackFourier( d_Output, size, nc, k, t, q, ov, periodicity );

	h_Output = (cuDoubleComplex *)malloc(nc * ov * size * size * sizeof(cuDoubleComplex));

	cudaMemcpy(h_Output, d_Output, nc * ov * size * size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

	cudaFree(d_Output);
	
	// An alternative method to transmit complex number, exhibiting the actual mechanism
	h_Scan = h_Output;
	WSPutFunction(stdlink, "List", nc * ov);
	for (ii = 0; ii < nc * ov; ii++){
		WSPutFunction(stdlink, "List", size);
		for (jj = 0; jj < size; jj++){
			WSPutFunction(stdlink, "List", size);
			for (kk = 0; kk < size; kk++){
				WSPutFunction(stdlink, "Complex", 2);
				WSPutReal64(stdlink, cuCreal(h_Scan[0]));
				WSPutReal64(stdlink, cuCimag(h_Scan[0]));
				h_Scan++;
			}
		}
	}

	free(h_Output);
}


void cudaCakeWaveletStack (int size, int nc, int k, double t, int q, int s, int ov, int periodicity)
{
	cuDoubleComplex *h_Output, *d_Output;
	int dim[4] = {nc * ov, size, size, 2};
	char *head[4] = {"List", "List", "List", "Complex"}; 

	cudaMalloc((void **)&d_Output, nc * ov * size * size * sizeof(cuDoubleComplex));

	cakeWaveletStack( d_Output, size, nc, k, t, q, s, ov, periodicity);

	h_Output = (cuDoubleComplex *)malloc(nc * ov * size * size * sizeof(cuDoubleComplex));

	cudaMemcpy(h_Output, d_Output, nc * ov * size * size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

	cudaFree(d_Output);

	WSPutReal64Array(stdlink, (double *)h_Output, dim, (const char **)head, 4);

	free(h_Output);
}


void cudaOS2DCakeTransform (int size, int nc, int k, double t, int q, int s, int ov, int periodicity, int method)
{
	double *h_Input, *d_Input;
	cuDoubleComplex *h_Output, *d_Output;
	int *dim;
	char **head;
	int d;
	char *oshead[4] = {"List", "List", "List", "Complex"};

	WSGetReal64Array(stdlink, &h_Input, &dim, &head, &d);

	int osdim[4] = {nc * ov, dim[0], dim[1], 2};

	cudaMalloc((void **)&d_Input, dim[0] * dim[1] * sizeof(double));
	cudaMalloc((void **)&d_Output, osdim[0] * osdim[1] * osdim[2] * sizeof(cuDoubleComplex));

	cudaMemcpy(d_Input, h_Input, dim[0] * dim[1] * sizeof(double), cudaMemcpyHostToDevice);

	OS2DCakeTransform( d_Output, d_Input, dim[1], dim[0], size, nc, k, t, q, s, ov, periodicity, method);

	h_Output = (cuDoubleComplex *)malloc(osdim[0] * osdim[1] * osdim[2] * sizeof(cuDoubleComplex));

	cudaMemcpy(h_Output, d_Output, osdim[0] * osdim[1] * osdim[2] * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

	cudaFree(d_Output);
        cudaFree(d_Input);

	WSPutReal64Array(stdlink, (double *)h_Output, osdim, (const char **)oshead, 4);

	WSReleaseReal64Array(stdlink, h_Input, dim, head, d);

	free(h_Output);
}


void cudaOS2DGaussianDerivative (double sigmaS, double sigmaO, double mu, int order)
{
	double *ptr, *d_Input, *d_Output;
	int *dim;
	char **head; 
	int d;

	WSGetReal64Array(stdlink, &ptr, &dim, &head, &d);

	cudaMalloc((void **)&d_Input, dim[0] * dim[1] * dim[2] * sizeof(double));
	cudaMalloc((void **)&d_Output, dim[0] * dim[1] * dim[2] * sizeof(double));

	cudaMemcpy(d_Input, ptr, dim[0] * dim[1] * dim[2] * sizeof(double), cudaMemcpyHostToDevice);

	OS2DGaussianDerivative( d_Output, d_Input, dim[2], dim[1], dim[0], sigmaS, sigmaO, mu, order);

	cudaMemcpy(ptr, d_Output, dim[0] * dim[1] * dim[2] * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_Output);
        cudaFree(d_Input);

	WSPutReal64Array(stdlink, ptr, dim, NULL, 3);

	WSReleaseReal64Array(stdlink, ptr, dim, head, d);
}


void cudaOS2DGaussianHessian (double sigmaS, double sigmaO, double mu)
{
	double *h_Input, *d_Input, *h_Output, *d_Output;
	int *dim;
	char **head; 
	int d;

	WSGetReal64Array(stdlink, &h_Input, &dim, &head, &d);

	int osdim[5] = {dim[0], dim[1], dim[2], 3, 3};

	cudaMalloc((void **)&d_Input, dim[0] * dim[1] * dim[2] * sizeof(double));
	cudaMalloc((void **)&d_Output, osdim[0] * osdim[1] * osdim[2] * osdim[3] * osdim[4] * sizeof(double));

	cudaMemcpy(d_Input, h_Input, dim[0] * dim[1] * dim[2] * sizeof(double), cudaMemcpyHostToDevice);

	OS2DGaussianHessian( d_Output, d_Input, dim[2], dim[1], dim[0], sigmaS, sigmaO, mu);

	h_Output = (double *)malloc(osdim[0] * osdim[1] * osdim[2] * osdim[3] * osdim[4] * sizeof(double));

	cudaMemcpy(h_Output, d_Output, osdim[0] * osdim[1] * osdim[2] * osdim[3] * osdim[4] * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_Output);
        cudaFree(d_Input);

	WSPutReal64Array(stdlink, h_Output, osdim, NULL, 5);

	WSReleaseReal64Array(stdlink, h_Input, dim, head, d);

	free(h_Output);
}


void cudaOS2DHessianFeatures (double mu)
{
	double *h_Input, *h_Output;
	int *dim;
	char **head; 
	int d;

	matrix3by3 *d_Input;
	double3 *d_Output;

	WSGetReal64Array(stdlink, &h_Input, &dim, &head, &d);

	int osdim[4] = {dim[0], dim[1], dim[2], 3};

	cudaMalloc((void **)&d_Input, dim[0] * dim[1] * dim[2] * dim[3] * dim[4] * sizeof(double));
	cudaMalloc((void **)&d_Output, osdim[0] * osdim[1] * osdim[2] * osdim[3] * sizeof(double));

	cudaMemcpy(d_Input, h_Input, dim[0] * dim[1] * dim[2] * dim[3] * dim[4] * sizeof(double), cudaMemcpyHostToDevice);

	OS2DHessianFeatures( d_Output, d_Input, dim[2], dim[1], dim[0], mu);

	h_Output = (double *)malloc(osdim[0] * osdim[1] * osdim[2] * osdim[3] * sizeof(double));
 	
	cudaMemcpy(h_Output, d_Output, osdim[0] * osdim[1] * osdim[2] * osdim[3] * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_Output);
	cudaFree(d_Input);

	WSPutReal64Array(stdlink, h_Output, osdim, NULL, 4);

	WSReleaseReal64Array(stdlink, h_Input, dim, head, d);
	free(h_Output);
}


void cudaNonlinearDiffusionFunction (double c)
{
	double *ptr;
	int *dim;
	char **head; 
	int d;

	double *d_Input, *d_Output;

	WSGetReal64Array(stdlink, &ptr, &dim, &head, &d);

	cudaMalloc((void **)&d_Input, dim[0] * dim[1] * dim[2] * sizeof(double));
	cudaMalloc((void **)&d_Output, dim[0] * dim[1] * dim[2] * sizeof(double));

	cudaMemcpy(d_Input, ptr, dim[0] * dim[1] * dim[2] * sizeof(double), cudaMemcpyHostToDevice);

	nonlinearDiffusionFunction( d_Output, d_Input, dim[2], dim[1], dim[0], c);
 	
	cudaMemcpy(ptr, d_Output, dim[0] * dim[1] * dim[2] * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_Output);
	cudaFree(d_Input);

	WSPutReal64Array(stdlink, ptr, dim, NULL, 3);

	WSReleaseReal64Array(stdlink, ptr, dim, head, d);
}


void cudaOS2DDiffusionStepExplicit (int k, int steps, double tau)
{
	double *h_Input, *h_Output;
	int *dim;
	char **head; 
	int d;

	double *d_Input, *d_Output;

	WSGetReal64Array(stdlink, &h_Input, &dim, &head, &d);

	int osdim[3] = {dim[1], dim[2], dim[3]};

	cudaMalloc((void **)&d_Input, dim[0] * dim[1] * dim[2] * dim[3] * sizeof(double));
	cudaMalloc((void **)&d_Output, osdim[0] * osdim[1] * osdim[2] * sizeof(double));

	cudaMemcpy(d_Input, h_Input, dim[0] * dim[1] * dim[2] * dim[3] * sizeof(double), cudaMemcpyHostToDevice);

	OS2DDiffusionStepExplicit( d_Output, d_Input, dim[3], dim[2], dim[1], k, steps, tau);
 	
	h_Output = (double *)malloc(osdim[0] * osdim[1] * osdim[2] * sizeof(double));

	cudaMemcpy(h_Output, d_Output, osdim[0] * osdim[1] * osdim[2] * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_Output);
	cudaFree(d_Input);

	WSPutReal64Array(stdlink, h_Output, osdim, NULL, 3);

	WSReleaseReal64Array(stdlink, h_Input, dim, head, d);

	free(h_Output);
}


void cudaOS2DCoherenceEnhancingDiffusionStep (int k, int steps, double tau, double mu)
{
	double *h_Input, *h_Output;
	int *dim;
	char **head; 
	int d;

	double *d_Input, *d_Output;

	WSGetReal64Array(stdlink, &h_Input, &dim, &head, &d);

	int osdim[3] = {dim[1], dim[2], dim[3]};

	cudaMalloc((void **)&d_Input, dim[0] * dim[1] * dim[2] * dim[3] * sizeof(double));
	cudaMalloc((void **)&d_Output, osdim[0] * osdim[1] * osdim[2] * sizeof(double));

	cudaMemcpy(d_Input, h_Input, dim[0] * dim[1] * dim[2] * dim[3] * sizeof(double), cudaMemcpyHostToDevice);

	OS2DCoherenceEnhancingDiffusionStep( d_Output, d_Input, dim[3], dim[2], dim[1], k, steps, tau, mu);
 	
	h_Output = (double *)malloc(osdim[0] * osdim[1] * osdim[2] * sizeof(double));

	cudaMemcpy(h_Output, d_Output, osdim[0] * osdim[1] * osdim[2] * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_Output);
	cudaFree(d_Input);

	WSPutReal64Array(stdlink, h_Output, osdim, NULL, 3);

	WSReleaseReal64Array(stdlink, h_Input, dim, head, d);

	free(h_Output);
}


void cudaSE2FiniteDerivative (int order, int k)
{
	double *ptr;
	int *dim;
	char **head; 
	int d;

	double *d_Input, *d_Output;

	WSGetReal64Array(stdlink, &ptr, &dim, &head, &d);

	cudaMalloc((void **)&d_Input, dim[0] * dim[1] * dim[2] * sizeof(double));
	cudaMalloc((void **)&d_Output, dim[0] * dim[1] * dim[2] * sizeof(double));

	cudaMemcpy(d_Input, ptr, dim[0] * dim[1] * dim[2] * sizeof(double), cudaMemcpyHostToDevice);

	SE2FiniteDerivative( d_Output, d_Input, dim[2], dim[1], dim[0], order, k);
 	
	cudaMemcpy(ptr, d_Output, dim[0] * dim[1] * dim[2] * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_Output);
	cudaFree(d_Input);

	WSPutReal64Array(stdlink, ptr, dim, NULL, 3);

	WSReleaseReal64Array(stdlink, ptr, dim, head, d);
}


void cudaSE2FiniteDerivativeHessian (int k)
{
	double *h_Input, *d_Input, *h_Output;
	matrix3by3 *d_Output;

	int *dim;
	char **head; 
	int d;

	WSGetReal64Array(stdlink, &h_Input, &dim, &head, &d);

	int osdim[5] = {dim[0], dim[1], dim[2], 3, 3};

	cudaMalloc((void **)&d_Input, dim[0] * dim[1] * dim[2] * sizeof(double));
	cudaMalloc((void **)&d_Output, osdim[0] * osdim[1] * osdim[2] * osdim[3] * osdim[4] * sizeof(double));

	cudaMemcpy(d_Input, h_Input, dim[0] * dim[1] * dim[2] * sizeof(double), cudaMemcpyHostToDevice);

	SE2FiniteDerivativeHessian( d_Output, d_Input, dim[2], dim[1], dim[0], k);

	h_Output = (double *)malloc(osdim[0] * osdim[1] * osdim[2] * osdim[3] * osdim[4] * sizeof(double));

	cudaMemcpy(h_Output, d_Output, osdim[0] * osdim[1] * osdim[2] * osdim[3] * osdim[4] * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_Output);
        cudaFree(d_Input);

	WSPutReal64Array(stdlink, h_Output, osdim, NULL, 5);

	WSReleaseReal64Array(stdlink, h_Input, dim, head, d);

	free(h_Output);
}


void cudaTest (double theta, int k)
{
/*	double *ptr;
	int *dim;
	char **head; 
	int d;

	double *d_Input, *d_Output;

	WSGetReal64Array(stdlink, &ptr, &dim, &head, &d);

	cudaMalloc((void **)&d_Input, dim[0] * dim[1] * sizeof(double));
	cudaMalloc((void **)&d_Output, dim[0] * dim[1] * sizeof(double));

	cudaMemcpy(d_Input, ptr, dim[0] * dim[1] * sizeof(double), cudaMemcpyHostToDevice);

	splineTest( d_Output, d_Input, dim[1], dim[0], theta, k);
 	
	cudaMemcpy(ptr, d_Output, dim[0] * dim[1] * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_Output);
	cudaFree(d_Input);

	WSPutReal64Array(stdlink, ptr, dim, NULL, 2);

	WSReleaseReal64Array(stdlink, ptr, dim, head, d);
*/
}



#if WINDOWS_WSTP

#if __BORLANDC__
#pragma argsused
#endif

int PASCAL WinMain( HINSTANCE hinstCurrent, HINSTANCE hinstPrevious, LPSTR lpszCmdLine, int nCmdShow)
{
	char  buff[512];
	char FAR * buff_start = buff;
	char FAR * argv[32];
	char FAR * FAR * argv_end = argv + 32;

	hinstPrevious = hinstPrevious; /* suppress warning */

	if( !WSInitializeIcon( hinstCurrent, nCmdShow)) return 1;
	WSScanString( argv, &argv_end, &lpszCmdLine, &buff_start);
	return WSMain( (int)(argv_end - argv), argv);
}

#else

int main(int argc, char* argv[])
{
	return WSMain(argc, argv);
}

#endif

