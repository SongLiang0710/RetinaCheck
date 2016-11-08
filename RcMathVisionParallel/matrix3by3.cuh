
#if !defined(MATRIX3BY3_H)
#define MATRIX3BY3_H


__inline__ __device__ matrix3by3 make_matrix3by3 (double a00, double a01, double a02, double a10, double a11, double a12, double a20, double a21, double a22)
{
    matrix3by3 res = {a00, a01, a02, a10, a11, a12, a20, a21, a22};
    return res;
}

__inline__ __device__ double getEntry_matrix3by3Index (matrix3by3 &a, int n)
{
    double *ptr = (double *)&a;
    return ptr[n];
}

__inline__ __device__ void setEntry_matrix3by3Index (matrix3by3 &a, int n, double e)
{
    double *ptr = (double *)&a;
    ptr[n] = e;
}

__inline__ __device__ matrix3by3 transpose_matrix3by3 (matrix3by3 a)
{
    matrix3by3 res = {a.D00, a.D10, a.D20, a.D01, a.D11, a.D21, a.D02, a.D12, a.D22};
    return res;
}

__inline__ __device__ matrix3by3 matrixMultiply (matrix3by3 a, matrix3by3 b)
{
	return make_matrix3by3(
		a.D00 * b.D00 + a.D01 * b.D10 + a.D02 * b.D20,
		a.D00 * b.D01 + a.D01 * b.D11 + a.D02 * b.D21,
		a.D00 * b.D02 + a.D01 * b.D12 + a.D02 * b.D22,
		a.D10 * b.D00 + a.D11 * b.D10 + a.D12 * b.D20,
		a.D10 * b.D01 + a.D11 * b.D11 + a.D12 * b.D21,
		a.D10 * b.D02 + a.D11 * b.D12 + a.D12 * b.D22,
		a.D20 * b.D00 + a.D21 * b.D10 + a.D22 * b.D20,
		a.D20 * b.D01 + a.D21 * b.D11 + a.D22 * b.D21,
		a.D20 * b.D02 + a.D21 * b.D12 + a.D22 * b.D22
	);
}

__inline__ __device__ double3 matrixVectorMultiply (matrix3by3 a, double3 v)
{
	return make_double3(
		a.D00 * v.x + a.D01 * v.y + a.D02 * v.z,
		a.D10 * v.x + a.D11 * v.y + a.D12 * v.z,
		a.D20 * v.x + a.D21 * v.y + a.D22 * v.z
	);
}

__inline__ __device__ double3 vectorMatrixMultiply (double3 v, matrix3by3 a)
{
	return make_double3(
		a.D00 * v.x + a.D10 * v.y + a.D20 * v.z,
		a.D01 * v.x + a.D11 * v.y + a.D21 * v.z,
		a.D02 * v.x + a.D12 * v.y + a.D22 * v.z
	);
}

__inline__ __device__ double vectorMultiply (double3 a, double3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__inline__ __device__ matrix3by3 make_identity3by3 ()
{
    return make_matrix3by3(1, 0, 0, 0, 1, 0, 0, 0, 1);
}

__inline__ __device__ matrix3by3 make_diagonal3by3 (double a, double b, double c)
{
    return make_matrix3by3(a, 0, 0, 0, b, 0, 0, 0, c);
}

#endif
