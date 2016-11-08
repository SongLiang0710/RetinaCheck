
#if !defined(BSPLINE_H)
#define BSPLINE_H


typedef double (*Spline_ptr)(double);

__device__ static double deBSplineK0(double vx)
{
	if (vx > -0.5 && vx < 0.5)
		return 1;
	else 
		return 0;
}

__device__ static double deBSplineK1(double x)
{
	if (x <= 0 && x > -1)
		return 1 + x;
	else if (x > 0 && x < 1)
		return 1 - x;
	else 
		return 0;
}

__device__ static double deBSplineK2(double x)
{
	if (x <= -0.5 && x > -1.5)
		return pow(3 + 2 * x, 2) / 8;
	else if (x > -0.5 && x <= 0.5)
		return 0.75 - x*x;
	else if (x > 0.5 && x < 1.5)
		return pow(3 - 2 * x, 2) / 8;
	else 
		return 0;
}

__device__ static double deBSplineK3(double x)
{
	if (x <= -1 && x > -2)
		return pow(2 + x, 3) / 6;
	else if (x > -1 && x <= 0)
		return (4 - 3 * x*x * (2 + x)) / 6;
	else if (x > 0 && x <= 1)
		return (4 - 3 * x*x * (2 - x)) / 6;
	else if (x > 1 && x < 2)
		return pow(2 - x, 3) / 6;
	else 
		return 0;
}

__device__ static Spline_ptr set_splineFunction(int k)
{
        Spline_ptr func;
        switch (k) {
	        case 0: func = deBSplineK0; break;
        	case 1: func = deBSplineK1; break;
        	case 2: func = deBSplineK2; break;
        	case 3: func = deBSplineK3; break;
        }
        return func;
}


#endif
