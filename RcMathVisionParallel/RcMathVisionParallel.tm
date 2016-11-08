
void cudaGaussianDerivative P(( int, int, double));

:Begin:
:Function:       cudaGaussianDerivative
:Pattern:        cudaGaussianDerivative[nx_Integer, ny_Integer, sigma_Real, data_List]
:Arguments:      { nx, ny, sigma, data }
:ArgumentTypes:  { Integer, Integer, Real, Manual }
:ReturnType:     Manual
:End:


void cudaGaugeDerivative P(( int, int, double));

:Begin:
:Function:       cudaGaugeDerivative
:Pattern:        cudaGaugeDerivative[nv_Integer, nw_Integer, sigma_Real, data_List]
:Arguments:      { nv, nw, sigma, data }
:ArgumentTypes:  { Integer, Integer, Real, Manual }
:ReturnType:     Manual
:End:


void cudaCakeWaveletStackFourier P(( int, int, int, double, int, int, int));

:Begin:
:Function:       cudaCakeWaveletStackFourier
:Pattern:        cudaCakeWaveletStackFourier[size_Integer, nc_Integer, k_Integer, t_Real, q_Integer, ov_Integer, periodicity_Integer]
:Arguments:      { size, nc, k, t, q, ov, periodicity }
:ArgumentTypes:  { Integer, Integer, Integer, Real, Integer, Integer, Integer }
:ReturnType:     Manual
:End:


void cudaCakeWaveletStack P(( int, int, int, double, int, int, int, int));

:Begin:
:Function:       cudaCakeWaveletStack
:Pattern:        cudaCakeWaveletStack[size_Integer, nc_Integer, k_Integer, t_Real, q_Integer, s_Integer, ov_Integer, periodicity_Integer]
:Arguments:      { size, nc, k, t, q, s, ov, periodicity }
:ArgumentTypes:  { Integer, Integer, Integer, Real, Integer, Integer, Integer, Integer }
:ReturnType:     Manual
:End:


void cudaOS2DCakeTransform P(( int, int, int, double, int, int, int, int, int));

:Begin:
:Function:       cudaOS2DCakeTransform
:Pattern:        cudaOS2DCakeTransform[size_Integer, nc_Integer, k_Integer, t_Real, q_Integer, s_Integer, ov_Integer, periodicity_Integer, method_Integer, data_List]
:Arguments:      { size, nc, k, t, q, s, ov, periodicity, method, data }
:ArgumentTypes:  { Integer, Integer, Integer, Real, Integer, Integer, Integer, Integer, Integer, Manual }
:ReturnType:     Manual
:End:


void cudaOS2DGaussianDerivative P(( double, double, double, int));

:Begin:
:Function:       cudaOS2DGaussianDerivative
:Pattern:        cudaOS2DGaussianDerivative[sigmaS_Real, sigmaO_Real, mu_Real, order_Integer, data_List]
:Arguments:      { sigmaS, sigmaO, mu, order, data }
:ArgumentTypes:  { Real, Real, Real, Integer, Manual }
:ReturnType:     Manual
:End:


void cudaOS2DGaussianHessian P(( double, double, double));

:Begin:
:Function:       cudaOS2DGaussianHessian
:Pattern:        cudaOS2DGaussianHessian[sigmaS_Real, sigmaO_Real, mu_Real, data_List]
:Arguments:      { sigmaS, sigmaO, mu, data }
:ArgumentTypes:  { Real, Real, Real, Manual }
:ReturnType:     Manual
:End:


void cudaOS2DHessianFeatures P(( double));

:Begin:
:Function:       cudaOS2DHessianFeatures
:Pattern:        cudaOS2DHessianFeatures[mu_Real, data_List]
:Arguments:      { mu, data }
:ArgumentTypes:  { Real, Manual }
:ReturnType:     Manual
:End:


void cudaNonlinearDiffusionFunction P(( double));

:Begin:
:Function:       cudaNonlinearDiffusionFunction
:Pattern:        cudaNonlinearDiffusionFunction[c_Real, data_List]
:Arguments:      { c, data }
:ArgumentTypes:  { Real, Manual }
:ReturnType:     Manual
:End:


void cudaOS2DDiffusionStepExplicit P(( int, int, double));

:Begin:
:Function:       cudaOS2DDiffusionStepExplicit
:Pattern:        cudaOS2DDiffusionStepExplicit[k_Integer, steps_Integer, tau_Real, data_List]
:Arguments:      { k, steps, tau, data }
:ArgumentTypes:  { Integer, Integer, Real, Manual }
:ReturnType:     Manual
:End:


void cudaOS2DCoherenceEnhancingDiffusionStep P(( int, int, double, double));

:Begin:
:Function:       cudaOS2DCoherenceEnhancingDiffusionStep
:Pattern:        cudaOS2DCoherenceEnhancingDiffusionStep[k_Integer, steps_Integer, tau_Real, mu_Real, data_List]
:Arguments:      { k, steps, tau, mu, data }
:ArgumentTypes:  { Integer, Integer, Real, Real, Manual }
:ReturnType:     Manual
:End:


void cudaSE2FiniteDerivative P(( int, int));

:Begin:
:Function:       cudaSE2FiniteDerivative
:Pattern:        cudaSE2FiniteDerivative[order_Integer, k_Integer, data_List]
:Arguments:      { order, k, data }
:ArgumentTypes:  { Integer, Integer, Manual }
:ReturnType:     Manual
:End:


void cudaSE2FiniteDerivativeHessian P(( int));

:Begin:
:Function:       cudaSE2FiniteDerivativeHessian
:Pattern:        cudaSE2FiniteDerivativeHessian[k_Integer, data_List]
:Arguments:      { k, data }
:ArgumentTypes:  { Integer, Manual }
:ReturnType:     Manual
:End:


void cudaTest P(( double, int));

:Begin:
:Function:       cudaTest
:Pattern:        cudaTest[theta_Real, k_Integer, data_List]
:Arguments:      { theta, k, data }
:ArgumentTypes:  { Real, Integer, Manual }
:ReturnType:     Manual
:End:


