(* ::Package:: *)

BeginPackage["RcMathVisionParallel`"]


(*This copy of the pacakage is for 64-bit windows only.*)


cudaGaussianDerivative::usage = "cudaGaussianDerivative[nx, ny, sigma, data]\n gives the separable convolution of Gaussian derivative."
cudaGaugeDerivative::usage = "cudaGaugeDerivative[nv, nw, sigma, data]\n gives the (common) Gauge derivative"
cudaCakeWaveletStackFourier::usage = "cudaCakeWaveletStackFourier[size, nc, k, t, q, ov, periodicity]\n generates the OS kernel stack in Fourier domain."
cudaCakeWaveletStack::usage = "cudaCakeWaveletStack[size, nc, k, t, q, s, ov, periodicity]\n gives the OS kernel stack in space domain."
cudaOS2DCakeTransform::usage = "cudaOS2DCakeTransform[size, nc, k, t, q, s, ov, periodicity, method, data]
 gives the OS transform stack, with size_ the size of OS kernel, nc_ the number of pieces, k_ the order of generating B spline, t_ and q_ parameters in high frequency decay, s_ the sigma value of spacial Gaussian window, ov_ the coefficient overlap, periodicity_ Pi or 2 Pi, method_ the type of implementation method, and data_ the original image data."

cudaOS2DGaussianDerivative::usage="cudaOS2DGaussianDerivative[sigmaS_, sigmaO_, mu_, order_, data_]\n gives Gaussian derivative of a certain order."
cudaOS2DGaussianHessian::usage="cudaOS2DGaussianHessian[sigmaS_, sigmaO_, mu_, data_]\n gives the Hessian matrix of each 'voxel' in SE(2) space."
cudaOS2DHessianFeatures::usage = "cudaOS2DHessianFeatures[mu, data]\n gives sequently the curvature, confidence and deviation from horizontal of the input Hessian matrix."

cudaNonlinearDiffusionFunction::usage = "cudaNonlinearDiffusionFunction[c, data]\n gives the nonlinear coefficient in the diffusion tensor."
cudaOS2DDiffusionStepExplicit::usage = "cudaOS2DDiffusionStepExplicit[k, steps, tau, {data, Dtt, Dww, Dvv, Dwt}]\n executes the nonlinear diffusion instructed by explicit tensor. k is the order of interpolation spline and steps stands for the number of step with tau stride. All the four tensor arrays must be of the identical dimensionality as data."
cudaOS2DCoherenceEnhancingDiffusionStep::usage = "cudaOS2DCoherenceEnhancingDiffusionStep[k, steps, tau, mu, {data, Dbb, curvature, deviation}]\n executes the nonlinear diffusion instructed by hessian features. k is the order of interpolation spline and steps stands for the number of step with tau stride. All the three tensor arrays must be of the identical dimensionality as data."
cudaSE2FiniteDerivative::usage="cudaSE2FiniteDerivative[order_, k_, data_]\n gives interpolated finete derivative of a certain order."
cudaSE2FiniteDerivativeHessian::usage="cudaSE2FiniteDerivativeHessian[k_, data_]\n gives the Hessian matrix of each 'voxel' in SE(2) space."

cudaTest::usage = "cudaTest is a test function."


cudaUninstall::usage="cudaUnistall[] release the current RcMathVisionParallel link."



RcMathVisionParallel::locallink="RcMathVisionParallel local link successfully installed."
RcMathVisionParallel::tcpiplink="RcMathVisionParallel TCPIP link successfully installed."
RcMathVisionParallel::invalidlink="Could not find valid link."
RcMathVisionParallel::multiplelink="Warning: multiple binary links found."


Begin["`Private`"]


Clear[link];
link=Global`RcMathVisionParallelLink;

If[Length@link==0,
    (* search the binary file from $Path list *)
    filepath=Select[$Path, FileExistsQ[FileNameJoin[{#,"RcMathVisionParallel","Binaries","RcMathVisionParallel_win64.exe"}]]&];
    Switch[Length@filepath,
		0,Message[RcMathVisionParallel::invalidlink],
		1,directory=First[filepath];Print[link=Install[directory<>"/RcMathVisionParallel/Binaries/RcMathVisionParallel_win64.exe"]],
		_,Message[RcMathVisionParallel::multiplelink];directory=First[filepath];Print[link=Install[directory<>"/RcMathVisionParallel/Binaries/RcMathVisionParallel_win64.exe"]]
	],
	(* link has been set by TCPIP connection *)
	Print[Install[link]]
]


libraries={"wstp64i3.lib","wstp64i3m.lib","wstp64i3s.lib","wstp64i4.lib","wstp64i4m.lib","wstp64i4s.lib"};

If[StringTake[link[[1]],1]=="\"",
    (* local link set up, make sure the link libraries ready*)
    MapThread[If[FileExistsQ[FileNameJoin[{directory,"RcMathVisionParallel","Binaries",#2}]]==False,CopyFile[#1,FileNameJoin[{"RcMathVisionParallel","Binaries",#2}]]]&,
        {FileNameJoin[{$InstallationDirectory,"SystemFiles/Links/WSTP/DeveloperKit/Windows-x86-64/CompilerAdditions",#}]&/@libraries,
        libraries}];
]


cudaUninstall[]:=(Clear[Global`RcMathVisionParallelLink];Uninstall[link])


End[]
EndPackage[]
