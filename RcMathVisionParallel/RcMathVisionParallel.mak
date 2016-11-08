
CFLAGS = /nologo /c /W3 /Z7 /Od /DWIN32 /D_DEBUG /D_WINDOWS

# Linking against gdi32.lib for access to windowing mechanisms
LFLAGS = /DEBUG /PDB:NONE /NOLOGO /SUBSYSTEM:windows /INCREMENTAL:no kernel32.lib user32.lib gdi32.lib


PLATFORM = WIN64

LIBFILE = wstp64i4m.lib wstp64i4.lib wstp64i4s.lib wstp64i3.lib wstp64i3m.lib wstp64i3s.lib

CUDADIR = D:/CUDAToolkit


CUINCDIR = $(CUDADIR)/include


CULIBDIR = $(CUDADIR)/lib/x64

NVCC = $(CUDADIR)/bin/nvcc.exe


RcMathVisionParallel_Win64.exe : RcMathVisionParallel.obj GaussianDerivative.obj OSTransform.obj HessianFeature.obj CEDiffusion.obj SE2FiniteDerivative.obj RcMathVisionParalleltm.obj
  LINK RcMathVisionParallel.obj GaussianDerivative.obj OSTransform.obj HessianFeature.obj CEDiffusion.obj SE2FiniteDerivative.obj RcMathVisionParalleltm.obj$(LIBFILE) $(CULIBDIR)/cudart.lib $(CULIBDIR)/cufft.lib /OUT:RcMathVisionParallel_Win64.exe @<<
$(LFLAGS)
<<


RcMathVisionParallel.obj : RcMathVisionParallel.c
  CL -I$(CUINCDIR) @<< RcMathVisionParallel.c
$(CFLAGS)
<<

RcMathVisionParalleltm.obj : RcMathVisionParalleltm.c
  CL @<< RcMathVisionParalleltm.c
$(CFLAGS)
<<

RcMathVisionParalleltm.c : RcMathVisionParallel.tm
  wsprep RcMathVisionParallel.tm -o RcMathVisionParalleltm.c

GaussianDerivative.obj : GaussianDerivative.cu
  $(NVCC) -c GaussianDerivative.cu

OSTransform.obj : OSTransform.cu
  $(NVCC) -c OSTransform.cu

HessianFeature.obj : HessianFeature.cu
  $(NVCC) -c HessianFeature.cu

CEDiffusion.obj : CEDiffusion.cu
  $(NVCC) -c CEDiffusion.cu

SE2FiniteDerivative.obj : SE2FiniteDerivative.cu
  $(NVCC) -c SE2FiniteDerivative.cu

