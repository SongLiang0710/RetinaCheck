
SYS = Linux-x86-64
VERSION = 11.0

WSTPLINKDIR = /usr/local/Wolfram/Mathematica/11.0/SystemFiles/Links/WSTP/DeveloperKit
CADDSDIR = ${WSTPLINKDIR}/${SYS}/CompilerAdditions

WSINCDIR = ${CADDSDIR}
WSLIBDIR = ${CADDSDIR}
WSPREP = ${CADDSDIR}/wsprep

CUDADIR = /usr/local/cuda-8.0

CUINCDIR = ${CUDADIR}/include
CULIBDIR = ${CUDADIR}/lib64
NVCC = ${CUDADIR}/bin/nvcc

EXTRA_CFLAGS = -m64
RM = rm

RcMathVisionParallel : RcMathVisionParallel.o GaussianDerivative.o OSTransform.o HessianFeature.o CEDiffusion.o SE2FiniteDerivative.o RcMathVisionParalleltm.o
	${CXX} ${EXTRA_CFLAGS} -I${WSINCDIR} -I${CUINCDIR} RcMathVisionParallel.o GaussianDerivative.o OSTransform.o HessianFeature.o CEDiffusion.o SE2FiniteDerivative.o RcMathVisionParalleltm.o -L${WSLIBDIR} -L${CULIBDIR} -lWSTP64i4 -lm -lpthread -lrt -lstdc++ -ldl -luuid -lcudart -lcufft -o $@ 


GaussianDerivative.o :
	${NVCC} -c GaussianDerivative.cu

OSTransform.o :
	${NVCC} -c OSTransform.cu

HessianFeature.o :
	${NVCC} -c HessianFeature.cu

CEDiffusion.o :
	${NVCC} -c CEDiffusion.cu

SE2FiniteDerivative.o :
	${NVCC} -c SE2FiniteDerivative.cu

.c.o :
	${CC} -c ${EXTRA_CFLAGS} -I${WSINCDIR} -I${CUINCDIR} $<

RcMathVisionParalleltm.c : RcMathVisionParallel.tm
	${WSPREP} $? -o $@

clean :
	@ ${RM} -rf *.o *tm.c 
