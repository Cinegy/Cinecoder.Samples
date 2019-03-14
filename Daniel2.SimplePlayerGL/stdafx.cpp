// stdafx.cpp : source file that includes just the standard includes
// SimpleDecodeDN2.pch will be the pre-compiled header
// stdafx.obj will contain the pre-compiled type information

#include "stdafx.h"

#ifdef USE_CUDA_SDK // CUDA
FTcudaGetLastError FUNC_CUDA(cudaGetLastError) = nullptr;
FTcudaGetErrorString FUNC_CUDA(cudaGetErrorString) = nullptr;

FTcudaMalloc FUNC_CUDA(cudaMalloc) = nullptr;
FTcudaMemset FUNC_CUDA(cudaMemset) = nullptr;
FTcudaMemcpy FUNC_CUDA(cudaMemcpy) = nullptr;
FTcudaFree FUNC_CUDA(cudaFree) = nullptr;

FTcudaStreamCreate FUNC_CUDA(cudaStreamCreate) = nullptr;
FTcudaStreamDestroy FUNC_CUDA(cudaStreamDestroy) = nullptr;
FTcudaStreamSynchronize FUNC_CUDA(cudaStreamSynchronize) = nullptr;

FTcudaGraphicsGLRegisterImage FUNC_CUDA(cudaGraphicsGLRegisterImage) = nullptr;

#if defined(__WIN32__)
FTcudaGraphicsD3D11RegisterResource FUNC_CUDA(cudaGraphicsD3D11RegisterResource) = nullptr;
#endif

FTcudaGraphicsUnregisterResource FUNC_CUDA(cudaGraphicsUnregisterResource) = nullptr;

FTcudaGraphicsMapResources FUNC_CUDA(cudaGraphicsMapResources) = nullptr;
FTcudaGraphicsUnmapResources FUNC_CUDA(cudaGraphicsUnmapResources) = nullptr;

FTcudaGraphicsSubResourceGetMappedArray FUNC_CUDA(cudaGraphicsSubResourceGetMappedArray) = nullptr;

FTcudaMemcpy2DToArray FUNC_CUDA(cudaMemcpy2DToArray) = nullptr;
FTcudaMemcpy2DToArrayAsync FUNC_CUDA(cudaMemcpy2DToArrayAsync) = nullptr;
#endif
