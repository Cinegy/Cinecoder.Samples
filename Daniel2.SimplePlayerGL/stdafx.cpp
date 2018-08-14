// stdafx.cpp : source file that includes just the standard includes
// SimpleDecodeDN2.pch will be the pre-compiled header
// stdafx.obj will contain the pre-compiled type information

#include "stdafx.h"

#if defined(__WIN32__) || defined(_WIN32)
FTcudaGetLastError FUNC_CUDA(cudaGetLastError) = nullptr;
FTcudaGetErrorString FUNC_CUDA(cudaGetErrorString) = nullptr;

FTcudaMalloc FUNC_CUDA(cudaMalloc) = nullptr;
FTcudaMemset FUNC_CUDA(cudaMemset) = nullptr;
FTcudaMemcpy FUNC_CUDA(cudaMemcpy) = nullptr;
FTcudaFree FUNC_CUDA(cudaFree) = nullptr;

FTcudaGraphicsGLRegisterImage FUNC_CUDA(cudaGraphicsGLRegisterImage) = nullptr;
FTcudaGraphicsUnregisterResource FUNC_CUDA(cudaGraphicsUnregisterResource) = nullptr;

FTcudaGraphicsMapResources FUNC_CUDA(cudaGraphicsMapResources) = nullptr;
FTcudaGraphicsUnmapResources FUNC_CUDA(cudaGraphicsUnmapResources) = nullptr;

FTcudaGraphicsSubResourceGetMappedArray FUNC_CUDA(cudaGraphicsSubResourceGetMappedArray) = nullptr;
FTcudaMemcpy2DToArray FUNC_CUDA(cudaMemcpy2DToArray) = nullptr;
#endif