#include "stdafx.h"
#include "Block.h"

C_Block::C_Block()
{
	Initialize();
}

C_Block::~C_Block()
{
	Destroy();
}

void C_Block::Initialize()
{
	iWidth = iHeight = iPitch = iSizeFrame = 0;

	bRotateFrame = false;

	pKernelDataOut = nullptr;
}

long C_Block::Init(size_t _iWidth, size_t _iHeight, size_t _iStride, bool bUseCuda)
{
	Destroy();

	iWidth = _iWidth;
	iHeight = _iHeight;

	iPitch = _iStride;
	iSizeFrame = iPitch * iHeight;

	frame_buffer.resize(iSizeFrame); // allocating CPU memory for current frame buffer

	if (frame_buffer.size() != iSizeFrame)
		return -1;

#if defined(__WIN32__) || defined(_WIN32)
	if (bUseCuda)
	{
		cudaError_t res = cudaSuccess;

		res = cudaMalloc(&pKernelDataOut, iSizeFrame); __vrcu // allocating GPU memory for current frame buffer

		if (res != cudaSuccess)
			return -1;
	}
#endif

	return 0;
}

void C_Block::Destroy()
{
	frame_buffer.clear();

#if defined(__WIN32__) || defined(_WIN32)
	if (pKernelDataOut)
	{
		cudaFree(pKernelDataOut); __vrcu
	}
#endif

	Initialize();
}

int C_Block::CopyToGPU()
{
#if defined(__WIN32__) || defined(_WIN32)
	if (!pKernelDataOut)
		return -1;

	cudaError_t res = cudaSuccess;
	
	res = cudaMemcpy(DataGPUPtr(), DataPtr(), Size(), cudaMemcpyHostToDevice); __vrcu

	return res;
#else
	return 0;
#endif
}