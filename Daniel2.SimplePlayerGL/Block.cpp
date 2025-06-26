#include "stdafx.h"
#include "Block.h"

inline void MemoryAlloc(void **pMemory, size_t iSize)
{
	//iSize = (iSize + 4095) & ~4095;

#ifdef _WIN32
	*pMemory = (LPBYTE)VirtualAlloc(NULL, iSize, MEM_COMMIT, PAGE_READWRITE);
#elif defined(__APPLE__)
	*pMemory = malloc(iSize);
#else
	*pMemory = aligned_alloc(4096, iSize);
#endif		
}

inline void MemoryFree(void *pMemory)
{
	if (pMemory)
	{
#ifdef _WIN32
		VirtualFree(pMemory, 0, MEM_RELEASE);
#elif defined(__APPLE__)
		free(pMemory);
#else
		free(pMemory);
#endif	
	}
}

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
	frame_buffer = nullptr;
	frame_buffer_tmp = nullptr;

	iWidth = iHeight = iPitch = iSizeFrame = 0;
	
	bRotateFrame = false;

	pKernelDataOut = nullptr;
	pKernelDataTmp = nullptr;

#if defined(__WIN32__)
	m_pResource = nullptr;
#endif
}

long C_Block::Init(size_t _iWidth, size_t _iHeight, size_t _iStride, size_t _iSize, bool bUseCuda)
{
	Destroy();

	iWidth = _iWidth;
	iHeight = _iHeight;

	iPitch = _iStride;
	iSizeFrame = iPitch * iHeight;
	
	if (_iSize >= iSizeFrame)
		iSizeFrame = _iSize;

	MemoryAlloc((void**)&frame_buffer, iSizeFrame); // allocating CPU memory for current frame buffer

	if (!frame_buffer)
		return -1;

#ifdef USE_CUDA_SDK
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

long C_Block::InitTmp(size_t _iSize, bool bUseCuda)
{
	if (bUseCuda)
	{
		cudaError_t res = cudaSuccess;

		res = cudaMalloc(&pKernelDataTmp, iSizeFrame); __vrcu

		if (res != cudaSuccess)
			return -1;
	}
	else
	{
		MemoryAlloc((void**)&frame_buffer_tmp, _iSize);

		if (!frame_buffer_tmp)
			return -1;
	}

	return 0;
}

void C_Block::Destroy()
{
	MemoryFree(frame_buffer);
	MemoryFree(frame_buffer_tmp);

#ifdef USE_CUDA_SDK
	if (pKernelDataOut)
	{
		cudaFree(pKernelDataOut); __vrcu
	}
	if (pKernelDataTmp)
	{
		cudaFree(pKernelDataTmp); __vrcu
	}
#endif

	Initialize();
}

int C_Block::CopyToGPU()
{
#ifdef USE_CUDA_SDK
	if (!DataGPUPtr() || !DataPtr())
		return -1;

	cudaError_t res = cudaSuccess;

	res = cudaMemcpy(DataGPUPtr(), DataPtr(), Size(), cudaMemcpyHostToDevice); __vrcu

	return res;
#else
	return 0;
#endif
}

int C_Block::CopyToCPU()
{
#ifdef USE_CUDA_SDK
	if (!DataGPUPtr() || !DataPtr())
		return -1;

	cudaError_t res = cudaSuccess;

	res = cudaMemcpy(DataPtr(), DataGPUPtr(), Size(), cudaMemcpyDeviceToHost); __vrcu

	return res;
#else
	return 0;
#endif
}
