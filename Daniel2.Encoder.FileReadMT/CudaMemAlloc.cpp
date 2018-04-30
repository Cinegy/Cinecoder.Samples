#include "stdafx.h"

void *cuda_alloc_pinned(size_t size)
{
	void *ptr = NULL;

	if(HMODULE hInstLib = LoadLibraryA("cudart64_80.dll"))
	{
		if(FARPROC pProc = GetProcAddress(hInstLib, "cudaMallocHost"))
		{
			typedef int (STDAPICALLTYPE *cudaMallocHostFunc)(void **ptr, size_t size);
			(cudaMallocHostFunc(pProc))(&ptr, size);
		}
	
		FreeLibrary(hInstLib);
	}

	return ptr;
}

void cuda_free_pinned(void *ptr)
{
	if(HMODULE hInstLib = LoadLibraryA("cudart64_80.dll"))
	{
		if(FARPROC pProc = GetProcAddress(hInstLib, "cudaFreeHost"))
		{
			typedef int (STDAPICALLTYPE *cudaFreeHostFunc)(void *ptr);
			(cudaFreeHostFunc(pProc))(ptr);
		}
	
		FreeLibrary(hInstLib);
	}
}
