#include "stdafx.h"

#ifndef _WIN32

#include <dlfcn.h>

typedef void* HMODULE;
typedef void* FARPROC;

#define STDAPICALLTYPE    /**/
#define LoadLibraryA(lib_name)	dlopen(lib_name, RTLD_LAZY)
#define GetProcAddress			dlsym
#define FreeLibrary			    dlclose

#ifdef __LINUX__
#define CUDART_DLL_NAME "libcudart.so.8.0"
#else //__APPLE__
#define CUDART_DLL_NAME "libcudart.dylib.8.0"
#endif

#else
#define CUDART_DLL_NAME "cudart64_80.dll"
#endif

void *cuda_alloc_pinned(size_t size)
{
	void *ptr = NULL;

	if(HMODULE hInstLib = LoadLibraryA(CUDART_DLL_NAME))
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
	if(HMODULE hInstLib = LoadLibraryA(CUDART_DLL_NAME))
	{
		if(FARPROC pProc = GetProcAddress(hInstLib, "cudaFreeHost"))
		{
			typedef int (STDAPICALLTYPE *cudaFreeHostFunc)(void *ptr);
			(cudaFreeHostFunc(pProc))(ptr);
		}
	
		FreeLibrary(hInstLib);
	}
}
