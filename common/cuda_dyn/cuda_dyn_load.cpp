//#pragma once

#if defined(_WIN32)
#include <windows.h> // for HMODULE, LoadLibrary/GetProcAddress
#endif

#include <vector>
#include <string>

#include "cuda_dyn_load.h"

#include "cuda_dyn_declare.h" // declare list of functuions 

#if defined(_WIN32)
#if _WIN64
static const std::vector<std::string> cudart_paths = {
	"cudart64_12.dll",
	"cudart64_110.dll",
	"cudart64_102.dll",
	"cudart64_101.dll",
	"cudart64_100.dll",
	"cudart64_90.dll",
	"cudart64_80.dll"
};
#else
static const std::vector<std::string> cudart_paths = {
	"cudart32_110.dll",
	"cudart32_102.dll",
	"cudart32_101.dll",
	"cudart32_100.dll",
	"cudart32_90.dll",
	"cudart32_80.dll"
};
#endif
//#define CUDART32_FILENAME "cudart32_80.dll"
//#define CUDART64_FILENAME "cudart64_80.dll"
//#if _WIN64
//#define CUDART_FILENAME CUDART64_FILENAME
//#else
//#define CUDART_FILENAME CUDART32_FILENAME
//#endif
#else
#include <dlfcn.h>
#define LoadLibraryA(name) dlopen(name, RTLD_LAZY)
#define FreeLibrary(lib) dlclose(lib)
#define GetProcAddress(lib, func) dlsym(lib, func)
typedef void* FARPROC;
typedef void* HMODULE;
//#define CUDART_FILENAME "/usr/local/cuda/lib64/libcudart.so"
//#define CUDART_FILENAME "libcudart.so"
static const std::vector<std::string> cudart_paths = {
	"libcudart.so",
	"/usr/local/cuda/lib64/libcudart.so",
	"/usr/local/cuda-11.4/targets/aarch64-linux/lib/libcudart.so",
	"/usr/local/cuda-10.0/targets/aarch64-linux/lib/libcudart.so"
};
#endif
static std::string str_cudart_path = "";

static HMODULE hCuda = nullptr;

#define LOAD_CUDA_FUNC(function) \
	FUNC_CUDA(function) = (FT##function)GetProcAddress(hCuda, #function); CHECK_FUNC_CUDA(function)

int DynamicLoadCUDA::InitCUDA()
{
	DynamicLoadCUDA::DestroyCUDA();


	for (size_t i = 0; i < cudart_paths.size(); i++)
	{
		std::string str_cudart_lib_path = cudart_paths[i];
		hCuda = LoadLibraryA(str_cudart_lib_path.c_str());
		if (hCuda)
		{
			str_cudart_path = str_cudart_lib_path;
			break;
		}
	}

	if (hCuda)
	{
		LOAD_CUDA_FUNC(cudaGetLastError)
		LOAD_CUDA_FUNC(cudaGetErrorString)

		LOAD_CUDA_FUNC(cudaMalloc)
		LOAD_CUDA_FUNC(cudaMemset)
		LOAD_CUDA_FUNC(cudaMemcpy)
		LOAD_CUDA_FUNC(cudaFree)

		LOAD_CUDA_FUNC(cudaMemcpy2D)

		LOAD_CUDA_FUNC(cudaMallocHost)
		LOAD_CUDA_FUNC(cudaFreeHost)

		LOAD_CUDA_FUNC(cudaGetDevice)
		LOAD_CUDA_FUNC(cudaSetDevice)

		LOAD_CUDA_FUNC(cudaStreamCreate)
		LOAD_CUDA_FUNC(cudaStreamDestroy)
		LOAD_CUDA_FUNC(cudaStreamSynchronize)

		LOAD_CUDA_FUNC(cudaGraphicsGLRegisterImage)
		LOAD_CUDA_FUNC(cudaGraphicsGLRegisterBuffer)

#if defined(_WIN32)
		LOAD_CUDA_FUNC(cudaGraphicsD3D11RegisterResource)
#endif

		LOAD_CUDA_FUNC(cudaGraphicsUnregisterResource)

		LOAD_CUDA_FUNC(cudaGraphicsMapResources)
		LOAD_CUDA_FUNC(cudaGraphicsUnmapResources)

		LOAD_CUDA_FUNC(cudaGraphicsSubResourceGetMappedArray)
		LOAD_CUDA_FUNC(cudaGraphicsResourceGetMappedPointer)

		LOAD_CUDA_FUNC(cudaGraphicsResourceSetMapFlags)

		LOAD_CUDA_FUNC(cudaMemcpy2DToArray)
		LOAD_CUDA_FUNC(cudaMemcpy2DToArrayAsync)
		LOAD_CUDA_FUNC(cudaMemcpyArrayToArray)
		LOAD_CUDA_FUNC(cudaMemcpy2DArrayToArray)

		LOAD_CUDA_FUNC(cudaMallocArray)
		LOAD_CUDA_FUNC(cudaFreeArray)
		LOAD_CUDA_FUNC(cudaCreateChannelDesc)

		LOAD_CUDA_FUNC(cudaHostAlloc)
		LOAD_CUDA_FUNC(cudaHostGetDevicePointer)
		LOAD_CUDA_FUNC(cudaSetDeviceFlags)
		LOAD_CUDA_FUNC(cudaHostRegister)
		LOAD_CUDA_FUNC(cudaHostUnregister)

		LOAD_CUDA_FUNC(cudaDeviceGetAttribute)
		LOAD_CUDA_FUNC(cudaGetDeviceProperties)
	}
	else
		return fprintf(stderr, "CUDA init error: failed to load!\n"), -1;

	return 0;
}

int DynamicLoadCUDA::DestroyCUDA()
{
	if (hCuda)
		FreeLibrary(hCuda);

	hCuda = nullptr;

	return 0;
}

