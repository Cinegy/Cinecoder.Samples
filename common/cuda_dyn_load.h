#pragma once

#ifdef CUDA_WRAPPER
#define cudaArrayDefault                    0x00  /**< Default CUDA array allocation flag */
#define cudaArrayLayered                    0x01  /**< Must be set in cudaMalloc3DArray to create a layered CUDA array */
#define cudaArraySurfaceLoadStore           0x02  /**< Must be set in cudaMallocArray or cudaMalloc3DArray in order to bind surfaces to the CUDA array */
#define cudaArrayCubemap                    0x04  /**< Must be set in cudaMalloc3DArray to create a cubemap CUDA array */
#define cudaArrayTextureGather              0x08  /**< Must be set in cudaMallocArray or cudaMalloc3DArray in order to perform texture gather operations on the CUDA array */

enum cudaChannelFormatKind
{
	cudaChannelFormatKindSigned = 0,      /**< Signed channel format */
	cudaChannelFormatKindUnsigned = 1,      /**< Unsigned channel format */
	cudaChannelFormatKindFloat = 2,      /**< Float channel format */
	cudaChannelFormatKindNone = 3       /**< No channel format */
};

struct cudaChannelFormatDesc
{
	int                        x; /**< x */
	int                        y; /**< y */
	int                        z; /**< z */
	int                        w; /**< w */
	enum cudaChannelFormatKind f; /**< Channel format kind */
};

enum cudaMemcpyKind
{
	cudaMemcpyHostToHost = 0,      /**< Host   -> Host */
	cudaMemcpyHostToDevice = 1,      /**< Host   -> Device */
	cudaMemcpyDeviceToHost = 2,      /**< Device -> Host */
	cudaMemcpyDeviceToDevice = 3,      /**< Device -> Device */
	cudaMemcpyDefault = 4       /**< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing */
};

enum cudaGraphicsRegisterFlags
{
	cudaGraphicsRegisterFlagsNone = 0,  /**< Default */
	cudaGraphicsRegisterFlagsReadOnly = 1,  /**< CUDA will not write to this resource */
	cudaGraphicsRegisterFlagsWriteDiscard = 2,  /**< CUDA will only write to and will not read from this resource */
	cudaGraphicsRegisterFlagsSurfaceLoadStore = 4,  /**< CUDA will bind this resource to a surface reference */
	cudaGraphicsRegisterFlagsTextureGather = 8   /**< CUDA will perform texture gather operations on this resource */
};

enum cudaGraphicsMapFlags
{
	cudaGraphicsMapFlagsNone = 0,  /**< Default; Assume resource can be read/written */
	cudaGraphicsMapFlagsReadOnly = 1,  /**< CUDA will not write to this resource */
	cudaGraphicsMapFlagsWriteDiscard = 2   /**< CUDA will only write to and will not read from this resource */
};

enum cudaError
{
	cudaSuccess = 0,
	//...
	cudaErrorStartupFailure = 0x7f,
	cudaErrorApiFailureBase = 10000
};

typedef struct cudaGraphicsResource *cudaGraphicsResource_t;
typedef struct cudaArray *cudaArray_t;
typedef const struct cudaArray *cudaArray_const_t;
typedef struct CUstream_st *cudaStream_t;
typedef enum cudaError cudaError_t;

#define FUNC_CUDA(func) func
#else
#define FUNC_CUDA(func) f_##func
#endif

#define CHECK_FUNC_CUDA(func) \
	if (!FUNC_CUDA(func)) { \
		fprintf(stderr, "CUDA init error: failed to find required functions (File: %s Line %d)\n", __FILE__, __LINE__); \
		return -2; \
	}

typedef cudaError_t(*FTcudaGetLastError)();
typedef const char*(*FTcudaGetErrorString)(cudaError_t error);

typedef cudaError_t(*FTcudaMalloc)(void **devPtr, size_t size);
typedef cudaError_t(*FTcudaMemset)(void *devPtr, int value, size_t count);
typedef cudaError_t(*FTcudaMemcpy)(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
typedef cudaError_t(*FTcudaFree)(void *devPtr);

typedef cudaError_t(*FTcudaMemcpy2D)(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);

typedef cudaError_t(*FTcudaMallocHost)(void **ptr, size_t size);
typedef cudaError_t(*FTcudaFreeHost)(void *ptr);

typedef cudaError_t(*FTcudaGetDevice)(int *);
typedef cudaError_t(*FTcudaSetDevice)(int);

typedef cudaError_t(*FTcudaStreamCreate)(cudaStream_t *pStream);
typedef cudaError_t(*FTcudaStreamDestroy)(cudaStream_t stream);
typedef cudaError_t(*FTcudaStreamSynchronize)(cudaStream_t stream);

typedef cudaError_t(*FTcudaGraphicsGLRegisterImage)(struct cudaGraphicsResource **resource, unsigned int image, unsigned int target, unsigned int flags);
typedef cudaError_t(*FTcudaGraphicsGLRegisterBuffer)(struct cudaGraphicsResource ** resource, unsigned int buffer, unsigned int flags);
#if defined(__WIN32__)
typedef cudaError_t(*FTcudaGraphicsD3D11RegisterResource)(struct cudaGraphicsResource **resource, ID3D11Resource *pD3DResource, unsigned int flags);
#endif
typedef cudaError_t(*FTcudaGraphicsUnregisterResource)(cudaGraphicsResource_t resource);
typedef cudaError_t(*FTcudaGraphicsMapResources)(int count, cudaGraphicsResource_t *resources, cudaStream_t stream);
typedef cudaError_t(*FTcudaGraphicsUnmapResources)(int count, cudaGraphicsResource_t *resources, cudaStream_t stream);

typedef cudaError_t(*FTcudaGraphicsSubResourceGetMappedArray)(cudaArray_t *array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel);
typedef cudaError_t(*FTcudaGraphicsResourceGetMappedPointer)(void **devPtr, size_t* size, cudaGraphicsResource_t resource);

typedef cudaError_t(*FTcudaGraphicsResourceSetMapFlags)(cudaGraphicsResource_t resource, unsigned int flags);

typedef cudaError_t(*FTcudaMemcpy2DToArray)(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind);
typedef cudaError_t(*FTcudaMemcpy2DToArrayAsync)(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream);
typedef cudaError_t(*FTcudaMemcpyArrayToArray)(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind);
typedef cudaError_t(*FTcudaMemcpy2DArrayToArray)(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind);

typedef cudaError_t(*FTcudaMallocArray)(cudaArray_t *array, const struct cudaChannelFormatDesc *desc, size_t width, size_t height, unsigned int flags);
typedef cudaError_t(*FTcudaFreeArray)(cudaArray_t array);
typedef cudaChannelFormatDesc(*FTcudaCreateChannelDesc)(int x, int y, int z, int w, enum cudaChannelFormatKind f);

#include "cuda_dyn_declare.h" // declare list of functuions 

#if defined(_WIN32)
#if _WIN64
static const std::vector<std::string> cudart_paths = {
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
	"/usr/local/cuda-10.0/targets/aarch64-linux/lib/libcudart.so"
};
#endif
static std::string str_cudart_path = "";

static HMODULE hCuda = nullptr;

#define LOAD_CUDA_FUNC(function) \
	FUNC_CUDA(function) = (FT##function)GetProcAddress(hCuda, #function); CHECK_FUNC_CUDA(function)

static void destroyCUDA()
{
	if (hCuda)
		FreeLibrary(hCuda);

	hCuda = nullptr;
}

static int initCUDA()
{
	destroyCUDA();

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

#if defined(__WIN32__)
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
	}
	else
		return fprintf(stderr, "CUDA init error: failed to load!\n"), -1;

	return 0;
}

