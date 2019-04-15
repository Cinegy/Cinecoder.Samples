#pragma once

#ifdef CUDA_WRAPPER
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
	typedef struct CUstream_st *cudaStream_t;
	typedef enum cudaError cudaError_t;

	#define FUNC_CUDA(func) func
#else
	#define FUNC_CUDA(func) f_##func
#endif

typedef cudaError_t(*FTcudaGetLastError)();
typedef const char*(*FTcudaGetErrorString)(cudaError_t error);

typedef cudaError_t(*FTcudaMalloc)(void **devPtr, size_t size);
typedef cudaError_t(*FTcudaMemset)(void *devPtr, int value, size_t count);
typedef cudaError_t(*FTcudaMemcpy)(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
typedef cudaError_t(*FTcudaFree)(void *devPtr);

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

typedef cudaError_t (*FTcudaGraphicsResourceSetMapFlags)(cudaGraphicsResource_t resource, unsigned int flags);

typedef cudaError_t(*FTcudaMemcpy2DToArray)(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind);
typedef cudaError_t(*FTcudaMemcpy2DToArrayAsync)(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream);
typedef cudaError_t(*FTcudaMemcpyArrayToArray)(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind);

extern FTcudaGetLastError FUNC_CUDA(cudaGetLastError);
extern FTcudaGetErrorString FUNC_CUDA(cudaGetErrorString);

extern FTcudaMalloc FUNC_CUDA(cudaMalloc);
extern FTcudaMemset FUNC_CUDA(cudaMemset);
extern FTcudaMemcpy FUNC_CUDA(cudaMemcpy);
extern FTcudaFree FUNC_CUDA(cudaFree);

extern FTcudaStreamCreate FUNC_CUDA(cudaStreamCreate);
extern FTcudaStreamDestroy FUNC_CUDA(cudaStreamDestroy);
extern FTcudaStreamSynchronize FUNC_CUDA(cudaStreamSynchronize);

extern FTcudaGraphicsGLRegisterImage FUNC_CUDA(cudaGraphicsGLRegisterImage);
extern FTcudaGraphicsGLRegisterBuffer FUNC_CUDA(cudaGraphicsGLRegisterBuffer);
#if defined(__WIN32__)
extern FTcudaGraphicsD3D11RegisterResource FUNC_CUDA(cudaGraphicsD3D11RegisterResource);
#endif
extern FTcudaGraphicsUnregisterResource FUNC_CUDA(cudaGraphicsUnregisterResource);

extern FTcudaGraphicsMapResources FUNC_CUDA(cudaGraphicsMapResources);
extern FTcudaGraphicsUnmapResources FUNC_CUDA(cudaGraphicsUnmapResources);

extern FTcudaGraphicsSubResourceGetMappedArray FUNC_CUDA(cudaGraphicsSubResourceGetMappedArray);
extern FTcudaGraphicsResourceGetMappedPointer FUNC_CUDA(cudaGraphicsResourceGetMappedPointer);

extern FTcudaGraphicsResourceSetMapFlags FUNC_CUDA(cudaGraphicsResourceSetMapFlags);

extern FTcudaMemcpy2DToArray FUNC_CUDA(cudaMemcpy2DToArray);
extern FTcudaMemcpy2DToArrayAsync FUNC_CUDA(cudaMemcpy2DToArrayAsync);
extern FTcudaMemcpyArrayToArray FUNC_CUDA(cudaMemcpyArrayToArray);

#if defined(_WIN32)
	#define CUDART32_FILENAME "cudart32_80.dll"
	#define CUDART64_FILENAME "cudart64_80.dll"
	#if _WIN64
		#define CUDART_FILENAME CUDART64_FILENAME
	#else
		#define CUDART_FILENAME CUDART32_FILENAME
	#endif
#else
	#include <dlfcn.h>
	#define LoadLibraryA(name) dlopen(name, RTLD_LAZY)
	#define FreeLibrary(lib) dlclose(lib)
	#define GetProcAddress(lib, func) dlsym(lib, func)
	typedef void* FARPROC;
	typedef void* HMODULE;
	//#define CUDART_FILENAME "./libcudart.so.10.0"
	//#define CUDART_FILENAME "/usr/local/cuda-10.0/lib64/libcudart.so.10.0"
	#define CUDART_FILENAME "/usr/local/cuda-10.0/lib64/libcudart.so"
#endif

static HMODULE hCuda = nullptr;

static int initCUDA()
{
	hCuda = LoadLibraryA(CUDART_FILENAME);

	if (hCuda)
	{
		FUNC_CUDA(cudaGetLastError) = (FTcudaGetLastError)GetProcAddress(hCuda, "cudaGetLastError");
		FUNC_CUDA(cudaGetErrorString) = (FTcudaGetErrorString)GetProcAddress(hCuda, "cudaGetErrorString");

		FUNC_CUDA(cudaMalloc) = (FTcudaMalloc)GetProcAddress(hCuda, "cudaMalloc");
		FUNC_CUDA(cudaMemset) = (FTcudaMemset)GetProcAddress(hCuda, "cudaMemset");
		FUNC_CUDA(cudaMemcpy) = (FTcudaMemcpy)GetProcAddress(hCuda, "cudaMemcpy");
		FUNC_CUDA(cudaFree) = (FTcudaFree)GetProcAddress(hCuda, "cudaFree");

		FUNC_CUDA(cudaStreamCreate) = (FTcudaStreamCreate)GetProcAddress(hCuda, "cudaStreamCreate");
		FUNC_CUDA(cudaStreamDestroy) = (FTcudaStreamDestroy)GetProcAddress(hCuda, "cudaStreamDestroy");
		FUNC_CUDA(cudaStreamSynchronize) = (FTcudaStreamSynchronize)GetProcAddress(hCuda, "cudaStreamSynchronize");

		FUNC_CUDA(cudaGraphicsGLRegisterImage) = (FTcudaGraphicsGLRegisterImage)GetProcAddress(hCuda, "cudaGraphicsGLRegisterImage");
		FUNC_CUDA(cudaGraphicsGLRegisterBuffer) = (FTcudaGraphicsGLRegisterBuffer)GetProcAddress(hCuda, "cudaGraphicsGLRegisterBuffer");
		
#if defined(__WIN32__)
		FUNC_CUDA(cudaGraphicsD3D11RegisterResource) = (FTcudaGraphicsD3D11RegisterResource)GetProcAddress(hCuda, "cudaGraphicsD3D11RegisterResource");
#endif

		FUNC_CUDA(cudaGraphicsUnregisterResource) = (FTcudaGraphicsUnregisterResource)GetProcAddress(hCuda, "cudaGraphicsUnregisterResource");

		FUNC_CUDA(cudaGraphicsMapResources) = (FTcudaGraphicsMapResources)GetProcAddress(hCuda, "cudaGraphicsMapResources");
		FUNC_CUDA(cudaGraphicsUnmapResources) = (FTcudaGraphicsUnmapResources)GetProcAddress(hCuda, "cudaGraphicsUnmapResources");

		FUNC_CUDA(cudaGraphicsSubResourceGetMappedArray) = (FTcudaGraphicsSubResourceGetMappedArray)GetProcAddress(hCuda, "cudaGraphicsSubResourceGetMappedArray");
		FUNC_CUDA(cudaGraphicsResourceGetMappedPointer) = (FTcudaGraphicsResourceGetMappedPointer)GetProcAddress(hCuda, "cudaGraphicsResourceGetMappedPointer");

		FUNC_CUDA(cudaGraphicsResourceSetMapFlags) = (FTcudaGraphicsResourceSetMapFlags)GetProcAddress(hCuda, "cudaGraphicsResourceSetMapFlags");
		
		FUNC_CUDA(cudaMemcpy2DToArray) = (FTcudaMemcpy2DToArray)GetProcAddress(hCuda, "cudaMemcpy2DToArray");
		FUNC_CUDA(cudaMemcpy2DToArrayAsync) = (FTcudaMemcpy2DToArrayAsync)GetProcAddress(hCuda, "cudaMemcpy2DToArrayAsync");
		FUNC_CUDA(cudaMemcpyArrayToArray) = (FTcudaMemcpyArrayToArray)GetProcAddress(hCuda, "cudaMemcpyArrayToArray");
	}

	if (!FUNC_CUDA(cudaGetLastError) || !FUNC_CUDA(cudaGetErrorString) ||
		!FUNC_CUDA(cudaMalloc) || !FUNC_CUDA(cudaMemset) || !FUNC_CUDA(cudaFree) ||
		!FUNC_CUDA(cudaGraphicsGLRegisterImage) || !FUNC_CUDA(cudaGraphicsUnregisterResource) ||
		!FUNC_CUDA(cudaGraphicsMapResources) || !FUNC_CUDA(cudaGraphicsUnmapResources) ||
		!FUNC_CUDA(cudaGraphicsSubResourceGetMappedArray) || !FUNC_CUDA(cudaMemcpy2DToArray))
		return -1;

	return 0;
}

static void destroyCUDA()
{
	if (hCuda)
		FreeLibrary(hCuda);
}
