#pragma once

//#include <cuda.h>
//#include <cuda_runtime.h>
//#include <cuda_gl_interop.h>

//#if defined(__WIN32__)
//	#pragma comment(lib, "cudart_static.lib")
//#else defined(__LINUX__)
//	#pragma comment(lib, "libcudart_static.a")
//	#endif
//#endif

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

typedef cudaError_t (*FTcudaGraphicsResourceSetMapFlags)(cudaGraphicsResource_t resource, unsigned int flags);

typedef cudaError_t(*FTcudaMemcpy2DToArray)(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind);
typedef cudaError_t(*FTcudaMemcpy2DToArrayAsync)(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream);
typedef cudaError_t(*FTcudaMemcpyArrayToArray)(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind);

typedef cudaError_t(*FTcudaMallocArray)(cudaArray_t *array, const struct cudaChannelFormatDesc *desc, size_t width, size_t height, unsigned int flags);
typedef cudaError_t(*FTcudaFreeArray)(cudaArray_t array);
typedef cudaChannelFormatDesc(*FTcudaCreateChannelDesc)(int x, int y, int z, int w, enum cudaChannelFormatKind f);

#define _extern static

_extern FTcudaGetLastError FUNC_CUDA(cudaGetLastError);
_extern FTcudaGetErrorString FUNC_CUDA(cudaGetErrorString);

_extern FTcudaMalloc FUNC_CUDA(cudaMalloc);
_extern FTcudaMemset FUNC_CUDA(cudaMemset);
_extern FTcudaMemcpy FUNC_CUDA(cudaMemcpy);
_extern FTcudaFree FUNC_CUDA(cudaFree);

_extern FTcudaMallocHost FUNC_CUDA(cudaMallocHost);
_extern FTcudaFreeHost FUNC_CUDA(cudaFreeHost);

_extern FTcudaGetDevice FUNC_CUDA(cudaGetDevice);
_extern FTcudaSetDevice FUNC_CUDA(cudaSetDevice);

_extern FTcudaStreamCreate FUNC_CUDA(cudaStreamCreate);
_extern FTcudaStreamDestroy FUNC_CUDA(cudaStreamDestroy);
_extern FTcudaStreamSynchronize FUNC_CUDA(cudaStreamSynchronize);

_extern FTcudaGraphicsGLRegisterImage FUNC_CUDA(cudaGraphicsGLRegisterImage);
_extern FTcudaGraphicsGLRegisterBuffer FUNC_CUDA(cudaGraphicsGLRegisterBuffer);
#if defined(__WIN32__)
_extern FTcudaGraphicsD3D11RegisterResource FUNC_CUDA(cudaGraphicsD3D11RegisterResource);
#endif
_extern FTcudaGraphicsUnregisterResource FUNC_CUDA(cudaGraphicsUnregisterResource);

_extern FTcudaGraphicsMapResources FUNC_CUDA(cudaGraphicsMapResources);
_extern FTcudaGraphicsUnmapResources FUNC_CUDA(cudaGraphicsUnmapResources);

_extern FTcudaGraphicsSubResourceGetMappedArray FUNC_CUDA(cudaGraphicsSubResourceGetMappedArray);
_extern FTcudaGraphicsResourceGetMappedPointer FUNC_CUDA(cudaGraphicsResourceGetMappedPointer);

_extern FTcudaGraphicsResourceSetMapFlags FUNC_CUDA(cudaGraphicsResourceSetMapFlags);

_extern FTcudaMemcpy2DToArray FUNC_CUDA(cudaMemcpy2DToArray);
_extern FTcudaMemcpy2DToArrayAsync FUNC_CUDA(cudaMemcpy2DToArrayAsync);
_extern FTcudaMemcpyArrayToArray FUNC_CUDA(cudaMemcpyArrayToArray);

_extern FTcudaMallocArray FUNC_CUDA(cudaMallocArray);
_extern FTcudaFreeArray FUNC_CUDA(cudaFreeArray);
_extern FTcudaCreateChannelDesc FUNC_CUDA(cudaCreateChannelDesc);

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
	#define CUDART_FILENAME "/usr/local/cuda/lib64/libcudart.so"
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

		FUNC_CUDA(cudaMallocHost) = (FTcudaMallocHost)GetProcAddress(hCuda, "cudaMallocHost");
		FUNC_CUDA(cudaFreeHost) = (FTcudaFreeHost)GetProcAddress(hCuda, "cudaFreeHost");

		FUNC_CUDA(cudaGetDevice) = (FTcudaGetDevice)GetProcAddress(hCuda, "cudaGetDevice");
		FUNC_CUDA(cudaSetDevice) = (FTcudaSetDevice)GetProcAddress(hCuda, "cudaSetDevice");

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

		FUNC_CUDA(cudaMallocArray) = (FTcudaMallocArray)GetProcAddress(hCuda, "cudaMallocArray");
		FUNC_CUDA(cudaFreeArray) = (FTcudaFreeArray)GetProcAddress(hCuda, "cudaFreeArray");
		FUNC_CUDA(cudaCreateChannelDesc) = (FTcudaCreateChannelDesc)GetProcAddress(hCuda, "cudaCreateChannelDesc");
	}
	else
		return fprintf(stderr, "CUDA init error: failed to load %s\n", CUDART_FILENAME), -1;

	if (!FUNC_CUDA(cudaGetLastError) || !FUNC_CUDA(cudaGetErrorString) ||
		!FUNC_CUDA(cudaMalloc) || !FUNC_CUDA(cudaMemset) || !FUNC_CUDA(cudaFree) ||
		!FUNC_CUDA(cudaGraphicsGLRegisterImage) || !FUNC_CUDA(cudaGraphicsUnregisterResource) ||
		!FUNC_CUDA(cudaGraphicsMapResources) || !FUNC_CUDA(cudaGraphicsUnmapResources) ||
		!FUNC_CUDA(cudaGraphicsSubResourceGetMappedArray) || !FUNC_CUDA(cudaMemcpy2DToArray))
		return fprintf(stderr, "CUDA init error: failed to find required functions\n"), -2;

	return 0;
}                                                                           

static void destroyCUDA()
{
	if (hCuda)
		FreeLibrary(hCuda);
}
