#pragma once

#if defined(_WIN32)
typedef interface ID3D11Resource ID3D11Resource;
#endif

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
#if defined(_WIN32)
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

#define FUNC_CUDA(func) func

#define CHECK_FUNC_CUDA(func) \
	if (!FUNC_CUDA(func)) { \
		fprintf(stderr, "CUDA init error: failed to find required functions (File: %s Line %d)\n", __FILE__, __LINE__); \
		return -2; \
	}

#define CUDA_DECLARE_EXPORT
#include "cuda_dyn_declare.h" // declare list of functuions 

class DynamicLoadCUDA
{
private:
	DynamicLoadCUDA();
public:
	static int InitCUDA();
	static int DestroyCUDA();
};

#define __InitCUDA DynamicLoadCUDA::InitCUDA
#define __DestroyCUDA DynamicLoadCUDA::DestroyCUDA
