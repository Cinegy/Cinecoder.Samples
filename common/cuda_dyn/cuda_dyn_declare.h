#if defined(CUDA_DECLARE_STATIC)
#define _extern static
#elif defined(CUDA_DECLARE_EXPORT)
#define _extern extern
#else
#define _extern 
#endif

#define DECLARE_FUNC_CUDA_SIMPLE(function) \
	_extern FT##function FUNC_CUDA(function);

#define DECLARE_FUNC_CUDA_WITH_NULLPTR(function) \
	_extern FT##function FUNC_CUDA(function) = nullptr;

#if defined(CUDA_DECLARE_EXPORT) || defined(CUDA_DECLARE_STATIC)
#define DECLARE_FUNC_CUDA(function) \
	DECLARE_FUNC_CUDA_SIMPLE(function)
#else
#define DECLARE_FUNC_CUDA(function) \
	DECLARE_FUNC_CUDA_WITH_NULLPTR(function)
#endif

DECLARE_FUNC_CUDA(cudaGetLastError)
DECLARE_FUNC_CUDA(cudaGetErrorString)

DECLARE_FUNC_CUDA(cudaMalloc)
DECLARE_FUNC_CUDA(cudaMemset)
DECLARE_FUNC_CUDA(cudaMemcpy)
DECLARE_FUNC_CUDA(cudaFree)

DECLARE_FUNC_CUDA(cudaMemcpy2D)

DECLARE_FUNC_CUDA(cudaMallocHost)
DECLARE_FUNC_CUDA(cudaFreeHost)

DECLARE_FUNC_CUDA(cudaGetDevice)
DECLARE_FUNC_CUDA(cudaSetDevice)

DECLARE_FUNC_CUDA(cudaStreamCreate)
DECLARE_FUNC_CUDA(cudaStreamDestroy)
DECLARE_FUNC_CUDA(cudaStreamSynchronize)

DECLARE_FUNC_CUDA(cudaGraphicsGLRegisterImage)
DECLARE_FUNC_CUDA(cudaGraphicsGLRegisterBuffer)
#if defined(_WIN32)
DECLARE_FUNC_CUDA(cudaGraphicsD3D11RegisterResource)
#endif
DECLARE_FUNC_CUDA(cudaGraphicsUnregisterResource)

DECLARE_FUNC_CUDA(cudaGraphicsMapResources)
DECLARE_FUNC_CUDA(cudaGraphicsUnmapResources)

DECLARE_FUNC_CUDA(cudaGraphicsSubResourceGetMappedArray)
DECLARE_FUNC_CUDA(cudaGraphicsResourceGetMappedPointer)

DECLARE_FUNC_CUDA(cudaGraphicsResourceSetMapFlags)

DECLARE_FUNC_CUDA(cudaMemcpy2DToArray)
DECLARE_FUNC_CUDA(cudaMemcpy2DToArrayAsync)
DECLARE_FUNC_CUDA(cudaMemcpyArrayToArray)
DECLARE_FUNC_CUDA(cudaMemcpy2DArrayToArray)

DECLARE_FUNC_CUDA(cudaMallocArray)
DECLARE_FUNC_CUDA(cudaFreeArray)
DECLARE_FUNC_CUDA(cudaCreateChannelDesc)

#ifdef _extern
#undef _extern
#endif

#ifdef DECLARE_FUNC_CUDA_SIMPLE
#undef DECLARE_FUNC_CUDA_SIMPLE
#endif

#ifdef DECLARE_FUNC_CUDA_WITH_NULLPTR
#undef DECLARE_FUNC_CUDA_WITH_NULLPTR
#endif

#ifdef DECLARE_FUNC_CUDA
#undef DECLARE_FUNC_CUDA
#endif

#ifdef CUDA_DECLARE_EXPORT
#undef CUDA_DECLARE_EXPORT
#endif

#ifdef CUDA_DECLARE_STATIC
#undef CUDA_DECLARE_STATIC
#endif