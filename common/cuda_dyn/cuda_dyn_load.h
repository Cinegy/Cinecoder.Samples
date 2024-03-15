#pragma once

#if defined(_WIN32)
typedef interface ID3D11Resource ID3D11Resource;
#endif

#define cudaArrayDefault                    0x00  /**< Default CUDA array allocation flag */
#define cudaArrayLayered                    0x01  /**< Must be set in cudaMalloc3DArray to create a layered CUDA array */
#define cudaArraySurfaceLoadStore           0x02  /**< Must be set in cudaMallocArray or cudaMalloc3DArray in order to bind surfaces to the CUDA array */
#define cudaArrayCubemap                    0x04  /**< Must be set in cudaMalloc3DArray to create a cubemap CUDA array */
#define cudaArrayTextureGather              0x08  /**< Must be set in cudaMallocArray or cudaMalloc3DArray in order to perform texture gather operations on the CUDA array */

#define cudaHostAllocDefault                0x00  /**< Default page-locked allocation flag */
#define cudaHostAllocPortable               0x01  /**< Pinned memory accessible by all CUDA contexts */
#define cudaHostAllocMapped                 0x02  /**< Map allocation into device space */
#define cudaHostAllocWriteCombined          0x04  /**< Write-combined memory */

#define cudaHostRegisterDefault             0x00  /**< Default host memory registration flag */
#define cudaHostRegisterPortable            0x01  /**< Pinned memory accessible by all CUDA contexts */
#define cudaHostRegisterMapped              0x02  /**< Map registered memory into device space */
#define cudaHostRegisterIoMemory            0x04  /**< Memory-mapped I/O space */
#define cudaHostRegisterReadOnly            0x08  /**< Memory-mapped read-only */

#define cudaPeerAccessDefault               0x00  /**< Default peer addressing enable flag */

#define cudaStreamDefault                   0x00  /**< Default stream flag */
#define cudaStreamNonBlocking               0x01  /**< Stream does not synchronize with stream 0 (the NULL stream) */

#define cudaDeviceScheduleMask              0x07  /**< Device schedule flags mask */
#define cudaDeviceMapHost                   0x08  /**< Device flag - Support mapped pinned allocations */
#define cudaDeviceLmemResizeToMax           0x10  /**< Device flag - Keep local memory allocation after launch */
#define cudaDeviceSyncMemops                0x80  /**< Device flag - Use synchronous behavior for cudaMemcpy/cudaMemset */
#define cudaDeviceMask                      0xff  /**< Device flags mask */

#ifndef CU_UUID_HAS_BEEN_DEFINED
#define CU_UUID_HAS_BEEN_DEFINED
struct CUuuid_st {
    char bytes[16];
};
typedef struct CUuuid_st CUuuid;
#endif
typedef struct CUuuid_st cudaUUID_t;

/**
 * CUDA device attributes
 */
enum cudaDeviceAttr
{
    cudaDevAttrMaxThreadsPerBlock = 1,  /**< Maximum number of threads per block */
    cudaDevAttrMaxBlockDimX = 2,  /**< Maximum block dimension X */
    cudaDevAttrMaxBlockDimY = 3,  /**< Maximum block dimension Y */
    cudaDevAttrMaxBlockDimZ = 4,  /**< Maximum block dimension Z */
    cudaDevAttrMaxGridDimX = 5,  /**< Maximum grid dimension X */
    cudaDevAttrMaxGridDimY = 6,  /**< Maximum grid dimension Y */
    cudaDevAttrMaxGridDimZ = 7,  /**< Maximum grid dimension Z */
    cudaDevAttrMaxSharedMemoryPerBlock = 8,  /**< Maximum shared memory available per block in bytes */
    cudaDevAttrTotalConstantMemory = 9,  /**< Memory available on device for __constant__ variables in a CUDA C kernel in bytes */
    cudaDevAttrWarpSize = 10, /**< Warp size in threads */
    cudaDevAttrMaxPitch = 11, /**< Maximum pitch in bytes allowed by memory copies */
    cudaDevAttrMaxRegistersPerBlock = 12, /**< Maximum number of 32-bit registers available per block */
    cudaDevAttrClockRate = 13, /**< Peak clock frequency in kilohertz */
    cudaDevAttrTextureAlignment = 14, /**< Alignment requirement for textures */
    cudaDevAttrGpuOverlap = 15, /**< Device can possibly copy memory and execute a kernel concurrently */
    cudaDevAttrMultiProcessorCount = 16, /**< Number of multiprocessors on device */
    cudaDevAttrKernelExecTimeout = 17, /**< Specifies whether there is a run time limit on kernels */
    cudaDevAttrIntegrated = 18, /**< Device is integrated with host memory */
    cudaDevAttrCanMapHostMemory = 19, /**< Device can map host memory into CUDA address space */
    cudaDevAttrComputeMode = 20, /**< Compute mode (See ::cudaComputeMode for details) */
    cudaDevAttrMaxTexture1DWidth = 21, /**< Maximum 1D texture width */
    cudaDevAttrMaxTexture2DWidth = 22, /**< Maximum 2D texture width */
    cudaDevAttrMaxTexture2DHeight = 23, /**< Maximum 2D texture height */
    cudaDevAttrMaxTexture3DWidth = 24, /**< Maximum 3D texture width */
    cudaDevAttrMaxTexture3DHeight = 25, /**< Maximum 3D texture height */
    cudaDevAttrMaxTexture3DDepth = 26, /**< Maximum 3D texture depth */
    cudaDevAttrMaxTexture2DLayeredWidth = 27, /**< Maximum 2D layered texture width */
    cudaDevAttrMaxTexture2DLayeredHeight = 28, /**< Maximum 2D layered texture height */
    cudaDevAttrMaxTexture2DLayeredLayers = 29, /**< Maximum layers in a 2D layered texture */
    cudaDevAttrSurfaceAlignment = 30, /**< Alignment requirement for surfaces */
    cudaDevAttrConcurrentKernels = 31, /**< Device can possibly execute multiple kernels concurrently */
    cudaDevAttrEccEnabled = 32, /**< Device has ECC support enabled */
    cudaDevAttrPciBusId = 33, /**< PCI bus ID of the device */
    cudaDevAttrPciDeviceId = 34, /**< PCI device ID of the device */
    cudaDevAttrTccDriver = 35, /**< Device is using TCC driver model */
    cudaDevAttrMemoryClockRate = 36, /**< Peak memory clock frequency in kilohertz */
    cudaDevAttrGlobalMemoryBusWidth = 37, /**< Global memory bus width in bits */
    cudaDevAttrL2CacheSize = 38, /**< Size of L2 cache in bytes */
    cudaDevAttrMaxThreadsPerMultiProcessor = 39, /**< Maximum resident threads per multiprocessor */
    cudaDevAttrAsyncEngineCount = 40, /**< Number of asynchronous engines */
    cudaDevAttrUnifiedAddressing = 41, /**< Device shares a unified address space with the host */
    cudaDevAttrMaxTexture1DLayeredWidth = 42, /**< Maximum 1D layered texture width */
    cudaDevAttrMaxTexture1DLayeredLayers = 43, /**< Maximum layers in a 1D layered texture */
    cudaDevAttrMaxTexture2DGatherWidth = 45, /**< Maximum 2D texture width if cudaArrayTextureGather is set */
    cudaDevAttrMaxTexture2DGatherHeight = 46, /**< Maximum 2D texture height if cudaArrayTextureGather is set */
    cudaDevAttrMaxTexture3DWidthAlt = 47, /**< Alternate maximum 3D texture width */
    cudaDevAttrMaxTexture3DHeightAlt = 48, /**< Alternate maximum 3D texture height */
    cudaDevAttrMaxTexture3DDepthAlt = 49, /**< Alternate maximum 3D texture depth */
    cudaDevAttrPciDomainId = 50, /**< PCI domain ID of the device */
    cudaDevAttrTexturePitchAlignment = 51, /**< Pitch alignment requirement for textures */
    cudaDevAttrMaxTextureCubemapWidth = 52, /**< Maximum cubemap texture width/height */
    cudaDevAttrMaxTextureCubemapLayeredWidth = 53, /**< Maximum cubemap layered texture width/height */
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54, /**< Maximum layers in a cubemap layered texture */
    cudaDevAttrMaxSurface1DWidth = 55, /**< Maximum 1D surface width */
    cudaDevAttrMaxSurface2DWidth = 56, /**< Maximum 2D surface width */
    cudaDevAttrMaxSurface2DHeight = 57, /**< Maximum 2D surface height */
    cudaDevAttrMaxSurface3DWidth = 58, /**< Maximum 3D surface width */
    cudaDevAttrMaxSurface3DHeight = 59, /**< Maximum 3D surface height */
    cudaDevAttrMaxSurface3DDepth = 60, /**< Maximum 3D surface depth */
    cudaDevAttrMaxSurface1DLayeredWidth = 61, /**< Maximum 1D layered surface width */
    cudaDevAttrMaxSurface1DLayeredLayers = 62, /**< Maximum layers in a 1D layered surface */
    cudaDevAttrMaxSurface2DLayeredWidth = 63, /**< Maximum 2D layered surface width */
    cudaDevAttrMaxSurface2DLayeredHeight = 64, /**< Maximum 2D layered surface height */
    cudaDevAttrMaxSurface2DLayeredLayers = 65, /**< Maximum layers in a 2D layered surface */
    cudaDevAttrMaxSurfaceCubemapWidth = 66, /**< Maximum cubemap surface width */
    cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67, /**< Maximum cubemap layered surface width */
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68, /**< Maximum layers in a cubemap layered surface */
    cudaDevAttrMaxTexture1DLinearWidth = 69, /**< Maximum 1D linear texture width */
    cudaDevAttrMaxTexture2DLinearWidth = 70, /**< Maximum 2D linear texture width */
    cudaDevAttrMaxTexture2DLinearHeight = 71, /**< Maximum 2D linear texture height */
    cudaDevAttrMaxTexture2DLinearPitch = 72, /**< Maximum 2D linear texture pitch in bytes */
    cudaDevAttrMaxTexture2DMipmappedWidth = 73, /**< Maximum mipmapped 2D texture width */
    cudaDevAttrMaxTexture2DMipmappedHeight = 74, /**< Maximum mipmapped 2D texture height */
    cudaDevAttrComputeCapabilityMajor = 75, /**< Major compute capability version number */
    cudaDevAttrComputeCapabilityMinor = 76, /**< Minor compute capability version number */
    cudaDevAttrMaxTexture1DMipmappedWidth = 77, /**< Maximum mipmapped 1D texture width */
    cudaDevAttrStreamPrioritiesSupported = 78, /**< Device supports stream priorities */
    cudaDevAttrGlobalL1CacheSupported = 79, /**< Device supports caching globals in L1 */
    cudaDevAttrLocalL1CacheSupported = 80, /**< Device supports caching locals in L1 */
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81, /**< Maximum shared memory available per multiprocessor in bytes */
    cudaDevAttrMaxRegistersPerMultiprocessor = 82, /**< Maximum number of 32-bit registers available per multiprocessor */
    cudaDevAttrManagedMemory = 83, /**< Device can allocate managed memory on this system */
    cudaDevAttrIsMultiGpuBoard = 84, /**< Device is on a multi-GPU board */
    cudaDevAttrMultiGpuBoardGroupID = 85, /**< Unique identifier for a group of devices on the same multi-GPU board */
    cudaDevAttrHostNativeAtomicSupported = 86, /**< Link between the device and the host supports native atomic operations */
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87, /**< Ratio of single precision performance (in floating-point operations per second) to double precision performance */
    cudaDevAttrPageableMemoryAccess = 88, /**< Device supports coherently accessing pageable memory without calling cudaHostRegister on it */
    cudaDevAttrConcurrentManagedAccess = 89, /**< Device can coherently access managed memory concurrently with the CPU */
    cudaDevAttrComputePreemptionSupported = 90, /**< Device supports Compute Preemption */
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91, /**< Device can access host registered memory at the same virtual address as the CPU */
    cudaDevAttrReserved92 = 92,
    cudaDevAttrReserved93 = 93,
    cudaDevAttrReserved94 = 94,
    cudaDevAttrCooperativeLaunch = 95, /**< Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel*/
    cudaDevAttrCooperativeMultiDeviceLaunch = 96, /**< Deprecated, cudaLaunchCooperativeKernelMultiDevice is deprecated. */
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97, /**< The maximum optin shared memory per block. This value may vary by chip. See ::cudaFuncSetAttribute */
    cudaDevAttrCanFlushRemoteWrites = 98, /**< Device supports flushing of outstanding remote writes. */
    cudaDevAttrHostRegisterSupported = 99, /**< Device supports host memory registration via ::cudaHostRegister. */
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100, /**< Device accesses pageable memory via the host's page tables. */
    cudaDevAttrDirectManagedMemAccessFromHost = 101, /**< Host can directly access managed memory on the device without migration. */
    cudaDevAttrMaxBlocksPerMultiprocessor = 106, /**< Maximum number of blocks per multiprocessor */
    cudaDevAttrMaxPersistingL2CacheSize = 108, /**< Maximum L2 persisting lines capacity setting in bytes. */
    cudaDevAttrMaxAccessPolicyWindowSize = 109, /**< Maximum value of cudaAccessPolicyWindow::num_bytes. */
    cudaDevAttrReservedSharedMemoryPerBlock = 111, /**< Shared memory reserved by CUDA driver per block in bytes */
    cudaDevAttrSparseCudaArraySupported = 112, /**< Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays */
    cudaDevAttrHostRegisterReadOnlySupported = 113,  /**< Device supports using the ::cudaHostRegister flag cudaHostRegisterReadOnly to register memory that must be mapped as read-only to the GPU */
    cudaDevAttrTimelineSemaphoreInteropSupported = 114,  /**< External timeline semaphore interop is supported on the device */
    cudaDevAttrMaxTimelineSemaphoreInteropSupported = 114,  /**< Deprecated, External timeline semaphore interop is supported on the device */
    cudaDevAttrMemoryPoolsSupported = 115, /**< Device supports using the ::cudaMallocAsync and ::cudaMemPool family of APIs */
    cudaDevAttrGPUDirectRDMASupported = 116, /**< Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see https://docs.nvidia.com/cuda/gpudirect-rdma for more information) */
    cudaDevAttrGPUDirectRDMAFlushWritesOptions = 117, /**< The returned attribute shall be interpreted as a bitmask, where the individual bits are listed in the ::cudaFlushGPUDirectRDMAWritesOptions enum */
    cudaDevAttrGPUDirectRDMAWritesOrdering = 118, /**< GPUDirect RDMA writes to the device do not need to be flushed for consumers within the scope indicated by the returned attribute. See ::cudaGPUDirectRDMAWritesOrdering for the numerical values returned here. */
    cudaDevAttrMemoryPoolSupportedHandleTypes = 119, /**< Handle types supported with mempool based IPC */
    cudaDevAttrClusterLaunch = 120, /**< Indicates device supports cluster launch */
    cudaDevAttrDeferredMappingCudaArraySupported = 121, /**< Device supports deferred mapping CUDA arrays and CUDA mipmapped arrays */
    cudaDevAttrReserved122 = 122,
    cudaDevAttrReserved123 = 123,
    cudaDevAttrReserved124 = 124,
    cudaDevAttrIpcEventSupport = 125, /**< Device supports IPC Events. */
    cudaDevAttrMemSyncDomainCount = 126, /**< Number of memory synchronization domains the device supports. */
    cudaDevAttrReserved127 = 127,
    cudaDevAttrReserved128 = 128,
    cudaDevAttrReserved129 = 129,
    cudaDevAttrReserved132 = 132,
    cudaDevAttrMax
};

/**
 * CUDA device properties
 */
struct cudaDeviceProp
{
    char         name[256];                  /**< ASCII string identifying device */
    cudaUUID_t   uuid;                       /**< 16-byte unique identifier */
    char         luid[8];                    /**< 8-byte locally unique identifier. Value is undefined on TCC and non-Windows platforms */
    unsigned int luidDeviceNodeMask;         /**< LUID device node mask. Value is undefined on TCC and non-Windows platforms */
    size_t       totalGlobalMem;             /**< Global memory available on device in bytes */
    size_t       sharedMemPerBlock;          /**< Shared memory available per block in bytes */
    int          regsPerBlock;               /**< 32-bit registers available per block */
    int          warpSize;                   /**< Warp size in threads */
    size_t       memPitch;                   /**< Maximum pitch in bytes allowed by memory copies */
    int          maxThreadsPerBlock;         /**< Maximum number of threads per block */
    int          maxThreadsDim[3];           /**< Maximum size of each dimension of a block */
    int          maxGridSize[3];             /**< Maximum size of each dimension of a grid */
    int          clockRate;                  /**< Deprecated, Clock frequency in kilohertz */
    size_t       totalConstMem;              /**< Constant memory available on device in bytes */
    int          major;                      /**< Major compute capability */
    int          minor;                      /**< Minor compute capability */
    size_t       textureAlignment;           /**< Alignment requirement for textures */
    size_t       texturePitchAlignment;      /**< Pitch alignment requirement for texture references bound to pitched memory */
    int          deviceOverlap;              /**< Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount. */
    int          multiProcessorCount;        /**< Number of multiprocessors on device */
    int          kernelExecTimeoutEnabled;   /**< Deprecated, Specified whether there is a run time limit on kernels */
    int          integrated;                 /**< Device is integrated as opposed to discrete */
    int          canMapHostMemory;           /**< Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer */
    int          computeMode;                /**< Deprecated, Compute mode (See ::cudaComputeMode) */
    int          maxTexture1D;               /**< Maximum 1D texture size */
    int          maxTexture1DMipmap;         /**< Maximum 1D mipmapped texture size */
    int          maxTexture1DLinear;         /**< Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or cuDeviceGetTexture1DLinearMaxWidth() instead. */
    int          maxTexture2D[2];            /**< Maximum 2D texture dimensions */
    int          maxTexture2DMipmap[2];      /**< Maximum 2D mipmapped texture dimensions */
    int          maxTexture2DLinear[3];      /**< Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory */
    int          maxTexture2DGather[2];      /**< Maximum 2D texture dimensions if texture gather operations have to be performed */
    int          maxTexture3D[3];            /**< Maximum 3D texture dimensions */
    int          maxTexture3DAlt[3];         /**< Maximum alternate 3D texture dimensions */
    int          maxTextureCubemap;          /**< Maximum Cubemap texture dimensions */
    int          maxTexture1DLayered[2];     /**< Maximum 1D layered texture dimensions */
    int          maxTexture2DLayered[3];     /**< Maximum 2D layered texture dimensions */
    int          maxTextureCubemapLayered[2];/**< Maximum Cubemap layered texture dimensions */
    int          maxSurface1D;               /**< Maximum 1D surface size */
    int          maxSurface2D[2];            /**< Maximum 2D surface dimensions */
    int          maxSurface3D[3];            /**< Maximum 3D surface dimensions */
    int          maxSurface1DLayered[2];     /**< Maximum 1D layered surface dimensions */
    int          maxSurface2DLayered[3];     /**< Maximum 2D layered surface dimensions */
    int          maxSurfaceCubemap;          /**< Maximum Cubemap surface dimensions */
    int          maxSurfaceCubemapLayered[2];/**< Maximum Cubemap layered surface dimensions */
    size_t       surfaceAlignment;           /**< Alignment requirements for surfaces */
    int          concurrentKernels;          /**< Device can possibly execute multiple kernels concurrently */
    int          ECCEnabled;                 /**< Device has ECC support enabled */
    int          pciBusID;                   /**< PCI bus ID of the device */
    int          pciDeviceID;                /**< PCI device ID of the device */
    int          pciDomainID;                /**< PCI domain ID of the device */
    int          tccDriver;                  /**< 1 if device is a Tesla device using TCC driver, 0 otherwise */
    int          asyncEngineCount;           /**< Number of asynchronous engines */
    int          unifiedAddressing;          /**< Device shares a unified address space with the host */
    int          memoryClockRate;            /**< Deprecated, Peak memory clock frequency in kilohertz */
    int          memoryBusWidth;             /**< Global memory bus width in bits */
    int          l2CacheSize;                /**< Size of L2 cache in bytes */
    int          persistingL2CacheMaxSize;   /**< Device's maximum l2 persisting lines capacity setting in bytes */
    int          maxThreadsPerMultiProcessor;/**< Maximum resident threads per multiprocessor */
    int          streamPrioritiesSupported;  /**< Device supports stream priorities */
    int          globalL1CacheSupported;     /**< Device supports caching globals in L1 */
    int          localL1CacheSupported;      /**< Device supports caching locals in L1 */
    size_t       sharedMemPerMultiprocessor; /**< Shared memory available per multiprocessor in bytes */
    int          regsPerMultiprocessor;      /**< 32-bit registers available per multiprocessor */
    int          managedMemory;              /**< Device supports allocating managed memory on this system */
    int          isMultiGpuBoard;            /**< Device is on a multi-GPU board */
    int          multiGpuBoardGroupID;       /**< Unique identifier for a group of devices on the same multi-GPU board */
    int          hostNativeAtomicSupported;  /**< Link between the device and the host supports native atomic operations */
    int          singleToDoublePrecisionPerfRatio; /**< Deprecated, Ratio of single precision performance (in floating-point operations per second) to double precision performance */
    int          pageableMemoryAccess;       /**< Device supports coherently accessing pageable memory without calling cudaHostRegister on it */
    int          concurrentManagedAccess;    /**< Device can coherently access managed memory concurrently with the CPU */
    int          computePreemptionSupported; /**< Device supports Compute Preemption */
    int          canUseHostPointerForRegisteredMem; /**< Device can access host registered memory at the same virtual address as the CPU */
    int          cooperativeLaunch;          /**< Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel */
    int          cooperativeMultiDeviceLaunch; /**< Deprecated, cudaLaunchCooperativeKernelMultiDevice is deprecated. */
    size_t       sharedMemPerBlockOptin;     /**< Per device maximum shared memory per block usable by special opt in */
    int          pageableMemoryAccessUsesHostPageTables; /**< Device accesses pageable memory via the host's page tables */
    int          directManagedMemAccessFromHost; /**< Host can directly access managed memory on the device without migration. */
    int          maxBlocksPerMultiProcessor; /**< Maximum number of resident blocks per multiprocessor */
    int          accessPolicyMaxWindowSize;  /**< The maximum value of ::cudaAccessPolicyWindow::num_bytes. */
    size_t       reservedSharedMemPerBlock;  /**< Shared memory reserved by CUDA driver per block in bytes */
    int          hostRegisterSupported;      /**< Device supports host memory registration via ::cudaHostRegister. */
    int          sparseCudaArraySupported;   /**< 1 if the device supports sparse CUDA arrays and sparse CUDA mipmapped arrays, 0 otherwise */
    int          hostRegisterReadOnlySupported; /**< Device supports using the ::cudaHostRegister flag cudaHostRegisterReadOnly to register memory that must be mapped as read-only to the GPU */
    int          timelineSemaphoreInteropSupported; /**< External timeline semaphore interop is supported on the device */
    int          memoryPoolsSupported;       /**< 1 if the device supports using the cudaMallocAsync and cudaMemPool family of APIs, 0 otherwise */
    int          gpuDirectRDMASupported;     /**< 1 if the device supports GPUDirect RDMA APIs, 0 otherwise */
    unsigned int gpuDirectRDMAFlushWritesOptions; /**< Bitmask to be interpreted according to the ::cudaFlushGPUDirectRDMAWritesOptions enum */
    int          gpuDirectRDMAWritesOrdering;/**< See the ::cudaGPUDirectRDMAWritesOrdering enum for numerical values */
    unsigned int memoryPoolSupportedHandleTypes; /**< Bitmask of handle types supported with mempool-based IPC */
    int          deferredMappingCudaArraySupported; /**< 1 if the device supports deferred mapping CUDA arrays and CUDA mipmapped arrays */
    int          ipcEventSupported;          /**< Device supports IPC Events. */
    int          clusterLaunch;              /**< Indicates device supports cluster launch */
    int          unifiedFunctionPointers;    /**< Indicates device supports unified pointers */
    int          reserved2[2];
    int          reserved[61];               /**< Reserved for future use */
};

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

typedef cudaError_t(*FTcudaHostAlloc)(void** pHost, size_t size, unsigned int flags);
typedef cudaError_t(*FTcudaHostGetDevicePointer)(void** pDevice, void* pHost, unsigned int flags);
typedef cudaError_t(*FTcudaSetDeviceFlags)(unsigned int flags);
typedef cudaError_t(*FTcudaHostRegister)(void* ptr, size_t size, unsigned int flags);
typedef cudaError_t(*FTcudaHostUnregister)(void* ptr);

typedef cudaError_t(*FTcudaDeviceGetAttribute)(int* value, enum cudaDeviceAttr attr, int device);
typedef cudaError_t(*FTcudaGetDeviceProperties)(struct cudaDeviceProp* prop, int device);

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
