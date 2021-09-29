#if defined(OPENCL_DECLARE_STATIC)
#define _extern static
//#pragma message("#define _extern static")
#elif defined(OPENCL_DECLARE_EXPORT)
#define _extern extern
//#pragma message("#define _extern extern")
#else
#define _extern 
//#pragma message("#define _extern")
#endif

#define DECLARE_FUNC_OPENCL_SIMPLE(function) \
	_extern FT##function FUNC_OPENCL(function);

#define DECLARE_FUNC_OPENCL_WITH_NULLPTR(function) \
	_extern FT##function FUNC_OPENCL(function) = nullptr;

#if defined(OPENCL_DECLARE_EXPORT) || defined(OPENCL_DECLARE_STATIC)
#define DECLARE_FUNC_OPENCL(function) \
	DECLARE_FUNC_OPENCL_SIMPLE(function)
#else
#define DECLARE_FUNC_OPENCL(function) \
	DECLARE_FUNC_OPENCL_WITH_NULLPTR(function)
#endif

/********************************************************************************************************/

/* Platform API */
DECLARE_FUNC_OPENCL(clGetPlatformIDs)
DECLARE_FUNC_OPENCL(clGetPlatformInfo)

/* Device APIs */
DECLARE_FUNC_OPENCL(clGetDeviceIDs)
DECLARE_FUNC_OPENCL(clGetDeviceInfo)
#ifdef CL_VERSION_1_2
DECLARE_FUNC_OPENCL(clCreateSubDevices)
DECLARE_FUNC_OPENCL(clRetainDevice)
DECLARE_FUNC_OPENCL(clReleaseDevice)
#endif
#ifdef CL_VERSION_2_1
DECLARE_FUNC_OPENCL(clSetDefaultDeviceCommandQueue)
DECLARE_FUNC_OPENCL(clGetDeviceAndHostTimer)
DECLARE_FUNC_OPENCL(clGetHostTimer)
#endif

/* Context APIs */
DECLARE_FUNC_OPENCL(clCreateContext)
DECLARE_FUNC_OPENCL(clCreateContextFromType)
DECLARE_FUNC_OPENCL(clRetainContext)
DECLARE_FUNC_OPENCL(clReleaseContext)
DECLARE_FUNC_OPENCL(clGetContextInfo)

/* Command Queue APIs */
#ifdef CL_VERSION_2_0
DECLARE_FUNC_OPENCL(clCreateCommandQueueWithProperties)
#endif
DECLARE_FUNC_OPENCL(clRetainCommandQueue)
DECLARE_FUNC_OPENCL(clReleaseCommandQueue)
DECLARE_FUNC_OPENCL(clGetCommandQueueInfo)

/* Memory Object APIs */
DECLARE_FUNC_OPENCL(clCreateBuffer)
#ifdef CL_VERSION_1_1
DECLARE_FUNC_OPENCL(clCreateSubBuffer)
#endif
#ifdef CL_VERSION_1_2
DECLARE_FUNC_OPENCL(clCreateImage)
#endif
#ifdef CL_VERSION_2_0
DECLARE_FUNC_OPENCL(clCreatePipe)
#endif
#ifdef CL_VERSION_3_0
DECLARE_FUNC_OPENCL(clCreateBufferWithProperties)
DECLARE_FUNC_OPENCL(clCreateImageWithProperties)
#endif
DECLARE_FUNC_OPENCL(clRetainMemObject)
DECLARE_FUNC_OPENCL(clReleaseMemObject)
DECLARE_FUNC_OPENCL(clGetSupportedImageFormats)
DECLARE_FUNC_OPENCL(clGetMemObjectInfo)
DECLARE_FUNC_OPENCL(clGetImageInfo)
#ifdef CL_VERSION_2_0
DECLARE_FUNC_OPENCL(clGetPipeInfo)
#endif
#ifdef CL_VERSION_1_1
DECLARE_FUNC_OPENCL(clSetMemObjectDestructorCallback)
#endif

/* SVM Allocation APIs */
#ifdef CL_VERSION_2_0
DECLARE_FUNC_OPENCL(clSVMAlloc)
DECLARE_FUNC_OPENCL(clSVMFree)
#endif

/* Sampler APIs */
#ifdef CL_VERSION_2_0
DECLARE_FUNC_OPENCL(clCreateSamplerWithProperties)
#endif
DECLARE_FUNC_OPENCL(clRetainSampler)
DECLARE_FUNC_OPENCL(clReleaseSampler)
DECLARE_FUNC_OPENCL(clGetSamplerInfo)

/* Program Object APIs */
DECLARE_FUNC_OPENCL(clCreateProgramWithSource)
DECLARE_FUNC_OPENCL(clCreateProgramWithBinary)
#ifdef CL_VERSION_1_2
DECLARE_FUNC_OPENCL(clCreateProgramWithBuiltInKernels)
#endif
#ifdef CL_VERSION_2_1
DECLARE_FUNC_OPENCL(clCreateProgramWithIL)
#endif
DECLARE_FUNC_OPENCL(clRetainProgram)
DECLARE_FUNC_OPENCL(clReleaseProgram)
DECLARE_FUNC_OPENCL(clBuildProgram)
#ifdef CL_VERSION_1_2
DECLARE_FUNC_OPENCL(clCompileProgram)
DECLARE_FUNC_OPENCL(clLinkProgram)
#endif
#ifdef CL_VERSION_2_2
DECLARE_FUNC_OPENCL(FTclSetProgramReleaseCallback)
DECLARE_FUNC_OPENCL(FTclSetProgramSpecializationConstant)
#endif
#ifdef CL_VERSION_1_2
DECLARE_FUNC_OPENCL(clUnloadPlatformCompiler)
#endif
DECLARE_FUNC_OPENCL(clGetProgramInfo)
DECLARE_FUNC_OPENCL(clGetProgramBuildInfo)

/* Kernel Object APIs */
DECLARE_FUNC_OPENCL(clCreateKernel)
DECLARE_FUNC_OPENCL(clCreateKernelsInProgram)
#ifdef CL_VERSION_2_1
DECLARE_FUNC_OPENCL(clCloneKernel)
#endif
DECLARE_FUNC_OPENCL(clRetainKernel)
DECLARE_FUNC_OPENCL(clReleaseKernel)
DECLARE_FUNC_OPENCL(clSetKernelArg)
#ifdef CL_VERSION_2_0
DECLARE_FUNC_OPENCL(clSetKernelArgSVMPointer)
DECLARE_FUNC_OPENCL(clSetKernelExecInfo)
#endif
DECLARE_FUNC_OPENCL(clGetKernelInfo)
#ifdef CL_VERSION_1_2
DECLARE_FUNC_OPENCL(clGetKernelArgInfo)
#endif
DECLARE_FUNC_OPENCL(clGetKernelWorkGroupInfo)
#ifdef CL_VERSION_2_1
DECLARE_FUNC_OPENCL(clGetKernelSubGroupInfo)
#endif

/* Event Object APIs */
DECLARE_FUNC_OPENCL(clWaitForEvents)
DECLARE_FUNC_OPENCL(clGetEventInfo)
#ifdef CL_VERSION_1_1
DECLARE_FUNC_OPENCL(clCreateUserEvent)
#endif
DECLARE_FUNC_OPENCL(clRetainEvent)
DECLARE_FUNC_OPENCL(clReleaseEvent)
#ifdef CL_VERSION_1_1
DECLARE_FUNC_OPENCL(clSetUserEventStatus)
DECLARE_FUNC_OPENCL(clSetEventCallback)
#endif

/* Profiling APIs */
DECLARE_FUNC_OPENCL(clGetEventProfilingInfo)

/* Flush and Finish APIs */
DECLARE_FUNC_OPENCL(clFlush)
DECLARE_FUNC_OPENCL(clFinish)

/* Enqueued Commands APIs */
DECLARE_FUNC_OPENCL(clEnqueueReadBuffer)
#ifdef CL_VERSION_1_1
DECLARE_FUNC_OPENCL(clEnqueueReadBufferRect)
#endif
DECLARE_FUNC_OPENCL(clEnqueueWriteBuffer)
#ifdef CL_VERSION_1_1
DECLARE_FUNC_OPENCL(clEnqueueWriteBufferRect)
#endif
#ifdef CL_VERSION_1_2
DECLARE_FUNC_OPENCL(clEnqueueFillBuffer)
#endif
DECLARE_FUNC_OPENCL(clEnqueueCopyBuffer)
#ifdef CL_VERSION_1_1
DECLARE_FUNC_OPENCL(clEnqueueCopyBufferRect)
#endif
DECLARE_FUNC_OPENCL(clEnqueueReadImage)
DECLARE_FUNC_OPENCL(clEnqueueWriteImage)
#ifdef CL_VERSION_1_2
DECLARE_FUNC_OPENCL(clEnqueueFillImage)
#endif
DECLARE_FUNC_OPENCL(clEnqueueCopyImage)
DECLARE_FUNC_OPENCL(clEnqueueCopyImageToBuffer)
DECLARE_FUNC_OPENCL(clEnqueueCopyBufferToImage)
DECLARE_FUNC_OPENCL(clEnqueueMapBuffer)
DECLARE_FUNC_OPENCL(clEnqueueMapImage)
DECLARE_FUNC_OPENCL(clEnqueueUnmapMemObject)
#ifdef CL_VERSION_1_2
DECLARE_FUNC_OPENCL(clEnqueueMigrateMemObjects)
#endif
DECLARE_FUNC_OPENCL(clEnqueueNDRangeKernel)
DECLARE_FUNC_OPENCL(clEnqueueNativeKernel)
#ifdef CL_VERSION_1_2
DECLARE_FUNC_OPENCL(clEnqueueMarkerWithWaitList)
DECLARE_FUNC_OPENCL(clEnqueueBarrierWithWaitList)
#endif
#ifdef CL_VERSION_2_0
DECLARE_FUNC_OPENCL(clEnqueueSVMFree)
DECLARE_FUNC_OPENCL(clEnqueueSVMMemcpy)
DECLARE_FUNC_OPENCL(clEnqueueSVMMemFill)
DECLARE_FUNC_OPENCL(clEnqueueSVMMap)
DECLARE_FUNC_OPENCL(clEnqueueSVMUnmap)
#endif
#ifdef CL_VERSION_2_1
DECLARE_FUNC_OPENCL(clEnqueueSVMMigrateMem)
#endif

#ifdef CL_VERSION_1_2
DECLARE_FUNC_OPENCL(clGetExtensionFunctionAddressForPlatform)
#endif

#ifdef CL_USE_DEPRECATED_OPENCL_1_0_APIS
DECLARE_FUNC_OPENCL(clSetCommandQueueProperty)
#endif

/* Deprecated OpenCL 1.1 APIs */
DECLARE_FUNC_OPENCL(clCreateImage2D)
DECLARE_FUNC_OPENCL(clCreateImage3D)
DECLARE_FUNC_OPENCL(clEnqueueMarker)
DECLARE_FUNC_OPENCL(clEnqueueWaitForEvents)
DECLARE_FUNC_OPENCL(clEnqueueBarrier)
DECLARE_FUNC_OPENCL(clUnloadCompiler)
DECLARE_FUNC_OPENCL(clGetExtensionFunctionAddress)

/* Deprecated OpenCL 2.0 APIs */
DECLARE_FUNC_OPENCL(clCreateCommandQueue)
DECLARE_FUNC_OPENCL(clCreateSampler)
DECLARE_FUNC_OPENCL(clEnqueueTask)

/********************************************************************************************************/

DECLARE_FUNC_OPENCL(clCreateFromGLBuffer)
#ifdef CL_VERSION_1_2
DECLARE_FUNC_OPENCL(clCreateFromGLTexture)
#endif
DECLARE_FUNC_OPENCL(clCreateFromGLRenderbuffer)
DECLARE_FUNC_OPENCL(clGetGLObjectInfo)
DECLARE_FUNC_OPENCL(clGetGLTextureInfo)
DECLARE_FUNC_OPENCL(clEnqueueAcquireGLObjects)
DECLARE_FUNC_OPENCL(clEnqueueReleaseGLObjects)

/* Deprecated OpenCL 1.1 APIs */
//DECLARE_FUNC_OPENCL(clCreateFromGLTexture2D)
//DECLARE_FUNC_OPENCL(clCreateFromGLTexture3D)

DECLARE_FUNC_OPENCL(clGetGLContextInfoKHR)
DECLARE_FUNC_OPENCL(clCreateEventFromGLsyncKHR)

/********************************************************************************************************/

#ifdef _extern
#undef _extern
#endif

#ifdef DECLARE_FUNC_OPENCL_SIMPLE
#undef DECLARE_FUNC_OPENCL_SIMPLE
#endif

#ifdef DECLARE_FUNC_OPENCL_WITH_NULLPTR
#undef DECLARE_FUNC_OPENCL_WITH_NULLPTR
#endif

#ifdef DECLARE_FUNC_OPENCL
#undef DECLARE_FUNC_OPENCL
#endif

#ifdef OPENCL_DECLARE_EXPORT
#undef OPENCL_DECLARE_EXPORT
#endif

#ifdef OPENCL_DECLARE_STATIC
#undef OPENCL_DECLARE_STATIC
#endif