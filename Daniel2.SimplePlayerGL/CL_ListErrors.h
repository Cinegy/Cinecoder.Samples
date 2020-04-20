#pragma once

static const char* clGetErrorString(cl_int err)
{
	switch (err)
	{
		case 0: 	return "CL_SUCCESS";
		case -1: 	return "No OpenCL devices that matched given device type were found.";
		case -2: 	return "No OpenCL compatible device was found.";
		case -3: 	return "OpenCL Compiler perhaps failed to configure itself, or check your OpenCL installation.";
		case -4: 	return "Failed to allocate memory for buffer object.";
		case -5: 	return "failure to allocate resources required by the OpenCL implementation on the defice";
		case -6: 	return "failure to allocate resources required by the OpenCL implementation on the host.";
		case -7: 	return "if the CL_QUEUE_PROFILING_ENABLE flag is not set for the command-queue and if the profiling inforamtion is currently not available";
		case -8: 	return "if source and destination buffers are the same buffer object and the source and destination regions overlap.";
		case -9: 	return "src and dst image do not use the same image format.";
		case -10: 	return "the image format is not supported.";
		case -11: 	return "program build error for given device, Use clGetProgramBuildInfo API call to get the build log of the kernel compilation.";
		case -12: 	return "failed to map the requested region into the host address space. This error does not occur for buffer objects created with CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR.";
		case -13: 	return "no devices in given context associated with buffer for which the origin value is aliigned to the CL_DEVICE_MEM_BASE_ADDR_ALIGN value.";
		case -14: 	return "execution status of any of the events in event list is a negative integer value i.e., error.";
		case -15: 	return "failed to compile the program source.";
		case -16: 	return "Linker unavailable";
		case -17: 	return "failed to link the compiled binaries and perhaps libraries";
		case -18: 	return "given partition name is supported by the inmplementation but input device couldn't be partitioned further";
		case -19: 	return "argument information is not available for the given kernel";

		case -30: 	return "values passed in the flags parameter is not valid";
		case -31: 	return "device type specified is not valid, its returned by clCreateContextFromType / clGetDeviceIDs";
		case -32: 	return "the specified platform is not a valid platform, its returned by clGetPlatformInfo /clGetDeviceIDs / clCreateContext / clCreateContextFromType";
		case -33: 	return "device/s specified are not valid";
		case -34: 	return "the given context is invalid OpenCL context, or the context associated with certain parameters are not the same";
		case -35: 	return "specified properties are valid but are not supported by the device, its returned by clCreateCommandQueue / clSetCommandQueueProperty";
		case -36: 	return "the specified command-queue is not a valid command-queue";
		case -37: 	return "host pointer is NULL and CL_MEM_COPY_HOST_PTR or CL_MEM_USE_HOST_PTR are set in flags or if host_ptr is not NULL but CL_MEM_COPY_HOST_PTR or CL_MEM_USE_HOST_PTR are not set in flags. returned by clCreateBuffer / clCreateImage2D / clCreateImage3D";
		case -38: 	return "the passed parameter is not a valid memory, image, or buffer object";
		case -39: 	return "image format specified is not valid or is NULL, clCreateImage2D /clCreateImage3D returns this.";
		case -40: 	return "Its returned by create Image functions 2D/3D, if specified image width or height are outbound or 0";
		case -41: 	return "specified sampler is an invalid sampler object";
		case -42: 	return "program binary is not a valid binary for the specified device, returned by clBuildProgram / clCreateProgramWithBinary";
		case -43: 	return "the given build options are not valid";
		case -44: 	return "the given program is an invalid program object, returned by clRetainProgram / clReleaseProgram / clBuildProgram / clGetProgramInfo / clGetProgramBuildInfo / clCreateKernel / clCreateKernelsInProgram";
		case -45: 	return "if there is no successfully built executable for program returned by clCreateKernel, there is no device in program then returned by clCreateKernelsInProgram, if no successfully built program executable present for device associated with command queue then returned by clEnqueueNDRangeKernel / clEnqueueTask";
		case -46: 	return "mentioned kernel name is not found in program";
		case -47: 	return "arguments mismatch for the __kernel function definition and the passed ones, returned by clCreateKernel";
		case -48: 	return "specified kernel is an invalid kernel object";
		case -49: 	return "clSetKernelArg if an invalid argument index is specified";
		case -50: 	return "the argument value specified is NULL, returned by clSetKernelArg";
		case -51: 	return "the given argument size (arg_size) do not match size of the data type for an argument, returned by clSetKernelArg";
		case -52: 	return "the kernel argument values have not been specified, returned by clEnqueueNDRangeKernel / clEnqueueTask";
		case -53: 	return "given work dimension is an invalid value, returned by clEnqueueNDRangeKernel";
		case -54: 	return "the specified local workgroup size and number of workitems specified by global workgroup size is not evenly divisible by local workgroup size";
		case -55: 	return "no. of workitems specified in any of local work group sizes is greater than the corresponding values specified by CL_DEVICE_MAX_WORK_ITEM_SIZES in that particular dimension";
		case -56: 	return "global_work_offset is not NULL. Must currently be a NULL value. In a future revision of OpenCL, global_work_offset can be used but not until OCL 1.2";
		case -57: 	return "event wait list is NULL and (no. of events in wait list > 0), or event wait list is not NULL and no. of events in wait list is 0, or specified event objects are not valid events";
		case -58: 	return "invalid event objects spacified.";
		case -59: 	return "CL_INVALID_OPERATION";
		case -60: 	return "not a valid GL buffer object";
		case -61: 	return "the value of the parameter size is 0 or exceeds CL_DEVICE_MAX_MEM_ALLOC_SIZE for all devices specified in the parameter context, returned by clCreateBuffer";
		case -62: 	return "CL_INVALID_MIP_LEVEL";
		case -63: 	return "specified global work size is NULL, or any of the values specified in global work dimensions are 0 or exceeds the range given by the sizeof(size_t) for the device on which the kernel will be enqueued, returned by clEnqueueNDRangeKernel";
		case -64: 	return "context property name in properties is not a supported property name, returned by clCreateContext";
		case -65: 	return "values specified in image description are invalid.";
		case -66: 	return "the compiller options specified by options are invaid, returned by clCompileProgram";
		case -67: 	return "linker options specified by options are invalid, returned by clLinkProgram.";
		case -68: 	return "invalid device partition count.";
		case -69: 	return "if pipe_packet_size is 0 or the pipe_packet_size exceeds CL_DEVICE_PIPE_MAX_PACKET_SIZE value";
		case -70: 	return "CL_INVALID_DEVICE_QUEUE";

		default: return "Unknown OpenCL error";
	}
}

static const char * get_error_string(cl_int err)
{
	switch(err)
	{
		case 0: 	return "CL_SUCCESS";
		case -1: 	return "CL_DEVICE_NOT_FOUND";
		case -2: 	return "CL_DEVICE_NOT_AVAILABLE";
		case -3: 	return "CL_COMPILER_NOT_AVAILABLE";
		case -4: 	return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
		case -5: 	return "CL_OUT_OF_RESOURCES";
		case -6: 	return "CL_OUT_OF_HOST_MEMORY";
		case -7: 	return "CL_PROFILING_INFO_NOT_AVAILABLE";
		case -8: 	return "CL_MEM_COPY_OVERLAP";
		case -9: 	return "CL_IMAGE_FORMAT_MISMATCH";
		case -10: 	return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
		case -11: 	return "CL_BUILD_PROGRAM_FAILURE";
		case -12: 	return "CL_MAP_FAILURE";
		case -13: 	return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
		case -14: 	return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
		case -15: 	return "CL_COMPILE_PROGRAM_FAILURE";
		case -16: 	return "CL_LINKER_NOT_AVAILABLE";
		case -17: 	return "CL_LINK_PROGRAM_FAILURE";
		case -18: 	return "CL_DEVICE_PARTITION_FAILED";
		case -19: 	return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

		case -30: 	return "CL_INVALID_VALUE";
		case -31: 	return "CL_INVALID_DEVICE_TYPE";
		case -32: 	return "CL_INVALID_PLATFORM";
		case -33: 	return "CL_INVALID_DEVICE";
		case -34: 	return "CL_INVALID_CONTEXT";
		case -35: 	return "CL_INVALID_QUEUE_PROPERTIES";
		case -36: 	return "CL_INVALID_COMMAND_QUEUE";
		case -37: 	return "CL_INVALID_HOST_PTR";
		case -38: 	return "CL_INVALID_MEM_OBJECT";
		case -39: 	return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
		case -40: 	return "CL_INVALID_IMAGE_SIZE";
		case -41: 	return "CL_INVALID_SAMPLER";
		case -42: 	return "CL_INVALID_BINARY";
		case -43: 	return "CL_INVALID_BUILD_OPTIONS";
		case -44: 	return "CL_INVALID_PROGRAM";
		case -45: 	return "CL_INVALID_PROGRAM_EXECUTABLE";
		case -46: 	return "CL_INVALID_KERNEL_NAME";
		case -47: 	return "CL_INVALID_KERNEL_DEFINITION";
		case -48: 	return "CL_INVALID_KERNEL";
		case -49: 	return "CL_INVALID_ARG_INDEX";
		case -50: 	return "CL_INVALID_ARG_VALUE";
		case -51: 	return "CL_INVALID_ARG_SIZE";
		case -52: 	return "CL_INVALID_KERNEL_ARGS";
		case -53: 	return "CL_INVALID_WORK_DIMENSION";
		case -54: 	return "CL_INVALID_WORK_GROUP_SIZE";
		case -55: 	return "CL_INVALID_WORK_ITEM_SIZE";
		case -56: 	return "CL_INVALID_GLOBAL_OFFSET";
		case -57: 	return "CL_INVALID_EVENT_WAIT_LIST";
		case -58: 	return "CL_INVALID_EVENT";
		case -59: 	return "CL_INVALID_OPERATION";
		case -60: 	return "CL_INVALID_GL_OBJECT";
		case -61: 	return "CL_INVALID_BUFFER_SIZE";
		case -62: 	return "CL_INVALID_MIP_LEVEL";
		case -63: 	return "CL_INVALID_GLOBAL_WORK_SIZE";
		case -64: 	return "CL_INVALID_PROPERTY";
		case -65: 	return "CL_INVALID_IMAGE_DESCRIPTOR";
		case -66: 	return "CL_INVALID_COMPILER_OPTIONS";
		case -67: 	return "CL_INVALID_LINKER_OPTIONS";
		case -68: 	return "CL_INVALID_DEVICE_PARTITION_COUNT";
		case -69: 	return "CL_INVALID_PIPE_SIZE";
		case -70: 	return "CL_INVALID_DEVICE_QUEUE";
		default: return "Unknown OpenCL error";
	}
}

cl_int g_clLastError = 0;

#define __clerror  \
	cl_int error = CL_SUCCESS;

//#ifdef _DEBUG
//#define __rcl	{ g_clLastError = error; \
//	if (g_clLastError != CL_SUCCESS)  \
//	{ \
//		char str[1024]; \
//		sprintf_s(str, "OpenCL error %d %s (%s %d)\n", static_cast<int>(g_clLastError), clGetErrorString(g_clLastError), __FILE__,__LINE__); \
//		OutputDebugStringA(str); \
//		return static_cast<int>(g_clLastError); \
//	}} 
//
//#define __vrcl	{ g_clLastError = error; \
//	if (g_clLastError != CL_SUCCESS)  \
//	{ \
//		char str[1024]; \
//		sprintf_s(str, "OpenCL error %d %s (%s %d)\n", static_cast<int>(g_clLastError), clGetErrorString(g_clLastError), __FILE__,__LINE__); \
//		OutputDebugStringA(str); \
//	}} 
//#else
#define __rcl	{ g_clLastError = error; \
	if (g_clLastError != CL_SUCCESS)  \
	{ \
		printf("OpenCL error %d %s (%s %d)\n", static_cast<int>(g_clLastError), clGetErrorString(g_clLastError), __FILE__,__LINE__); \
		return static_cast<int>(g_clLastError); \
	}} 

#define __vrcl	{ g_clLastError = error; \
	if (g_clLastError != CL_SUCCESS)  \
	{ \
		printf("OpenCL error %d %s (%s %d)\n", static_cast<int>(g_clLastError), clGetErrorString(g_clLastError), __FILE__,__LINE__); \
	}} 

#define __ccl	{ g_clLastError = error; \
	if (g_clLastError != CL_SUCCESS)  \
	{ \
		printf("OpenCL error %d %s (%s %d)\n", static_cast<int>(g_clLastError), clGetErrorString(g_clLastError), __FILE__,__LINE__); \
		continue; \
	}} 

//#endif

#define CL_CHECK(_expr)                                                         \
   do {                                                                         \
     cl_int _err = _expr;                                                       \
     if (_err == CL_SUCCESS)                                                    \
       break;                                                                   \
     fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);   \
   } while (0)

#define CL_CHECK_ERR(_ret, _expr)                                               \
   {                                                                           \
     cl_int _err = CL_INVALID_VALUE;                                            \
     _ret = _expr;									                            \
     if (_err != CL_SUCCESS) {                                                  \
       fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
     }                                                                          \
     _ret;                                                                      \
   }
