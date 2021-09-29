#pragma once

//#ifdef OPENCL_WRAPPER

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS // for clGetExtensionFunctionAddress
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#define CL_TARGET_OPENCL_VERSION 200

//#define CL_VERSION_1_0
//#define CL_VERSION_1_1
//#define CL_VERSION_1_2
//#define CL_VERSION_2_0

#ifdef __APPLE__
#include <OpenCL/cl_version.h>
#include <OpenCL/cl_platform.h>
#else
#include <CL/cl_version.h>
#include <CL/cl_platform.h>
#endif

#include "opencl_def_cl.h"
#include "opencl_def_cl_gl.h"

#define FUNC_OPENCL(func) func

#define CHECK_FUNC_OPENCL(func) \
	if (!FUNC_OPENCL(func)) { \
		fprintf(stderr, "OpenCL init error: failed to find required functions (File: %s Line %d)\n", __FILE__, __LINE__); \
		return -2; \
	}

#define OPENCL_DECLARE_EXPORT
#include "opencl_dyn_declare.h" // declare list of functions 

class DynamicLoadOpenCL
{
private:
	DynamicLoadOpenCL();
public:
	static int InitOpenCL();
	static int DestroyOpenCL();
};

#define __InitOpenCL DynamicLoadOpenCL::InitOpenCL
#define __DestroyOpenCL DynamicLoadOpenCL::DestroyOpenCL
