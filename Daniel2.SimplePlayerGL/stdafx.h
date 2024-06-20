// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include <algorithm>    // std::max / std::min
#include <string>
#include <cassert>
#include <list>
#include <vector>
#include <queue>
#include <memory>
#include <fstream>
#include <atomic>
#include <thread>
#include <chrono>
#include <algorithm>

///////////////////////////////////////////////////////////////////////////////

// Cinegy utils
#include "utils/HMTSTDUtil.h"
using namespace cinegy::threading_std;

#include "utils/simpl_queue.h"
using namespace cinegy::simpl;
#define USE_SIMPL_QUEUE

#include "../common/com_ptr.h"
#include "../common/timer.h"

///////////////////////////////////////////////////////////////////////////////

//#define D2PL_TARGET_OS(X) D2PL_##X()
//#define D2PL_MAC() 1
//#define D2PL_WINDOWS() 0
//#define D2PL_LINUX() 0

//#if defined(_WIN32)
//#define D2PL_WINDOWS() 1
//#else 
//#define D2PL_WINDOWS() 0
//#endif
//
//#if defined(__linux__) || defined(__LINUX__)
//#define D2PL_LINUX() 1
//#else
//#define D2PL_LINUX() 0
//#endif
//
//#if defined(__APPLE__)
//#define D2PL_MAC() 1
//#else
//#define D2PL_MAC() 0
//#endif

#if defined(_WIN32)
	#if !defined(__WIN32__)
		#define __WIN32__
	#endif
#endif

#if defined(__linux__) || defined(__LINUX__)
	#if !defined(__LINUX__)
		#define __LINUX__
	#endif
#endif

#if defined(__WIN32__) // for ConvertStringToBSTR
	#include <comutil.h>
	#pragma comment(lib, "comsuppw.lib")
#endif

#if defined(__WIN32__) || defined(__LINUX__) // CUDA
	#define USE_CUDA_SDK
	#define CUDA_WRAPPER

	#define USE_OPENCL_SDK
	#define OPENCL_WRAPPER
#endif

#ifdef USE_OPENCL_SDK

#ifndef OPENCL_WRAPPER // for build this code on Windows for example need add "opencl-nug" nuget package (nuget.org)
//#define CL_USE_DEPRECATED_OPENCL_1_1_APIS // for clGetExtensionFunctionAddress
#include <CL/cl.h>
#include <CL/cl_gl.h>

#if defined(__WIN32__)
#pragma comment(lib, "OpenCL.lib")
#else defined(__LINUX__)
#pragma comment(lib, "libOpenCL.so")
#endif // #ifndef OPENCL_WRAPPER

#else

#include "../external/opencl_dyn_load/src/opencl_dyn_load.h"

#endif // #ifndef OPENCL_WRAPPER 

#endif // #ifdef USE_OPENCL_SDK

// for build this code on Windows for example need add "glew.v140"/glew.c and "freeglut.3.0.0.v140" nuget packages (nuget.org)
#if !defined(__WIN32__)
	#define __USE_GLUT_RENDER__ 
#endif

#ifndef GLEW_STATIC
	#define GLEW_STATIC // use GLEW as building glew.c
#endif

#if defined(__WIN32__)
#include <GL/glew.h> // GLEW framework
#endif

#if defined(__WIN32__) // use DirectX 11
#include <d3d11.h>
#include <dxgi.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#endif

#ifdef USE_CUDA_SDK

#ifndef CUDA_WRAPPER // for build this code on Windows for example need add "CUDASDK"  nuget packages (Cinegy nuget packages)
#include <cuda.h>
//#if (CUDA_VERSION >= 10010)
//#define CUDA_ENABLE_DEPRECATED // fix error C4996: 'cudaMemcpyArrayToArray': was declared deprecated
//#endif
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda_d3d11_interop.h>

#if defined(__WIN32__)
	#pragma comment(lib, "cudart_static.lib")
#else defined(__LINUX__)
	#pragma comment(lib, "libcudart_static.a")
#endif

#else

#include "../common/cuda_dyn/cuda_dyn_load.h"

#endif // #ifndef CUDA_WRAPPER

#if defined(__WIN32__) || defined(__LINUX__) // use CUDA convert library
#include "CUDAConvertLib.h"
#ifndef __CUDAConvertLib__  
#define __CUDAConvertLib__
#endif
#endif

#define __vrcu \
{ \
	cudaError cudaLastError = cudaGetLastError(); \
	if (cudaLastError != cudaSuccess) \
	{ \
		printf("CUDA error %d %s (%s %d)\n", \
		cudaLastError, cudaGetErrorString(cudaLastError), __FILE__,__LINE__); \
	} \
}

#endif // #ifdef USE_CUDA_SDK

#define __check_hr \
{ \
	if (hr != S_OK && hr != S_FALSE) \
	{ \
		printf("HRESULT error %d (%s %d)\n", \
		hr, __FILE__,__LINE__); \
	} \
}

#define _assert(exp)

#if defined(__WIN32__)
#if defined(__USE_GLUT_RENDER__) // Was added for fix #error:  gl.h included before glew.h
#include <GL/glew.h> // GLEW framework
#include <GL/freeglut.h> // GLUT framework
#endif

#include "GPURenderGL.h"
#include "GPURenderDX.h"
#endif
///////////////////////////////////////////////////////////////////////////////
