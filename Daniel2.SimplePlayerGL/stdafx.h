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

///////////////////////////////////////////////////////////////////////////////

// Cinegy utils
#include "utils/HMTSTDUtil.h"
using namespace cinegy::threading_std;

///////////////////////////////////////////////////////////////////////////////

#if defined(__WIN32__) || defined(_WIN32) // for ConvertStringToBSTR
#include <comutil.h>
#pragma comment(lib, "comsuppw.lib")
#endif

#if defined(__WIN32__) || defined(_WIN32) // CUDA
#define CUDA_WRAPPER
#ifndef CUDA_WRAPPER
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#pragma comment(lib, "cudart_static.lib")
#endif

#include "cudaDefines.h"

#define __vrcu \
{ \
	cudaError cudaLastError = cudaGetLastError(); \
	if (cudaLastError != cudaSuccess) \
	{ \
		printf("CUDA error %d %s (%s %d)\n", \
		cudaLastError, cudaGetErrorString(cudaLastError), __FILE__,__LINE__); \
	} \
}
#endif

///////////////////////////////////////////////////////////////////////////////
