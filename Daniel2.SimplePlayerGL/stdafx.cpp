// stdafx.cpp : source file that includes just the standard includes
// SimpleDecodeDN2.pch will be the pre-compiled header
// stdafx.obj will contain the pre-compiled type information

#include "stdafx.h"

#ifdef CUDA_WRAPPER // CUDA
#include "../common/cuda_dyn_declare.h"
#endif

#ifdef OPENCL_WRAPPER // OpenCL
#include "../common/opencl_dyn_declare.h"
#endif