// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

///////////////////////////////////////////////////////////////////////////////

// Cinegy utils
#include "utils/HMTSTDUtil.h"
using namespace cinegy::threading_std;

///////////////////////////////////////////////////////////////////////////////

#if defined(__WIN32__) || defined(_WIN32) // for ConvertStringToBSTR
#include <comutil.h>
#pragma comment(lib, "comsuppw.lib")
#endif

#if defined(__WIN32__) || defined(_WIN32) // for Direct Sound
#include <dsound.h>
#pragma comment(lib, "dsound.lib")
#endif

///////////////////////////////////////////////////////////////////////////////
