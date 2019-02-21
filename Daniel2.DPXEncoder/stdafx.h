// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include <cstdio>
#include <tchar.h>
#include <conio.h>

#include <Cinecoder_h.h>
#include <Cinecoder.Plugin.Multiplexers.h>
#include <iostream>
#include <limits.h>

#include "../common/com_ptr.h"

int print_error(int err, const char *str = nullptr);
