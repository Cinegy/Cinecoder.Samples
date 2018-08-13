#pragma once

#include "utils/HMTSTDUtil.h"
using namespace cinegy::threading_std;

class C_CinecoderErrorHandler : public ICC_ErrorHandler
{
	C_CritSec m_CritSec;

	STDMETHOD(QueryInterface)(REFIID riid, void**p)
	{
		if(p == 0)
		return E_POINTER;

		if(riid != IID_ICC_ErrorHandler && riid != IID_IUnknown)
		return E_NOINTERFACE;

		*p = this;
		return S_OK;
	}
	STDMETHOD_(ULONG,AddRef)()
	{
		return 2;
	}
	STDMETHOD_(ULONG,Release)()
	{
		return 1;
	}
	STDMETHOD(ErrorHandlerFunc)(HRESULT ErrCode, LPCSTR ErrDescription, LPCSTR pFileName, INT LineNo)
	{
		C_AutoLock lock(&m_CritSec);
		
		if (ErrCode == HRESULT_FROM_WIN32(ERROR_MOD_NOT_FOUND)) // not error
			return 0;

		printf("Error %08xh (%s) at %s(%d): %s\n", ErrCode, Cinecoder_GetErrorString(ErrCode), pFileName, LineNo, ErrDescription);

		return 0;
	}
}
g_ErrorHandler;
