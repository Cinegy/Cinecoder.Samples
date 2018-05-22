class C_CinecoderErrorHandler : public ICC_ErrorHandler
{
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
     fprintf(stderr, "Error %08xh (%s) at %s(%d): %s\n", ErrCode, Cinecoder_GetErrorString(ErrCode), pFileName, LineNo, ErrDescription);
     return 0;
   }
}
g_ErrorHandler;
