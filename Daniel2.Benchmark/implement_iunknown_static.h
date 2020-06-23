#pragma once

#define _IMPLEMENT_IUNKNOWN_ADDREF_RELEASE_STATIC()\
STDMETHOD_(ULONG,AddRef)()                       \
{                                                \
  return 2;                                      \
}                                                \
STDMETHOD_(ULONG,Release)()                      \
{                                                \
  return 1;                                      \
}


//----------------------------------------------------
#define _IMPLEMENT_IUNKNOWN_STATICALLY(I)        \
                                                 \
_IMPLEMENT_IUNKNOWN_ADDREF_RELEASE_STATIC()      \
                                                 \
STDMETHOD(QueryInterface)(REFIID riid, void**p)  \
{                                                \
  if(p == 0)                                     \
    return E_POINTER;                            \
                                                 \
  *p = NULL;                                     \
                                                 \
  if(riid != IID_##I && riid != IID_IUnknown)    \
    return E_NOINTERFACE;                        \
                                                 \
  *p = this;                                     \
  return S_OK;                                   \
}

//----------------------------------------------------
#define _IMPLEMENT_IUNKNOWN_2_STATICALLY(I1,I2)  \
                                                 \
_IMPLEMENT_IUNKNOWN_ADDREF_RELEASE_STATIC()      \
                                                 \
STDMETHOD(QueryInterface)(REFIID riid, void**p)  \
{                                                \
  if(p == 0)                                     \
    return E_POINTER;                            \
                                                 \
  *p = static_cast<I1*>(this);                   \
  if(riid == IID_##I1 || riid == IID_IUnknown) return S_OK;\
                                                 \
  *p = static_cast<I2*>(this);                   \
  if(riid == IID_##I2) return S_OK;              \
                                                 \
  *p = NULL;                                     \
  return E_NOINTERFACE;                          \
}

//----------------------------------------------------
#define _IMPLEMENT_IUNKNOWN_3_STATICALLY(I1,I2,I3)\
                                                 \
_IMPLEMENT_IUNKNOWN_ADDREF_RELEASE_STATIC()      \
                                                 \
STDMETHOD(QueryInterface)(REFIID riid, void**p)  \
{                                                \
  if(p == 0)                                     \
    return E_POINTER;                            \
                                                 \
  *p = static_cast<I1*>(this);                   \
  if(riid == IID_##I1 || riid == IID_IUnknown) return S_OK;\
                                                 \
  *p = static_cast<I2*>(this);                   \
  if(riid == IID_##I2) return S_OK;              \
                                                 \
  *p = static_cast<I3*>(this);                   \
  if(riid == IID_##I3) return S_OK;              \
                                                 \
  *p = NULL;                                     \
  return E_NOINTERFACE;                          \
}
