#pragma once

#include <atomic>

//////////////////////////////////////////////////////////////////////
// C_Unknown -- our small class to handle COM-style objects
// Requires at least C++0x11
//////////////////////////////////////////////////////////////////////

class C_Unknown : public IUnknown
{
private:
  std::atomic<ULONG> m_cRef;

public:
  C_Unknown(void) : m_cRef(0)
  {}

  virtual ~C_Unknown(void) 
  {}

public:

  STDMETHOD_(ULONG, AddRef)(void)
  {
    return ++m_cRef;
  }
  STDMETHOD_(ULONG, Release)(void)
  {
    int count = --m_cRef;
    if (count == 0) delete this;
    return count;
  }
  STDMETHOD(QueryInterface)(REFIID riid, void **ppv)
  {
    if(ppv == NULL) return E_POINTER;
    *ppv = NULL;
    if (riid == IID_IUnknown) return GetInterface((IUnknown*)this, ppv);
    return E_NOINTERFACE;
  }

protected:
  static HRESULT GetInterface(IUnknown *pUnk, void **ppv)
  {
    if(ppv == NULL) return E_POINTER;
    *ppv = pUnk; 
    pUnk->AddRef(); 
    return S_OK;
  }
};

#define _IMPLEMENT_IUNKNOWN()                                        \
  STDMETHOD_(ULONG, AddRef )(void) { return C_Unknown::AddRef();  }  \
  STDMETHOD_(ULONG, Release)(void) { return C_Unknown::Release(); }

#define _IMPLEMENT_INTERFACE(I) if(riid==IID_##I) return C_Unknown::GetInterface(const_cast<I*>(static_cast<const I*>(this)),ppv)

#define _IMPLEMENT_IUNKNOWN_1(I1)                                 \
  _IMPLEMENT_IUNKNOWN();                                          \
  STDMETHOD(QueryInterface)(REFIID riid, void **ppv)              \
  {                                                               \
    _IMPLEMENT_INTERFACE(I1);                                     \
    return C_Unknown::QueryInterface(riid, ppv);                  \
  }

#define _IMPLEMENT_IUNKNOWN_2(I1,I2)                              \
  _IMPLEMENT_IUNKNOWN();                                          \
  STDMETHOD(QueryInterface)(REFIID riid, void **ppv)              \
  {                                                               \
    _IMPLEMENT_INTERFACE(I1);                                     \
    _IMPLEMENT_INTERFACE(I2);                                     \
    return C_Unknown::QueryInterface(riid, ppv);                  \
  }

#define _IMPLEMENT_IUNKNOWN_3(I1,I2,I3)                           \
  _IMPLEMENT_IUNKNOWN();                                          \
  STDMETHOD(QueryInterface)(REFIID riid, void **ppv)              \
  {                                                               \
    _IMPLEMENT_INTERFACE(I1);                                     \
    _IMPLEMENT_INTERFACE(I2);                                     \
    _IMPLEMENT_INTERFACE(I3);                                     \
    return C_Unknown::QueryInterface(riid, ppv);                  \
  }

#define _IMPLEMENT_IUNKNOWN_4(I1,I2,I3,I4)                        \
  _IMPLEMENT_IUNKNOWN();                                          \
  STDMETHOD(QueryInterface)(REFIID riid, void **ppv)              \
  {                                                               \
    _IMPLEMENT_INTERFACE(I1);                                     \
    _IMPLEMENT_INTERFACE(I2);                                     \
    _IMPLEMENT_INTERFACE(I3);                                     \
    _IMPLEMENT_INTERFACE(I4);                                     \
    return C_Unknown::QueryInterface(riid, ppv);                  \
  }
