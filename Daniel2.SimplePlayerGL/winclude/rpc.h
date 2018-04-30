#pragma once

#include "wtypes.h"
#include "guiddef.h"
#include "winerror.h"

#define STDMETHODCALLTYPE /**/
#define STDAPICALLTYPE    /**/
#define COM_DECLSPEC_NOTHROW /**/

#define __STRUCT__ struct
#define interface __STRUCT__
#define STDMETHOD(method)        virtual COM_DECLSPEC_NOTHROW HRESULT STDMETHODCALLTYPE method
#define STDMETHOD_(type,method)  virtual COM_DECLSPEC_NOTHROW type STDMETHODCALLTYPE method
#define STDMETHODV(method)       virtual COM_DECLSPEC_NOTHROW HRESULT STDMETHODVCALLTYPE method
#define STDMETHODV_(type,method) virtual COM_DECLSPEC_NOTHROW type STDMETHODVCALLTYPE method
#define PURE                    = 0
#define THIS_
#define THIS                    void
#define DECLARE_INTERFACE(iface)    interface DECLSPEC_NOVTABLE iface
#define DECLARE_INTERFACE_(iface, baseiface)    interface DECLSPEC_NOVTABLE iface : public baseiface

#define IFACEMETHOD(method)         __override STDMETHOD(method)
#define IFACEMETHOD_(type,method)   __override STDMETHOD_(type,method)
#define IFACEMETHODV(method)        __override STDMETHODV(method)
#define IFACEMETHODV_(type,method)  __override STDMETHODV_(type,method)

// basetypes.h
#define STDAPI EXTERN_C HRESULT STDAPICALLTYPE
#define STDAPI_(type)     EXTERN_C type STDAPICALLTYPE

interface IUnknown
{
    virtual HRESULT STDMETHODCALLTYPE QueryInterface(
                /* [in] */ REFIID riid,
                /* [iid_is][out] */ void **ppvObject) = 0;

    virtual ULONG STDMETHODCALLTYPE AddRef( void) = 0;

    virtual ULONG STDMETHODCALLTYPE Release( void) = 0;

};

typedef IUnknown* LPUNKNOWN;

typedef void *RPC_IF_HANDLE;
typedef int   RPC_STATUS;

#define RPC_S_OK              0
#define RPC_S_UUID_LOCAL_ONLY 1

#define MIDL_INTERFACE(iid) interface
#define DECLSPEC_UUID(iid) /**/

static UUID IID_IUnknown = { 00000000, 0000, 0000, { 0xc0,00, 00,00,00,00,00,0x46 } };
static UUID IID_IMarshal = { 00000003, 0000, 0000, { 0xc0,00, 00,00,00,00,00,0x46 } };
#define __RPC_USER /**/
#define __RPC_STUB /**/

RPC_STATUS UuidCreateSequential(UUID *piid);

HRESULT CoCreateFreeThreadedMarshaler(
  _In_   LPUNKNOWN punkOuter,
  _Out_  LPUNKNOWN *ppunkMarshal
);

