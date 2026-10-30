

/* this ALWAYS GENERATED file contains the definitions for the interfaces */


 /* File created by MIDL compiler version 8.01.0628 */
/* at Tue Jan 19 06:14:07 2038
 */
/* Compiler settings for Cinecoder.Plugin.JpegXS_SVT.idl:
    Oicf, W1, Zp8, env=Win64 (32b run), target_arch=AMD64 8.01.0628 
    protocol : all , ms_ext, c_ext, robust
    error checks: allocation ref bounds_check enum stub_data 
    VC __declspec() decoration level: 
         __declspec(uuid()), __declspec(selectany), __declspec(novtable)
         DECLSPEC_UUID(), MIDL_INTERFACE()
*/
/* @@MIDL_FILE_HEADING(  ) */



/* verify that the <rpcndr.h> version is high enough to compile this file*/
#ifndef __REQUIRED_RPCNDR_H_VERSION__
#define __REQUIRED_RPCNDR_H_VERSION__ 500
#endif

#include "rpc.h"
#include "rpcndr.h"

#ifndef __RPCNDR_H_VERSION__
#error this stub requires an updated version of <rpcndr.h>
#endif /* __RPCNDR_H_VERSION__ */

#ifndef COM_NO_WINDOWS_H
#include "windows.h"
#include "ole2.h"
#endif /*COM_NO_WINDOWS_H*/

#ifndef __Cinecoder2EPlugin2EJpegXS_SVT_h_h__
#define __Cinecoder2EPlugin2EJpegXS_SVT_h_h__

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
#pragma once
#endif

#ifndef DECLSPEC_XFGVIRT
#if defined(_CONTROL_FLOW_GUARD_XFG)
#define DECLSPEC_XFGVIRT(base, func) __declspec(xfg_virtual(base, func))
#else
#define DECLSPEC_XFGVIRT(base, func)
#endif
#endif

/* Forward Declarations */ 

#ifndef __ICC_JpegXS_VideoDecoder_SVT_FWD_DEFINED__
#define __ICC_JpegXS_VideoDecoder_SVT_FWD_DEFINED__
typedef interface ICC_JpegXS_VideoDecoder_SVT ICC_JpegXS_VideoDecoder_SVT;

#endif 	/* __ICC_JpegXS_VideoDecoder_SVT_FWD_DEFINED__ */


#ifndef __ICC_JpegXS_VideoEncoder_SVT_FWD_DEFINED__
#define __ICC_JpegXS_VideoEncoder_SVT_FWD_DEFINED__
typedef interface ICC_JpegXS_VideoEncoder_SVT ICC_JpegXS_VideoEncoder_SVT;

#endif 	/* __ICC_JpegXS_VideoEncoder_SVT_FWD_DEFINED__ */


#ifndef __ICC_JpegXS_VideoEncoderSettings_SVT_FWD_DEFINED__
#define __ICC_JpegXS_VideoEncoderSettings_SVT_FWD_DEFINED__
typedef interface ICC_JpegXS_VideoEncoderSettings_SVT ICC_JpegXS_VideoEncoderSettings_SVT;

#endif 	/* __ICC_JpegXS_VideoEncoderSettings_SVT_FWD_DEFINED__ */


#ifndef __ICC_JpegXS_VideoStreamInfo_SVT_FWD_DEFINED__
#define __ICC_JpegXS_VideoStreamInfo_SVT_FWD_DEFINED__
typedef interface ICC_JpegXS_VideoStreamInfo_SVT ICC_JpegXS_VideoStreamInfo_SVT;

#endif 	/* __ICC_JpegXS_VideoStreamInfo_SVT_FWD_DEFINED__ */


#ifndef __ICC_JpegXS_VideoFrameInfo_SVT_FWD_DEFINED__
#define __ICC_JpegXS_VideoFrameInfo_SVT_FWD_DEFINED__
typedef interface ICC_JpegXS_VideoFrameInfo_SVT ICC_JpegXS_VideoFrameInfo_SVT;

#endif 	/* __ICC_JpegXS_VideoFrameInfo_SVT_FWD_DEFINED__ */


#ifndef __CC_JpegXS_VideoDecoder_SVT_FWD_DEFINED__
#define __CC_JpegXS_VideoDecoder_SVT_FWD_DEFINED__

#ifdef __cplusplus
typedef class CC_JpegXS_VideoDecoder_SVT CC_JpegXS_VideoDecoder_SVT;
#else
typedef struct CC_JpegXS_VideoDecoder_SVT CC_JpegXS_VideoDecoder_SVT;
#endif /* __cplusplus */

#endif 	/* __CC_JpegXS_VideoDecoder_SVT_FWD_DEFINED__ */


#ifndef __CC_JpegXS_VideoEncoder_SVT_FWD_DEFINED__
#define __CC_JpegXS_VideoEncoder_SVT_FWD_DEFINED__

#ifdef __cplusplus
typedef class CC_JpegXS_VideoEncoder_SVT CC_JpegXS_VideoEncoder_SVT;
#else
typedef struct CC_JpegXS_VideoEncoder_SVT CC_JpegXS_VideoEncoder_SVT;
#endif /* __cplusplus */

#endif 	/* __CC_JpegXS_VideoEncoder_SVT_FWD_DEFINED__ */


#ifndef __CC_JpegXS_VideoEncoderSettings_SVT_FWD_DEFINED__
#define __CC_JpegXS_VideoEncoderSettings_SVT_FWD_DEFINED__

#ifdef __cplusplus
typedef class CC_JpegXS_VideoEncoderSettings_SVT CC_JpegXS_VideoEncoderSettings_SVT;
#else
typedef struct CC_JpegXS_VideoEncoderSettings_SVT CC_JpegXS_VideoEncoderSettings_SVT;
#endif /* __cplusplus */

#endif 	/* __CC_JpegXS_VideoEncoderSettings_SVT_FWD_DEFINED__ */


/* header files for imported files */
#include "oaidl.h"
#include "ocidl.h"

#ifdef __cplusplus
extern "C"{
#endif 


#ifndef __ICC_JpegXS_VideoDecoder_SVT_INTERFACE_DEFINED__
#define __ICC_JpegXS_VideoDecoder_SVT_INTERFACE_DEFINED__

/* interface ICC_JpegXS_VideoDecoder_SVT */
/* [local][unique][uuid][object] */ 


EXTERN_C const IID IID_ICC_JpegXS_VideoDecoder_SVT;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("bd87be1d-f8ef-49b1-b092-b297c379b8ef")
    ICC_JpegXS_VideoDecoder_SVT : public ICC_VideoDecoder
    {
    public:
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_JpegXS_VideoDecoder_SVTVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_JpegXS_VideoDecoder_SVT * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_JpegXS_VideoDecoder_SVT * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_JpegXS_VideoDecoder_SVT * This);
        
        DECLSPEC_XFGVIRT(ICC_StreamProcessor, Init)
        HRESULT ( STDMETHODCALLTYPE *Init )( 
            ICC_JpegXS_VideoDecoder_SVT * This,
            /* [defaultvalue][in] */ ICC_Settings *pSettings);
        
        DECLSPEC_XFGVIRT(ICC_StreamProcessor, InitByXml)
        HRESULT ( STDMETHODCALLTYPE *InitByXml )( 
            ICC_JpegXS_VideoDecoder_SVT * This,
            /* [in] */ CC_STRING strXML);
        
        DECLSPEC_XFGVIRT(ICC_StreamProcessor, Done)
        HRESULT ( STDMETHODCALLTYPE *Done )( 
            ICC_JpegXS_VideoDecoder_SVT * This,
            /* [in] */ CC_BOOL bFlush,
            /* [defaultvalue][retval][out] */ CC_BOOL *pbDone);
        
        DECLSPEC_XFGVIRT(ICC_StreamProcessor, get_IsActive)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_IsActive )( 
            ICC_JpegXS_VideoDecoder_SVT * This,
            /* [defaultvalue][retval][out] */ CC_BOOL *__MIDL__ICC_StreamProcessor0000);
        
        DECLSPEC_XFGVIRT(ICC_StreamProcessor, get_TimeBase)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_TimeBase )( 
            ICC_JpegXS_VideoDecoder_SVT * This,
            /* [retval][out] */ CC_TIMEBASE *p);
        
        DECLSPEC_XFGVIRT(ICC_StreamProcessor, put_TimeBase)
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_TimeBase )( 
            ICC_JpegXS_VideoDecoder_SVT * This,
            /* [in] */ CC_TIMEBASE p);
        
        DECLSPEC_XFGVIRT(ICC_StreamProcessor, get_BitRate)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_BitRate )( 
            ICC_JpegXS_VideoDecoder_SVT * This,
            /* [retval][out] */ CC_BITRATE *p);
        
        DECLSPEC_XFGVIRT(ICC_StreamProcessor, get_StreamInfo)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_StreamInfo )( 
            ICC_JpegXS_VideoDecoder_SVT * This,
            /* [retval][out] */ ICC_Settings **p);
        
        DECLSPEC_XFGVIRT(ICC_StreamProcessor, get_IsDataReady)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_IsDataReady )( 
            ICC_JpegXS_VideoDecoder_SVT * This,
            /* [defaultvalue][retval][out] */ CC_BOOL *p);
        
        DECLSPEC_XFGVIRT(ICC_StreamProcessor, get_DataInfo)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_DataInfo )( 
            ICC_JpegXS_VideoDecoder_SVT * This,
            /* [retval][out] */ ICC_Settings **s);
        
        DECLSPEC_XFGVIRT(ICC_StreamProcessor, get_OutputCallback)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_OutputCallback )( 
            ICC_JpegXS_VideoDecoder_SVT * This,
            /* [retval][out] */ IUnknown **p);
        
        DECLSPEC_XFGVIRT(ICC_StreamProcessor, put_OutputCallback)
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_OutputCallback )( 
            ICC_JpegXS_VideoDecoder_SVT * This,
            /* [in] */ IUnknown *p);
        
        DECLSPEC_XFGVIRT(ICC_Decoder, ProcessData)
        HRESULT ( STDMETHODCALLTYPE *ProcessData )( 
            ICC_JpegXS_VideoDecoder_SVT * This,
            /* [size_is][in] */ CC_PCBYTE pbData,
            /* [in] */ CC_UINT cbSize,
            /* [defaultvalue][in] */ CC_UINT cbOffset,
            /* [defaultvalue][in] */ CC_TIME pts,
            /* [defaultvalue][retval][out] */ CC_UINT *pcbProcessed);
        
        DECLSPEC_XFGVIRT(ICC_Decoder, Break)
        HRESULT ( STDMETHODCALLTYPE *Break )( 
            ICC_JpegXS_VideoDecoder_SVT * This,
            /* [in] */ CC_BOOL bFlush,
            /* [defaultvalue][retval][out] */ CC_BOOL *pbDone);
        
        DECLSPEC_XFGVIRT(ICC_VideoDecoder, GetFrame)
        HRESULT ( STDMETHODCALLTYPE *GetFrame )( 
            ICC_JpegXS_VideoDecoder_SVT * This,
            /* [in] */ CC_COLOR_FMT Format,
            /* [size_is][out] */ BYTE *pbVideoData,
            /* [in] */ DWORD cbSize,
            /* [defaultvalue][in] */ INT stride,
            /* [defaultvalue][retval][out] */ DWORD *pcbRetSize);
        
        DECLSPEC_XFGVIRT(ICC_VideoDecoder, GetStride)
        HRESULT ( STDMETHODCALLTYPE *GetStride )( 
            ICC_JpegXS_VideoDecoder_SVT * This,
            /* [in] */ CC_COLOR_FMT fmt,
            /* [retval][out] */ DWORD *pNumBytes);
        
        DECLSPEC_XFGVIRT(ICC_VideoDecoder, IsFormatSupported)
        HRESULT ( STDMETHODCALLTYPE *IsFormatSupported )( 
            ICC_JpegXS_VideoDecoder_SVT * This,
            /* [in] */ CC_COLOR_FMT fmt,
            /* [defaultvalue][retval][out] */ CC_BOOL *pResult);
        
        DECLSPEC_XFGVIRT(ICC_VideoDecoder, GetVideoStreamInfo)
        HRESULT ( STDMETHODCALLTYPE *GetVideoStreamInfo )( 
            ICC_JpegXS_VideoDecoder_SVT * This,
            /* [retval][out] */ ICC_VideoStreamInfo **pDescr);
        
        DECLSPEC_XFGVIRT(ICC_VideoDecoder, GetVideoFrameInfo)
        HRESULT ( STDMETHODCALLTYPE *GetVideoFrameInfo )( 
            ICC_JpegXS_VideoDecoder_SVT * This,
            /* [retval][out] */ ICC_VideoFrameInfo **pDescr);
        
        END_INTERFACE
    } ICC_JpegXS_VideoDecoder_SVTVtbl;

    interface ICC_JpegXS_VideoDecoder_SVT
    {
        CONST_VTBL struct ICC_JpegXS_VideoDecoder_SVTVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_JpegXS_VideoDecoder_SVT_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_JpegXS_VideoDecoder_SVT_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_JpegXS_VideoDecoder_SVT_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ICC_JpegXS_VideoDecoder_SVT_Init(This,pSettings)	\
    ( (This)->lpVtbl -> Init(This,pSettings) ) 

#define ICC_JpegXS_VideoDecoder_SVT_InitByXml(This,strXML)	\
    ( (This)->lpVtbl -> InitByXml(This,strXML) ) 

#define ICC_JpegXS_VideoDecoder_SVT_Done(This,bFlush,pbDone)	\
    ( (This)->lpVtbl -> Done(This,bFlush,pbDone) ) 

#define ICC_JpegXS_VideoDecoder_SVT_get_IsActive(This,__MIDL__ICC_StreamProcessor0000)	\
    ( (This)->lpVtbl -> get_IsActive(This,__MIDL__ICC_StreamProcessor0000) ) 

#define ICC_JpegXS_VideoDecoder_SVT_get_TimeBase(This,p)	\
    ( (This)->lpVtbl -> get_TimeBase(This,p) ) 

#define ICC_JpegXS_VideoDecoder_SVT_put_TimeBase(This,p)	\
    ( (This)->lpVtbl -> put_TimeBase(This,p) ) 

#define ICC_JpegXS_VideoDecoder_SVT_get_BitRate(This,p)	\
    ( (This)->lpVtbl -> get_BitRate(This,p) ) 

#define ICC_JpegXS_VideoDecoder_SVT_get_StreamInfo(This,p)	\
    ( (This)->lpVtbl -> get_StreamInfo(This,p) ) 

#define ICC_JpegXS_VideoDecoder_SVT_get_IsDataReady(This,p)	\
    ( (This)->lpVtbl -> get_IsDataReady(This,p) ) 

#define ICC_JpegXS_VideoDecoder_SVT_get_DataInfo(This,s)	\
    ( (This)->lpVtbl -> get_DataInfo(This,s) ) 

#define ICC_JpegXS_VideoDecoder_SVT_get_OutputCallback(This,p)	\
    ( (This)->lpVtbl -> get_OutputCallback(This,p) ) 

#define ICC_JpegXS_VideoDecoder_SVT_put_OutputCallback(This,p)	\
    ( (This)->lpVtbl -> put_OutputCallback(This,p) ) 


#define ICC_JpegXS_VideoDecoder_SVT_ProcessData(This,pbData,cbSize,cbOffset,pts,pcbProcessed)	\
    ( (This)->lpVtbl -> ProcessData(This,pbData,cbSize,cbOffset,pts,pcbProcessed) ) 

#define ICC_JpegXS_VideoDecoder_SVT_Break(This,bFlush,pbDone)	\
    ( (This)->lpVtbl -> Break(This,bFlush,pbDone) ) 


#define ICC_JpegXS_VideoDecoder_SVT_GetFrame(This,Format,pbVideoData,cbSize,stride,pcbRetSize)	\
    ( (This)->lpVtbl -> GetFrame(This,Format,pbVideoData,cbSize,stride,pcbRetSize) ) 

#define ICC_JpegXS_VideoDecoder_SVT_GetStride(This,fmt,pNumBytes)	\
    ( (This)->lpVtbl -> GetStride(This,fmt,pNumBytes) ) 

#define ICC_JpegXS_VideoDecoder_SVT_IsFormatSupported(This,fmt,pResult)	\
    ( (This)->lpVtbl -> IsFormatSupported(This,fmt,pResult) ) 

#define ICC_JpegXS_VideoDecoder_SVT_GetVideoStreamInfo(This,pDescr)	\
    ( (This)->lpVtbl -> GetVideoStreamInfo(This,pDescr) ) 

#define ICC_JpegXS_VideoDecoder_SVT_GetVideoFrameInfo(This,pDescr)	\
    ( (This)->lpVtbl -> GetVideoFrameInfo(This,pDescr) ) 


#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_JpegXS_VideoDecoder_SVT_INTERFACE_DEFINED__ */


#ifndef __ICC_JpegXS_VideoEncoder_SVT_INTERFACE_DEFINED__
#define __ICC_JpegXS_VideoEncoder_SVT_INTERFACE_DEFINED__

/* interface ICC_JpegXS_VideoEncoder_SVT */
/* [local][unique][uuid][object] */ 


EXTERN_C const IID IID_ICC_JpegXS_VideoEncoder_SVT;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("e0f876ef-acc3-4791-94db-f9683710ad51")
    ICC_JpegXS_VideoEncoder_SVT : public ICC_VideoEncoder
    {
    public:
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_JpegXS_VideoEncoder_SVTVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_JpegXS_VideoEncoder_SVT * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_JpegXS_VideoEncoder_SVT * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_JpegXS_VideoEncoder_SVT * This);
        
        DECLSPEC_XFGVIRT(ICC_StreamProcessor, Init)
        HRESULT ( STDMETHODCALLTYPE *Init )( 
            ICC_JpegXS_VideoEncoder_SVT * This,
            /* [defaultvalue][in] */ ICC_Settings *pSettings);
        
        DECLSPEC_XFGVIRT(ICC_StreamProcessor, InitByXml)
        HRESULT ( STDMETHODCALLTYPE *InitByXml )( 
            ICC_JpegXS_VideoEncoder_SVT * This,
            /* [in] */ CC_STRING strXML);
        
        DECLSPEC_XFGVIRT(ICC_StreamProcessor, Done)
        HRESULT ( STDMETHODCALLTYPE *Done )( 
            ICC_JpegXS_VideoEncoder_SVT * This,
            /* [in] */ CC_BOOL bFlush,
            /* [defaultvalue][retval][out] */ CC_BOOL *pbDone);
        
        DECLSPEC_XFGVIRT(ICC_StreamProcessor, get_IsActive)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_IsActive )( 
            ICC_JpegXS_VideoEncoder_SVT * This,
            /* [defaultvalue][retval][out] */ CC_BOOL *__MIDL__ICC_StreamProcessor0000);
        
        DECLSPEC_XFGVIRT(ICC_StreamProcessor, get_TimeBase)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_TimeBase )( 
            ICC_JpegXS_VideoEncoder_SVT * This,
            /* [retval][out] */ CC_TIMEBASE *p);
        
        DECLSPEC_XFGVIRT(ICC_StreamProcessor, put_TimeBase)
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_TimeBase )( 
            ICC_JpegXS_VideoEncoder_SVT * This,
            /* [in] */ CC_TIMEBASE p);
        
        DECLSPEC_XFGVIRT(ICC_StreamProcessor, get_BitRate)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_BitRate )( 
            ICC_JpegXS_VideoEncoder_SVT * This,
            /* [retval][out] */ CC_BITRATE *p);
        
        DECLSPEC_XFGVIRT(ICC_StreamProcessor, get_StreamInfo)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_StreamInfo )( 
            ICC_JpegXS_VideoEncoder_SVT * This,
            /* [retval][out] */ ICC_Settings **p);
        
        DECLSPEC_XFGVIRT(ICC_StreamProcessor, get_IsDataReady)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_IsDataReady )( 
            ICC_JpegXS_VideoEncoder_SVT * This,
            /* [defaultvalue][retval][out] */ CC_BOOL *p);
        
        DECLSPEC_XFGVIRT(ICC_StreamProcessor, get_DataInfo)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_DataInfo )( 
            ICC_JpegXS_VideoEncoder_SVT * This,
            /* [retval][out] */ ICC_Settings **s);
        
        DECLSPEC_XFGVIRT(ICC_StreamProcessor, get_OutputCallback)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_OutputCallback )( 
            ICC_JpegXS_VideoEncoder_SVT * This,
            /* [retval][out] */ IUnknown **p);
        
        DECLSPEC_XFGVIRT(ICC_StreamProcessor, put_OutputCallback)
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_OutputCallback )( 
            ICC_JpegXS_VideoEncoder_SVT * This,
            /* [in] */ IUnknown *p);
        
        DECLSPEC_XFGVIRT(ICC_Encoder, get_DataSize)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_DataSize )( 
            ICC_JpegXS_VideoEncoder_SVT * This,
            /* [retval][out] */ CC_UINT *__MIDL__ICC_Encoder0000);
        
        DECLSPEC_XFGVIRT(ICC_Encoder, GetData)
        HRESULT ( STDMETHODCALLTYPE *GetData )( 
            ICC_JpegXS_VideoEncoder_SVT * This,
            /* [size_is][out] */ CC_PBYTE pbData,
            /* [in] */ CC_UINT cbBufSize,
            /* [defaultvalue][retval][out] */ CC_UINT *pcbRetSize);
        
        DECLSPEC_XFGVIRT(ICC_VideoEncoder, AddFrame)
        HRESULT ( STDMETHODCALLTYPE *AddFrame )( 
            ICC_JpegXS_VideoEncoder_SVT * This,
            /* [in] */ CC_COLOR_FMT Format,
            /* [size_is][in] */ const BYTE *pData,
            /* [in] */ DWORD cbSize,
            /* [defaultvalue][in] */ INT stride,
            /* [defaultvalue][retval][out] */ CC_BOOL *pResult);
        
        DECLSPEC_XFGVIRT(ICC_VideoEncoder, GetStride)
        HRESULT ( STDMETHODCALLTYPE *GetStride )( 
            ICC_JpegXS_VideoEncoder_SVT * This,
            /* [in] */ CC_COLOR_FMT fmt,
            /* [retval][out] */ DWORD *pNumBytes);
        
        DECLSPEC_XFGVIRT(ICC_VideoEncoder, IsFormatSupported)
        HRESULT ( STDMETHODCALLTYPE *IsFormatSupported )( 
            ICC_JpegXS_VideoEncoder_SVT * This,
            /* [in] */ CC_COLOR_FMT fmt,
            /* [defaultvalue][retval][out] */ CC_BOOL *pResult);
        
        DECLSPEC_XFGVIRT(ICC_VideoEncoder, AddScaleFrame)
        HRESULT ( STDMETHODCALLTYPE *AddScaleFrame )( 
            ICC_JpegXS_VideoEncoder_SVT * This,
            /* [size_is][in] */ const BYTE *pData,
            /* [in] */ DWORD cbSize,
            /* [in] */ CC_ADD_VIDEO_FRAME_PARAMS *pParams,
            /* [defaultvalue][retval][out] */ CC_BOOL *pResult);
        
        DECLSPEC_XFGVIRT(ICC_VideoEncoder, IsScaleAvailable)
        HRESULT ( STDMETHODCALLTYPE *IsScaleAvailable )( 
            ICC_JpegXS_VideoEncoder_SVT * This,
            /* [in] */ CC_ADD_VIDEO_FRAME_PARAMS *pParams,
            /* [defaultvalue][retval][out] */ CC_BOOL *__MIDL__ICC_VideoEncoder0000);
        
        DECLSPEC_XFGVIRT(ICC_VideoEncoder, GetVideoStreamInfo)
        HRESULT ( STDMETHODCALLTYPE *GetVideoStreamInfo )( 
            ICC_JpegXS_VideoEncoder_SVT * This,
            /* [retval][out] */ ICC_VideoStreamInfo **pDescr);
        
        DECLSPEC_XFGVIRT(ICC_VideoEncoder, GetVideoFrameInfo)
        HRESULT ( STDMETHODCALLTYPE *GetVideoFrameInfo )( 
            ICC_JpegXS_VideoEncoder_SVT * This,
            /* [retval][out] */ ICC_VideoFrameInfo **pDescr);
        
        END_INTERFACE
    } ICC_JpegXS_VideoEncoder_SVTVtbl;

    interface ICC_JpegXS_VideoEncoder_SVT
    {
        CONST_VTBL struct ICC_JpegXS_VideoEncoder_SVTVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_JpegXS_VideoEncoder_SVT_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_JpegXS_VideoEncoder_SVT_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_JpegXS_VideoEncoder_SVT_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ICC_JpegXS_VideoEncoder_SVT_Init(This,pSettings)	\
    ( (This)->lpVtbl -> Init(This,pSettings) ) 

#define ICC_JpegXS_VideoEncoder_SVT_InitByXml(This,strXML)	\
    ( (This)->lpVtbl -> InitByXml(This,strXML) ) 

#define ICC_JpegXS_VideoEncoder_SVT_Done(This,bFlush,pbDone)	\
    ( (This)->lpVtbl -> Done(This,bFlush,pbDone) ) 

#define ICC_JpegXS_VideoEncoder_SVT_get_IsActive(This,__MIDL__ICC_StreamProcessor0000)	\
    ( (This)->lpVtbl -> get_IsActive(This,__MIDL__ICC_StreamProcessor0000) ) 

#define ICC_JpegXS_VideoEncoder_SVT_get_TimeBase(This,p)	\
    ( (This)->lpVtbl -> get_TimeBase(This,p) ) 

#define ICC_JpegXS_VideoEncoder_SVT_put_TimeBase(This,p)	\
    ( (This)->lpVtbl -> put_TimeBase(This,p) ) 

#define ICC_JpegXS_VideoEncoder_SVT_get_BitRate(This,p)	\
    ( (This)->lpVtbl -> get_BitRate(This,p) ) 

#define ICC_JpegXS_VideoEncoder_SVT_get_StreamInfo(This,p)	\
    ( (This)->lpVtbl -> get_StreamInfo(This,p) ) 

#define ICC_JpegXS_VideoEncoder_SVT_get_IsDataReady(This,p)	\
    ( (This)->lpVtbl -> get_IsDataReady(This,p) ) 

#define ICC_JpegXS_VideoEncoder_SVT_get_DataInfo(This,s)	\
    ( (This)->lpVtbl -> get_DataInfo(This,s) ) 

#define ICC_JpegXS_VideoEncoder_SVT_get_OutputCallback(This,p)	\
    ( (This)->lpVtbl -> get_OutputCallback(This,p) ) 

#define ICC_JpegXS_VideoEncoder_SVT_put_OutputCallback(This,p)	\
    ( (This)->lpVtbl -> put_OutputCallback(This,p) ) 


#define ICC_JpegXS_VideoEncoder_SVT_get_DataSize(This,__MIDL__ICC_Encoder0000)	\
    ( (This)->lpVtbl -> get_DataSize(This,__MIDL__ICC_Encoder0000) ) 

#define ICC_JpegXS_VideoEncoder_SVT_GetData(This,pbData,cbBufSize,pcbRetSize)	\
    ( (This)->lpVtbl -> GetData(This,pbData,cbBufSize,pcbRetSize) ) 


#define ICC_JpegXS_VideoEncoder_SVT_AddFrame(This,Format,pData,cbSize,stride,pResult)	\
    ( (This)->lpVtbl -> AddFrame(This,Format,pData,cbSize,stride,pResult) ) 

#define ICC_JpegXS_VideoEncoder_SVT_GetStride(This,fmt,pNumBytes)	\
    ( (This)->lpVtbl -> GetStride(This,fmt,pNumBytes) ) 

#define ICC_JpegXS_VideoEncoder_SVT_IsFormatSupported(This,fmt,pResult)	\
    ( (This)->lpVtbl -> IsFormatSupported(This,fmt,pResult) ) 

#define ICC_JpegXS_VideoEncoder_SVT_AddScaleFrame(This,pData,cbSize,pParams,pResult)	\
    ( (This)->lpVtbl -> AddScaleFrame(This,pData,cbSize,pParams,pResult) ) 

#define ICC_JpegXS_VideoEncoder_SVT_IsScaleAvailable(This,pParams,__MIDL__ICC_VideoEncoder0000)	\
    ( (This)->lpVtbl -> IsScaleAvailable(This,pParams,__MIDL__ICC_VideoEncoder0000) ) 

#define ICC_JpegXS_VideoEncoder_SVT_GetVideoStreamInfo(This,pDescr)	\
    ( (This)->lpVtbl -> GetVideoStreamInfo(This,pDescr) ) 

#define ICC_JpegXS_VideoEncoder_SVT_GetVideoFrameInfo(This,pDescr)	\
    ( (This)->lpVtbl -> GetVideoFrameInfo(This,pDescr) ) 


#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_JpegXS_VideoEncoder_SVT_INTERFACE_DEFINED__ */


#ifndef __ICC_JpegXS_VideoEncoderSettings_SVT_INTERFACE_DEFINED__
#define __ICC_JpegXS_VideoEncoderSettings_SVT_INTERFACE_DEFINED__

/* interface ICC_JpegXS_VideoEncoderSettings_SVT */
/* [local][unique][uuid][object] */ 


EXTERN_C const IID IID_ICC_JpegXS_VideoEncoderSettings_SVT;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("507da1c5-8a4e-402f-9adc-7f89fe2a64b1")
    ICC_JpegXS_VideoEncoderSettings_SVT : public ICC_Settings
    {
    public:
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_FrameSize( 
            /* [retval][out] */ CC_SIZE *v) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_FrameSize( 
            /* [in] */ CC_SIZE v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_ChromaFormat( 
            /* [retval][out] */ CC_CHROMA_FORMAT *v) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_ChromaFormat( 
            /* [in] */ CC_CHROMA_FORMAT v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_BitDepth( 
            /* [retval][out] */ CC_UINT *v) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_BitDepth( 
            /* [in] */ CC_UINT v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_NumDecomp( 
            /* [retval][out] */ CC_SIZE *v) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_NumDecomp( 
            /* [in] */ CC_SIZE v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_SliceHeight( 
            /* [retval][out] */ CC_UINT *v) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_SliceHeight( 
            /* [in] */ CC_UINT v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_Quantization( 
            /* [retval][out] */ CC_UINT *v) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_Quantization( 
            /* [in] */ CC_UINT v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_CodingSignsHandling( 
            /* [retval][out] */ CC_UINT *v) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_CodingSignsHandling( 
            /* [in] */ CC_UINT v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_CodingSignificance( 
            /* [retval][out] */ CC_UINT *v) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_CodingSignificance( 
            /* [in] */ CC_UINT v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_EncBitsPerPixel( 
            /* [retval][out] */ CC_RATIONAL *v) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_EncBitsPerPixel( 
            /* [in] */ CC_RATIONAL v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_RateControlMode( 
            /* [retval][out] */ CC_UINT *v) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_RateControlMode( 
            /* [in] */ CC_UINT v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_CodingVerticalPredictionMode( 
            /* [retval][out] */ CC_UINT *v) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_CodingVerticalPredictionMode( 
            /* [in] */ CC_UINT v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_SlicePacketizationMode( 
            /* [retval][out] */ CC_UINT *v) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_SlicePacketizationMode( 
            /* [in] */ CC_UINT v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_CodingRawDisable( 
            /* [retval][out] */ CC_BOOL *v) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_CodingRawDisable( 
            /* [in] */ CC_BOOL v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_CapCompat( 
            /* [retval][out] */ CC_UINT *v) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_CapCompat( 
            /* [in] */ CC_UINT v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_ProfilePpihOverride( 
            /* [retval][out] */ CC_UINT *v) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_ProfilePpihOverride( 
            /* [in] */ CC_UINT v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_LevelPlevOverride( 
            /* [retval][out] */ CC_UINT *v) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_LevelPlevOverride( 
            /* [in] */ CC_UINT v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_CpuProfile( 
            /* [retval][out] */ CC_UINT *v) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_CpuProfile( 
            /* [in] */ CC_UINT v) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_JpegXS_VideoEncoderSettings_SVTVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This);
        
        DECLSPEC_XFGVIRT(ICC_Settings, Clear)
        HRESULT ( STDMETHODCALLTYPE *Clear )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [in] */ LPCSTR strVarName);
        
        DECLSPEC_XFGVIRT(ICC_Settings, Assigned)
        HRESULT ( STDMETHODCALLTYPE *Assigned )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [in] */ LPCSTR strVarName,
            /* [defaultvalue][retval][out] */ CC_BOOL *__MIDL__ICC_Settings0000);
        
        DECLSPEC_XFGVIRT(ICC_Settings, get_XML)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_XML )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [retval][out] */ CC_STRING *pstrXml);
        
        DECLSPEC_XFGVIRT(ICC_Settings, put_XML)
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_XML )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [in] */ CC_STRING strXml);
        
        DECLSPEC_XFGVIRT(ICC_JpegXS_VideoEncoderSettings_SVT, get_FrameSize)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_FrameSize )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [retval][out] */ CC_SIZE *v);
        
        DECLSPEC_XFGVIRT(ICC_JpegXS_VideoEncoderSettings_SVT, put_FrameSize)
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_FrameSize )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [in] */ CC_SIZE v);
        
        DECLSPEC_XFGVIRT(ICC_JpegXS_VideoEncoderSettings_SVT, get_ChromaFormat)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_ChromaFormat )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [retval][out] */ CC_CHROMA_FORMAT *v);
        
        DECLSPEC_XFGVIRT(ICC_JpegXS_VideoEncoderSettings_SVT, put_ChromaFormat)
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_ChromaFormat )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [in] */ CC_CHROMA_FORMAT v);
        
        DECLSPEC_XFGVIRT(ICC_JpegXS_VideoEncoderSettings_SVT, get_BitDepth)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_BitDepth )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [retval][out] */ CC_UINT *v);
        
        DECLSPEC_XFGVIRT(ICC_JpegXS_VideoEncoderSettings_SVT, put_BitDepth)
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_BitDepth )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [in] */ CC_UINT v);
        
        DECLSPEC_XFGVIRT(ICC_JpegXS_VideoEncoderSettings_SVT, get_NumDecomp)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_NumDecomp )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [retval][out] */ CC_SIZE *v);
        
        DECLSPEC_XFGVIRT(ICC_JpegXS_VideoEncoderSettings_SVT, put_NumDecomp)
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_NumDecomp )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [in] */ CC_SIZE v);
        
        DECLSPEC_XFGVIRT(ICC_JpegXS_VideoEncoderSettings_SVT, get_SliceHeight)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_SliceHeight )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [retval][out] */ CC_UINT *v);
        
        DECLSPEC_XFGVIRT(ICC_JpegXS_VideoEncoderSettings_SVT, put_SliceHeight)
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_SliceHeight )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [in] */ CC_UINT v);
        
        DECLSPEC_XFGVIRT(ICC_JpegXS_VideoEncoderSettings_SVT, get_Quantization)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Quantization )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [retval][out] */ CC_UINT *v);
        
        DECLSPEC_XFGVIRT(ICC_JpegXS_VideoEncoderSettings_SVT, put_Quantization)
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_Quantization )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [in] */ CC_UINT v);
        
        DECLSPEC_XFGVIRT(ICC_JpegXS_VideoEncoderSettings_SVT, get_CodingSignsHandling)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_CodingSignsHandling )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [retval][out] */ CC_UINT *v);
        
        DECLSPEC_XFGVIRT(ICC_JpegXS_VideoEncoderSettings_SVT, put_CodingSignsHandling)
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_CodingSignsHandling )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [in] */ CC_UINT v);
        
        DECLSPEC_XFGVIRT(ICC_JpegXS_VideoEncoderSettings_SVT, get_CodingSignificance)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_CodingSignificance )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [retval][out] */ CC_UINT *v);
        
        DECLSPEC_XFGVIRT(ICC_JpegXS_VideoEncoderSettings_SVT, put_CodingSignificance)
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_CodingSignificance )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [in] */ CC_UINT v);
        
        DECLSPEC_XFGVIRT(ICC_JpegXS_VideoEncoderSettings_SVT, get_EncBitsPerPixel)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_EncBitsPerPixel )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [retval][out] */ CC_RATIONAL *v);
        
        DECLSPEC_XFGVIRT(ICC_JpegXS_VideoEncoderSettings_SVT, put_EncBitsPerPixel)
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_EncBitsPerPixel )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [in] */ CC_RATIONAL v);
        
        DECLSPEC_XFGVIRT(ICC_JpegXS_VideoEncoderSettings_SVT, get_RateControlMode)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_RateControlMode )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [retval][out] */ CC_UINT *v);
        
        DECLSPEC_XFGVIRT(ICC_JpegXS_VideoEncoderSettings_SVT, put_RateControlMode)
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_RateControlMode )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [in] */ CC_UINT v);
        
        DECLSPEC_XFGVIRT(ICC_JpegXS_VideoEncoderSettings_SVT, get_CodingVerticalPredictionMode)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_CodingVerticalPredictionMode )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [retval][out] */ CC_UINT *v);
        
        DECLSPEC_XFGVIRT(ICC_JpegXS_VideoEncoderSettings_SVT, put_CodingVerticalPredictionMode)
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_CodingVerticalPredictionMode )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [in] */ CC_UINT v);
        
        DECLSPEC_XFGVIRT(ICC_JpegXS_VideoEncoderSettings_SVT, get_SlicePacketizationMode)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_SlicePacketizationMode )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [retval][out] */ CC_UINT *v);
        
        DECLSPEC_XFGVIRT(ICC_JpegXS_VideoEncoderSettings_SVT, put_SlicePacketizationMode)
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_SlicePacketizationMode )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [in] */ CC_UINT v);
        
        DECLSPEC_XFGVIRT(ICC_JpegXS_VideoEncoderSettings_SVT, get_CodingRawDisable)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_CodingRawDisable )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [retval][out] */ CC_BOOL *v);
        
        DECLSPEC_XFGVIRT(ICC_JpegXS_VideoEncoderSettings_SVT, put_CodingRawDisable)
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_CodingRawDisable )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [in] */ CC_BOOL v);
        
        DECLSPEC_XFGVIRT(ICC_JpegXS_VideoEncoderSettings_SVT, get_CapCompat)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_CapCompat )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [retval][out] */ CC_UINT *v);
        
        DECLSPEC_XFGVIRT(ICC_JpegXS_VideoEncoderSettings_SVT, put_CapCompat)
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_CapCompat )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [in] */ CC_UINT v);
        
        DECLSPEC_XFGVIRT(ICC_JpegXS_VideoEncoderSettings_SVT, get_ProfilePpihOverride)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_ProfilePpihOverride )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [retval][out] */ CC_UINT *v);
        
        DECLSPEC_XFGVIRT(ICC_JpegXS_VideoEncoderSettings_SVT, put_ProfilePpihOverride)
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_ProfilePpihOverride )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [in] */ CC_UINT v);
        
        DECLSPEC_XFGVIRT(ICC_JpegXS_VideoEncoderSettings_SVT, get_LevelPlevOverride)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_LevelPlevOverride )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [retval][out] */ CC_UINT *v);
        
        DECLSPEC_XFGVIRT(ICC_JpegXS_VideoEncoderSettings_SVT, put_LevelPlevOverride)
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_LevelPlevOverride )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [in] */ CC_UINT v);
        
        DECLSPEC_XFGVIRT(ICC_JpegXS_VideoEncoderSettings_SVT, get_CpuProfile)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_CpuProfile )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [retval][out] */ CC_UINT *v);
        
        DECLSPEC_XFGVIRT(ICC_JpegXS_VideoEncoderSettings_SVT, put_CpuProfile)
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_CpuProfile )( 
            ICC_JpegXS_VideoEncoderSettings_SVT * This,
            /* [in] */ CC_UINT v);
        
        END_INTERFACE
    } ICC_JpegXS_VideoEncoderSettings_SVTVtbl;

    interface ICC_JpegXS_VideoEncoderSettings_SVT
    {
        CONST_VTBL struct ICC_JpegXS_VideoEncoderSettings_SVTVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_JpegXS_VideoEncoderSettings_SVT_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ICC_JpegXS_VideoEncoderSettings_SVT_Clear(This,strVarName)	\
    ( (This)->lpVtbl -> Clear(This,strVarName) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_Assigned(This,strVarName,__MIDL__ICC_Settings0000)	\
    ( (This)->lpVtbl -> Assigned(This,strVarName,__MIDL__ICC_Settings0000) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_get_XML(This,pstrXml)	\
    ( (This)->lpVtbl -> get_XML(This,pstrXml) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_put_XML(This,strXml)	\
    ( (This)->lpVtbl -> put_XML(This,strXml) ) 


#define ICC_JpegXS_VideoEncoderSettings_SVT_get_FrameSize(This,v)	\
    ( (This)->lpVtbl -> get_FrameSize(This,v) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_put_FrameSize(This,v)	\
    ( (This)->lpVtbl -> put_FrameSize(This,v) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_get_ChromaFormat(This,v)	\
    ( (This)->lpVtbl -> get_ChromaFormat(This,v) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_put_ChromaFormat(This,v)	\
    ( (This)->lpVtbl -> put_ChromaFormat(This,v) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_get_BitDepth(This,v)	\
    ( (This)->lpVtbl -> get_BitDepth(This,v) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_put_BitDepth(This,v)	\
    ( (This)->lpVtbl -> put_BitDepth(This,v) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_get_NumDecomp(This,v)	\
    ( (This)->lpVtbl -> get_NumDecomp(This,v) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_put_NumDecomp(This,v)	\
    ( (This)->lpVtbl -> put_NumDecomp(This,v) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_get_SliceHeight(This,v)	\
    ( (This)->lpVtbl -> get_SliceHeight(This,v) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_put_SliceHeight(This,v)	\
    ( (This)->lpVtbl -> put_SliceHeight(This,v) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_get_Quantization(This,v)	\
    ( (This)->lpVtbl -> get_Quantization(This,v) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_put_Quantization(This,v)	\
    ( (This)->lpVtbl -> put_Quantization(This,v) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_get_CodingSignsHandling(This,v)	\
    ( (This)->lpVtbl -> get_CodingSignsHandling(This,v) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_put_CodingSignsHandling(This,v)	\
    ( (This)->lpVtbl -> put_CodingSignsHandling(This,v) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_get_CodingSignificance(This,v)	\
    ( (This)->lpVtbl -> get_CodingSignificance(This,v) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_put_CodingSignificance(This,v)	\
    ( (This)->lpVtbl -> put_CodingSignificance(This,v) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_get_EncBitsPerPixel(This,v)	\
    ( (This)->lpVtbl -> get_EncBitsPerPixel(This,v) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_put_EncBitsPerPixel(This,v)	\
    ( (This)->lpVtbl -> put_EncBitsPerPixel(This,v) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_get_RateControlMode(This,v)	\
    ( (This)->lpVtbl -> get_RateControlMode(This,v) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_put_RateControlMode(This,v)	\
    ( (This)->lpVtbl -> put_RateControlMode(This,v) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_get_CodingVerticalPredictionMode(This,v)	\
    ( (This)->lpVtbl -> get_CodingVerticalPredictionMode(This,v) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_put_CodingVerticalPredictionMode(This,v)	\
    ( (This)->lpVtbl -> put_CodingVerticalPredictionMode(This,v) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_get_SlicePacketizationMode(This,v)	\
    ( (This)->lpVtbl -> get_SlicePacketizationMode(This,v) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_put_SlicePacketizationMode(This,v)	\
    ( (This)->lpVtbl -> put_SlicePacketizationMode(This,v) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_get_CodingRawDisable(This,v)	\
    ( (This)->lpVtbl -> get_CodingRawDisable(This,v) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_put_CodingRawDisable(This,v)	\
    ( (This)->lpVtbl -> put_CodingRawDisable(This,v) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_get_CapCompat(This,v)	\
    ( (This)->lpVtbl -> get_CapCompat(This,v) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_put_CapCompat(This,v)	\
    ( (This)->lpVtbl -> put_CapCompat(This,v) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_get_ProfilePpihOverride(This,v)	\
    ( (This)->lpVtbl -> get_ProfilePpihOverride(This,v) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_put_ProfilePpihOverride(This,v)	\
    ( (This)->lpVtbl -> put_ProfilePpihOverride(This,v) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_get_LevelPlevOverride(This,v)	\
    ( (This)->lpVtbl -> get_LevelPlevOverride(This,v) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_put_LevelPlevOverride(This,v)	\
    ( (This)->lpVtbl -> put_LevelPlevOverride(This,v) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_get_CpuProfile(This,v)	\
    ( (This)->lpVtbl -> get_CpuProfile(This,v) ) 

#define ICC_JpegXS_VideoEncoderSettings_SVT_put_CpuProfile(This,v)	\
    ( (This)->lpVtbl -> put_CpuProfile(This,v) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_JpegXS_VideoEncoderSettings_SVT_INTERFACE_DEFINED__ */


#ifndef __ICC_JpegXS_VideoStreamInfo_SVT_INTERFACE_DEFINED__
#define __ICC_JpegXS_VideoStreamInfo_SVT_INTERFACE_DEFINED__

/* interface ICC_JpegXS_VideoStreamInfo_SVT */
/* [local][unique][uuid][object] */ 


EXTERN_C const IID IID_ICC_JpegXS_VideoStreamInfo_SVT;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("2a1debae-b8f9-40b2-a1ec-d3454d89c262")
    ICC_JpegXS_VideoStreamInfo_SVT : public ICC_VideoStreamInfo
    {
    public:
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_JpegXS_VideoStreamInfo_SVTVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_JpegXS_VideoStreamInfo_SVT * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_JpegXS_VideoStreamInfo_SVT * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_JpegXS_VideoStreamInfo_SVT * This);
        
        DECLSPEC_XFGVIRT(ICC_Settings, Clear)
        HRESULT ( STDMETHODCALLTYPE *Clear )( 
            ICC_JpegXS_VideoStreamInfo_SVT * This,
            /* [in] */ LPCSTR strVarName);
        
        DECLSPEC_XFGVIRT(ICC_Settings, Assigned)
        HRESULT ( STDMETHODCALLTYPE *Assigned )( 
            ICC_JpegXS_VideoStreamInfo_SVT * This,
            /* [in] */ LPCSTR strVarName,
            /* [defaultvalue][retval][out] */ CC_BOOL *__MIDL__ICC_Settings0000);
        
        DECLSPEC_XFGVIRT(ICC_Settings, get_XML)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_XML )( 
            ICC_JpegXS_VideoStreamInfo_SVT * This,
            /* [retval][out] */ CC_STRING *pstrXml);
        
        DECLSPEC_XFGVIRT(ICC_Settings, put_XML)
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_XML )( 
            ICC_JpegXS_VideoStreamInfo_SVT * This,
            /* [in] */ CC_STRING strXml);
        
        DECLSPEC_XFGVIRT(ICC_ElementaryStreamInfo, get_StreamType)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_StreamType )( 
            ICC_JpegXS_VideoStreamInfo_SVT * This,
            /* [retval][out] */ CC_ELEMENTARY_STREAM_TYPE *p);
        
        DECLSPEC_XFGVIRT(ICC_ElementaryStreamInfo, get_BitRate)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_BitRate )( 
            ICC_JpegXS_VideoStreamInfo_SVT * This,
            /* [retval][out] */ CC_BITRATE *p);
        
        DECLSPEC_XFGVIRT(ICC_ElementaryStreamInfo, get_FrameRate)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_FrameRate )( 
            ICC_JpegXS_VideoStreamInfo_SVT * This,
            /* [retval][out] */ CC_FRAME_RATE *p);
        
        DECLSPEC_XFGVIRT(ICC_VideoStreamInfo, get_FrameSize)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_FrameSize )( 
            ICC_JpegXS_VideoStreamInfo_SVT * This,
            /* [retval][out] */ CC_SIZE *s);
        
        DECLSPEC_XFGVIRT(ICC_VideoStreamInfo, get_AspectRatio)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_AspectRatio )( 
            ICC_JpegXS_VideoStreamInfo_SVT * This,
            /* [retval][out] */ CC_RATIONAL *a);
        
        DECLSPEC_XFGVIRT(ICC_VideoStreamInfo, get_ProgressiveSequence)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_ProgressiveSequence )( 
            ICC_JpegXS_VideoStreamInfo_SVT * This,
            /* [retval][out] */ CC_BOOL *x);
        
        END_INTERFACE
    } ICC_JpegXS_VideoStreamInfo_SVTVtbl;

    interface ICC_JpegXS_VideoStreamInfo_SVT
    {
        CONST_VTBL struct ICC_JpegXS_VideoStreamInfo_SVTVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_JpegXS_VideoStreamInfo_SVT_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_JpegXS_VideoStreamInfo_SVT_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_JpegXS_VideoStreamInfo_SVT_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ICC_JpegXS_VideoStreamInfo_SVT_Clear(This,strVarName)	\
    ( (This)->lpVtbl -> Clear(This,strVarName) ) 

#define ICC_JpegXS_VideoStreamInfo_SVT_Assigned(This,strVarName,__MIDL__ICC_Settings0000)	\
    ( (This)->lpVtbl -> Assigned(This,strVarName,__MIDL__ICC_Settings0000) ) 

#define ICC_JpegXS_VideoStreamInfo_SVT_get_XML(This,pstrXml)	\
    ( (This)->lpVtbl -> get_XML(This,pstrXml) ) 

#define ICC_JpegXS_VideoStreamInfo_SVT_put_XML(This,strXml)	\
    ( (This)->lpVtbl -> put_XML(This,strXml) ) 


#define ICC_JpegXS_VideoStreamInfo_SVT_get_StreamType(This,p)	\
    ( (This)->lpVtbl -> get_StreamType(This,p) ) 

#define ICC_JpegXS_VideoStreamInfo_SVT_get_BitRate(This,p)	\
    ( (This)->lpVtbl -> get_BitRate(This,p) ) 

#define ICC_JpegXS_VideoStreamInfo_SVT_get_FrameRate(This,p)	\
    ( (This)->lpVtbl -> get_FrameRate(This,p) ) 


#define ICC_JpegXS_VideoStreamInfo_SVT_get_FrameSize(This,s)	\
    ( (This)->lpVtbl -> get_FrameSize(This,s) ) 

#define ICC_JpegXS_VideoStreamInfo_SVT_get_AspectRatio(This,a)	\
    ( (This)->lpVtbl -> get_AspectRatio(This,a) ) 

#define ICC_JpegXS_VideoStreamInfo_SVT_get_ProgressiveSequence(This,x)	\
    ( (This)->lpVtbl -> get_ProgressiveSequence(This,x) ) 


#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_JpegXS_VideoStreamInfo_SVT_INTERFACE_DEFINED__ */


#ifndef __ICC_JpegXS_VideoFrameInfo_SVT_INTERFACE_DEFINED__
#define __ICC_JpegXS_VideoFrameInfo_SVT_INTERFACE_DEFINED__

/* interface ICC_JpegXS_VideoFrameInfo_SVT */
/* [local][unique][uuid][object] */ 


EXTERN_C const IID IID_ICC_JpegXS_VideoFrameInfo_SVT;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("36286a62-6c51-4c02-8149-9b8cf5557991")
    ICC_JpegXS_VideoFrameInfo_SVT : public ICC_VideoFrameInfo
    {
    public:
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_JpegXS_VideoFrameInfo_SVTVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_JpegXS_VideoFrameInfo_SVT * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_JpegXS_VideoFrameInfo_SVT * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_JpegXS_VideoFrameInfo_SVT * This);
        
        DECLSPEC_XFGVIRT(ICC_Settings, Clear)
        HRESULT ( STDMETHODCALLTYPE *Clear )( 
            ICC_JpegXS_VideoFrameInfo_SVT * This,
            /* [in] */ LPCSTR strVarName);
        
        DECLSPEC_XFGVIRT(ICC_Settings, Assigned)
        HRESULT ( STDMETHODCALLTYPE *Assigned )( 
            ICC_JpegXS_VideoFrameInfo_SVT * This,
            /* [in] */ LPCSTR strVarName,
            /* [defaultvalue][retval][out] */ CC_BOOL *__MIDL__ICC_Settings0000);
        
        DECLSPEC_XFGVIRT(ICC_Settings, get_XML)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_XML )( 
            ICC_JpegXS_VideoFrameInfo_SVT * This,
            /* [retval][out] */ CC_STRING *pstrXml);
        
        DECLSPEC_XFGVIRT(ICC_Settings, put_XML)
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_XML )( 
            ICC_JpegXS_VideoFrameInfo_SVT * This,
            /* [in] */ CC_STRING strXml);
        
        DECLSPEC_XFGVIRT(ICC_ByteStreamDataInfo, get_DataSize)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_DataSize )( 
            ICC_JpegXS_VideoFrameInfo_SVT * This,
            /* [retval][out] */ CC_UINT *__MIDL__ICC_ByteStreamDataInfo0000);
        
        DECLSPEC_XFGVIRT(ICC_ByteStreamDataInfo, get_ByteOffset)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_ByteOffset )( 
            ICC_JpegXS_VideoFrameInfo_SVT * This,
            /* [retval][out] */ CC_OFFSET *__MIDL__ICC_ByteStreamDataInfo0001);
        
        DECLSPEC_XFGVIRT(ICC_ElementaryDataInfo, get_NumSamples)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_NumSamples )( 
            ICC_JpegXS_VideoFrameInfo_SVT * This,
            /* [retval][out] */ CC_UINT *__MIDL__ICC_ElementaryDataInfo0000);
        
        DECLSPEC_XFGVIRT(ICC_ElementaryDataInfo, get_SampleOffset)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_SampleOffset )( 
            ICC_JpegXS_VideoFrameInfo_SVT * This,
            /* [retval][out] */ CC_OFFSET *__MIDL__ICC_ElementaryDataInfo0001);
        
        DECLSPEC_XFGVIRT(ICC_ElementaryDataInfo, get_Duration)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Duration )( 
            ICC_JpegXS_VideoFrameInfo_SVT * This,
            /* [retval][out] */ CC_TIME *__MIDL__ICC_ElementaryDataInfo0002);
        
        DECLSPEC_XFGVIRT(ICC_ElementaryDataInfo, get_PresentationDelta)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_PresentationDelta )( 
            ICC_JpegXS_VideoFrameInfo_SVT * This,
            /* [retval][out] */ CC_INT *__MIDL__ICC_ElementaryDataInfo0003);
        
        DECLSPEC_XFGVIRT(ICC_ElementaryDataInfo, get_PTS)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_PTS )( 
            ICC_JpegXS_VideoFrameInfo_SVT * This,
            /* [retval][out] */ CC_TIME *__MIDL__ICC_ElementaryDataInfo0004);
        
        DECLSPEC_XFGVIRT(ICC_ElementaryDataInfo, get_DTS)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_DTS )( 
            ICC_JpegXS_VideoFrameInfo_SVT * This,
            /* [retval][out] */ CC_TIME *__MIDL__ICC_ElementaryDataInfo0005);
        
        DECLSPEC_XFGVIRT(ICC_ElementaryDataInfo, get_SequenceEntryFlag)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_SequenceEntryFlag )( 
            ICC_JpegXS_VideoFrameInfo_SVT * This,
            /* [retval][out] */ CC_BOOL *__MIDL__ICC_ElementaryDataInfo0006);
        
        DECLSPEC_XFGVIRT(ICC_VideoFrameInfo, get_Number)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Number )( 
            ICC_JpegXS_VideoFrameInfo_SVT * This,
            /* [retval][out] */ CC_UINT *n);
        
        DECLSPEC_XFGVIRT(ICC_VideoFrameInfo, get_CodingNumber)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_CodingNumber )( 
            ICC_JpegXS_VideoFrameInfo_SVT * This,
            /* [retval][out] */ CC_UINT *c);
        
        DECLSPEC_XFGVIRT(ICC_VideoFrameInfo, get_TimeCode)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_TimeCode )( 
            ICC_JpegXS_VideoFrameInfo_SVT * This,
            /* [retval][out] */ CC_TIMECODE *t);
        
        DECLSPEC_XFGVIRT(ICC_VideoFrameInfo, get_InterlaceType)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_InterlaceType )( 
            ICC_JpegXS_VideoFrameInfo_SVT * This,
            /* [retval][out] */ CC_INTERLACE_TYPE *i);
        
        DECLSPEC_XFGVIRT(ICC_VideoFrameInfo, get_FrameType)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_FrameType )( 
            ICC_JpegXS_VideoFrameInfo_SVT * This,
            /* [retval][out] */ CC_FRAME_TYPE *x);
        
        DECLSPEC_XFGVIRT(ICC_VideoFrameInfo, get_PictStruct)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_PictStruct )( 
            ICC_JpegXS_VideoFrameInfo_SVT * This,
            /* [retval][out] */ CC_PICTURE_STRUCTURE *x);
        
        DECLSPEC_XFGVIRT(ICC_VideoFrameInfo, get_Flags)
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Flags )( 
            ICC_JpegXS_VideoFrameInfo_SVT * This,
            /* [retval][out] */ DWORD *x);
        
        END_INTERFACE
    } ICC_JpegXS_VideoFrameInfo_SVTVtbl;

    interface ICC_JpegXS_VideoFrameInfo_SVT
    {
        CONST_VTBL struct ICC_JpegXS_VideoFrameInfo_SVTVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_JpegXS_VideoFrameInfo_SVT_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_JpegXS_VideoFrameInfo_SVT_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_JpegXS_VideoFrameInfo_SVT_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ICC_JpegXS_VideoFrameInfo_SVT_Clear(This,strVarName)	\
    ( (This)->lpVtbl -> Clear(This,strVarName) ) 

#define ICC_JpegXS_VideoFrameInfo_SVT_Assigned(This,strVarName,__MIDL__ICC_Settings0000)	\
    ( (This)->lpVtbl -> Assigned(This,strVarName,__MIDL__ICC_Settings0000) ) 

#define ICC_JpegXS_VideoFrameInfo_SVT_get_XML(This,pstrXml)	\
    ( (This)->lpVtbl -> get_XML(This,pstrXml) ) 

#define ICC_JpegXS_VideoFrameInfo_SVT_put_XML(This,strXml)	\
    ( (This)->lpVtbl -> put_XML(This,strXml) ) 


#define ICC_JpegXS_VideoFrameInfo_SVT_get_DataSize(This,__MIDL__ICC_ByteStreamDataInfo0000)	\
    ( (This)->lpVtbl -> get_DataSize(This,__MIDL__ICC_ByteStreamDataInfo0000) ) 

#define ICC_JpegXS_VideoFrameInfo_SVT_get_ByteOffset(This,__MIDL__ICC_ByteStreamDataInfo0001)	\
    ( (This)->lpVtbl -> get_ByteOffset(This,__MIDL__ICC_ByteStreamDataInfo0001) ) 


#define ICC_JpegXS_VideoFrameInfo_SVT_get_NumSamples(This,__MIDL__ICC_ElementaryDataInfo0000)	\
    ( (This)->lpVtbl -> get_NumSamples(This,__MIDL__ICC_ElementaryDataInfo0000) ) 

#define ICC_JpegXS_VideoFrameInfo_SVT_get_SampleOffset(This,__MIDL__ICC_ElementaryDataInfo0001)	\
    ( (This)->lpVtbl -> get_SampleOffset(This,__MIDL__ICC_ElementaryDataInfo0001) ) 

#define ICC_JpegXS_VideoFrameInfo_SVT_get_Duration(This,__MIDL__ICC_ElementaryDataInfo0002)	\
    ( (This)->lpVtbl -> get_Duration(This,__MIDL__ICC_ElementaryDataInfo0002) ) 

#define ICC_JpegXS_VideoFrameInfo_SVT_get_PresentationDelta(This,__MIDL__ICC_ElementaryDataInfo0003)	\
    ( (This)->lpVtbl -> get_PresentationDelta(This,__MIDL__ICC_ElementaryDataInfo0003) ) 

#define ICC_JpegXS_VideoFrameInfo_SVT_get_PTS(This,__MIDL__ICC_ElementaryDataInfo0004)	\
    ( (This)->lpVtbl -> get_PTS(This,__MIDL__ICC_ElementaryDataInfo0004) ) 

#define ICC_JpegXS_VideoFrameInfo_SVT_get_DTS(This,__MIDL__ICC_ElementaryDataInfo0005)	\
    ( (This)->lpVtbl -> get_DTS(This,__MIDL__ICC_ElementaryDataInfo0005) ) 

#define ICC_JpegXS_VideoFrameInfo_SVT_get_SequenceEntryFlag(This,__MIDL__ICC_ElementaryDataInfo0006)	\
    ( (This)->lpVtbl -> get_SequenceEntryFlag(This,__MIDL__ICC_ElementaryDataInfo0006) ) 


#define ICC_JpegXS_VideoFrameInfo_SVT_get_Number(This,n)	\
    ( (This)->lpVtbl -> get_Number(This,n) ) 

#define ICC_JpegXS_VideoFrameInfo_SVT_get_CodingNumber(This,c)	\
    ( (This)->lpVtbl -> get_CodingNumber(This,c) ) 

#define ICC_JpegXS_VideoFrameInfo_SVT_get_TimeCode(This,t)	\
    ( (This)->lpVtbl -> get_TimeCode(This,t) ) 

#define ICC_JpegXS_VideoFrameInfo_SVT_get_InterlaceType(This,i)	\
    ( (This)->lpVtbl -> get_InterlaceType(This,i) ) 

#define ICC_JpegXS_VideoFrameInfo_SVT_get_FrameType(This,x)	\
    ( (This)->lpVtbl -> get_FrameType(This,x) ) 

#define ICC_JpegXS_VideoFrameInfo_SVT_get_PictStruct(This,x)	\
    ( (This)->lpVtbl -> get_PictStruct(This,x) ) 

#define ICC_JpegXS_VideoFrameInfo_SVT_get_Flags(This,x)	\
    ( (This)->lpVtbl -> get_Flags(This,x) ) 


#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_JpegXS_VideoFrameInfo_SVT_INTERFACE_DEFINED__ */



#ifndef __Cinecoder_Plugin_JpegXS_SVT_LIBRARY_DEFINED__
#define __Cinecoder_Plugin_JpegXS_SVT_LIBRARY_DEFINED__

/* library Cinecoder_Plugin_JpegXS_SVT */
/* [uuid] */ 


EXTERN_C const IID LIBID_Cinecoder_Plugin_JpegXS_SVT;

EXTERN_C const CLSID CLSID_CC_JpegXS_VideoDecoder_SVT;

#ifdef __cplusplus

class DECLSPEC_UUID("55d08907-c686-4f34-a47a-85ad285cf0e7")
CC_JpegXS_VideoDecoder_SVT;
#endif

EXTERN_C const CLSID CLSID_CC_JpegXS_VideoEncoder_SVT;

#ifdef __cplusplus

class DECLSPEC_UUID("601e23e1-6ede-434d-a43d-b651a6a75bf1")
CC_JpegXS_VideoEncoder_SVT;
#endif

EXTERN_C const CLSID CLSID_CC_JpegXS_VideoEncoderSettings_SVT;

#ifdef __cplusplus

class DECLSPEC_UUID("051f439c-04a8-48eb-a1c1-8fbef3105f66")
CC_JpegXS_VideoEncoderSettings_SVT;
#endif
#endif /* __Cinecoder_Plugin_JpegXS_SVT_LIBRARY_DEFINED__ */

/* Additional Prototypes for ALL interfaces */

/* end of Additional Prototypes */

#ifdef __cplusplus
}
#endif

#endif


