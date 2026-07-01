

/* this ALWAYS GENERATED file contains the definitions for the interfaces */


 /* File created by MIDL compiler version 8.00.0603 */
/* at Tue Aug 01 07:10:52 2023
 */
/* Compiler settings for Cinecoder.Plugin.Codecs.DNxHD.idl:
    Oicf, W1, Zp8, env=Win64 (32b run), target_arch=AMD64 8.00.0603 
    protocol : dce , ms_ext, c_ext, robust
    error checks: allocation ref bounds_check enum stub_data 
    VC __declspec() decoration level: 
         __declspec(uuid()), __declspec(selectany), __declspec(novtable)
         DECLSPEC_UUID(), MIDL_INTERFACE()
*/
/* @@MIDL_FILE_HEADING(  ) */

#pragma warning( disable: 4049 )  /* more than 64k source lines */


/* verify that the <rpcndr.h> version is high enough to compile this file*/
#ifndef __REQUIRED_RPCNDR_H_VERSION__
#define __REQUIRED_RPCNDR_H_VERSION__ 475
#endif

#include "rpc.h"
#include "rpcndr.h"

#ifndef __RPCNDR_H_VERSION__
#error this stub requires an updated version of <rpcndr.h>
#endif // __RPCNDR_H_VERSION__

#ifndef COM_NO_WINDOWS_H
#include "windows.h"
#include "ole2.h"
#endif /*COM_NO_WINDOWS_H*/

#ifndef __Cinecoder2EPlugin2ECodecs2EDNxHD_h__
#define __Cinecoder2EPlugin2ECodecs2EDNxHD_h__

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
#pragma once
#endif

/* Forward Declarations */ 

#ifndef __ICC_DNxHD_VideoEncoder_FWD_DEFINED__
#define __ICC_DNxHD_VideoEncoder_FWD_DEFINED__
typedef interface ICC_DNxHD_VideoEncoder ICC_DNxHD_VideoEncoder;

#endif 	/* __ICC_DNxHD_VideoEncoder_FWD_DEFINED__ */


#ifndef __ICC_DNxHD_VideoStreamInfo_FWD_DEFINED__
#define __ICC_DNxHD_VideoStreamInfo_FWD_DEFINED__
typedef interface ICC_DNxHD_VideoStreamInfo ICC_DNxHD_VideoStreamInfo;

#endif 	/* __ICC_DNxHD_VideoStreamInfo_FWD_DEFINED__ */


#ifndef __ICC_DNxHD_VideoEncoderSettings_FWD_DEFINED__
#define __ICC_DNxHD_VideoEncoderSettings_FWD_DEFINED__
typedef interface ICC_DNxHD_VideoEncoderSettings ICC_DNxHD_VideoEncoderSettings;

#endif 	/* __ICC_DNxHD_VideoEncoderSettings_FWD_DEFINED__ */


#ifndef __ICC_DNX_VideoEncoderSettings_FWD_DEFINED__
#define __ICC_DNX_VideoEncoderSettings_FWD_DEFINED__
typedef interface ICC_DNX_VideoEncoderSettings ICC_DNX_VideoEncoderSettings;

#endif 	/* __ICC_DNX_VideoEncoderSettings_FWD_DEFINED__ */


#ifndef __ICC_DNX_VideoStreamInfo_FWD_DEFINED__
#define __ICC_DNX_VideoStreamInfo_FWD_DEFINED__
typedef interface ICC_DNX_VideoStreamInfo ICC_DNX_VideoStreamInfo;

#endif 	/* __ICC_DNX_VideoStreamInfo_FWD_DEFINED__ */


#ifndef __ICC_DNX_VideoEncoder_FWD_DEFINED__
#define __ICC_DNX_VideoEncoder_FWD_DEFINED__
typedef interface ICC_DNX_VideoEncoder ICC_DNX_VideoEncoder;

#endif 	/* __ICC_DNX_VideoEncoder_FWD_DEFINED__ */


#ifndef __CC_DNxHD_VideoEncoder_FWD_DEFINED__
#define __CC_DNxHD_VideoEncoder_FWD_DEFINED__

#ifdef __cplusplus
typedef class CC_DNxHD_VideoEncoder CC_DNxHD_VideoEncoder;
#else
typedef struct CC_DNxHD_VideoEncoder CC_DNxHD_VideoEncoder;
#endif /* __cplusplus */

#endif 	/* __CC_DNxHD_VideoEncoder_FWD_DEFINED__ */


#ifndef __CC_DNxHD_VideoEncoderSettings_FWD_DEFINED__
#define __CC_DNxHD_VideoEncoderSettings_FWD_DEFINED__

#ifdef __cplusplus
typedef class CC_DNxHD_VideoEncoderSettings CC_DNxHD_VideoEncoderSettings;
#else
typedef struct CC_DNxHD_VideoEncoderSettings CC_DNxHD_VideoEncoderSettings;
#endif /* __cplusplus */

#endif 	/* __CC_DNxHD_VideoEncoderSettings_FWD_DEFINED__ */


#ifndef __CC_DNX_VideoEncoderSettings_FWD_DEFINED__
#define __CC_DNX_VideoEncoderSettings_FWD_DEFINED__

#ifdef __cplusplus
typedef class CC_DNX_VideoEncoderSettings CC_DNX_VideoEncoderSettings;
#else
typedef struct CC_DNX_VideoEncoderSettings CC_DNX_VideoEncoderSettings;
#endif /* __cplusplus */

#endif 	/* __CC_DNX_VideoEncoderSettings_FWD_DEFINED__ */


#ifndef __CC_DNX_VideoEncoder_FWD_DEFINED__
#define __CC_DNX_VideoEncoder_FWD_DEFINED__

#ifdef __cplusplus
typedef class CC_DNX_VideoEncoder CC_DNX_VideoEncoder;
#else
typedef struct CC_DNX_VideoEncoder CC_DNX_VideoEncoder;
#endif /* __cplusplus */

#endif 	/* __CC_DNX_VideoEncoder_FWD_DEFINED__ */


/* header files for imported files */
#include "oaidl.h"
#include "ocidl.h"

#ifdef __cplusplus
extern "C"{
#endif 


/* interface __MIDL_itf_Cinecoder2EPlugin2ECodecs2EDNxHD_0000_0000 */
/* [local] */ 

typedef /* [v1_enum] */ 
enum CC_DNxHD_COMPRESSION_ID
    {
        CC_DNxHD_CID_UNKNOWN	= 0,
        CC_DNxHD_CID_220X_1080p	= 1235,
        CC_DNxHD_CID_145_1080p	= 1237,
        CC_DNxHD_CID_220_1080p	= 1238,
        CC_DNxHD_CID_220X_720p	= 1250,
        CC_DNxHD_CID_220_720p	= 1251,
        CC_DNxHD_CID_145_720p	= 1252,
        CC_DNxHD_CID_220X_1080i	= 1241,
        CC_DNxHD_CID_145_1080i	= 1242,
        CC_DNxHD_CID_220_1080i	= 1243,
        CC_DNxHD_CID_36_1080p	= 1253
    } 	CC_DNxHD_CID;

typedef /* [v1_enum] */ 
enum CC_DNX_COMPRESSION_ID
    {
        CC_DNX_UNKNOWN_COMPRESSION_ID	= 0,
        CC_DNX_HQX_1080p_COMPRESSION_ID	= 1235,
        CC_DNX_SQ_1080p_COMPRESSION_ID	= 1237,
        CC_DNX_HQ_1080p_COMPRESSION_ID	= 1238,
        CC_DNX_HQX_720p_COMPRESSION_ID	= 1250,
        CC_DNX_HQ_720p_COMPRESSION_ID	= 1251,
        CC_DNX_SQ_720p_COMPRESSION_ID	= 1252,
        CC_DNX_HQX_1080i_COMPRESSION_ID	= 1241,
        CC_DNX_SQ_1080i_COMPRESSION_ID	= 1242,
        CC_DNX_HQ_1080i_COMPRESSION_ID	= 1243,
        CC_DNX_HQ_TR_1080i_COMPRESSION_ID	= 1244,
        CC_DNX_LB_1080p_COMPRESSION_ID	= 1253,
        CC_DNX_444_1080p_COMPRESSION_ID	= 1256,
        CC_DNX_SQ_TR_720p_COMPRESSION_ID	= 1258,
        CC_DNX_SQ_TR_1080p_COMPRESSION_ID	= 1259,
        CC_DNX_SQ_TR_1080i_COMPRESSION_ID	= 1260,
        CC_DNX_444_COMPRESSION_ID	= 1270,
        CC_DNX_HQX_COMPRESSION_ID	= 1271,
        CC_DNX_HQ_COMPRESSION_ID	= 1272,
        CC_DNX_SQ_COMPRESSION_ID	= 1273,
        CC_DNX_LB_COMPRESSION_ID	= 1274
    } 	CC_DNX_COMPRESSION_ID;

typedef /* [v1_enum] */ 
enum CC_DNX_COLOR_VOLUME
    {
        CC_DNX_CV_INVALID	= 0,
        CC_DNX_CV_709	= 0x1,
        CC_DNX_CV_2020	= 0x2,
        CC_DNX_CV_2020c	= 0x4,
        CC_DNX_CV_OutOfBand	= 0x8
    } 	CC_DNX_COLOR_VOLUME;

typedef /* [v1_enum] */ 
enum CC_DNX_COLOR_FORMAT
    {
        CC_DNX_CF_INVALID	= 0,
        CC_DNX_CF_YCbCr	= 0x1,
        CC_DNX_CF_RGB	= 0x2
    } 	CC_DNX_COLOR_FORMAT;



extern RPC_IF_HANDLE __MIDL_itf_Cinecoder2EPlugin2ECodecs2EDNxHD_0000_0000_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_Cinecoder2EPlugin2ECodecs2EDNxHD_0000_0000_v0_0_s_ifspec;

#ifndef __ICC_DNxHD_VideoEncoder_INTERFACE_DEFINED__
#define __ICC_DNxHD_VideoEncoder_INTERFACE_DEFINED__

/* interface ICC_DNxHD_VideoEncoder */
/* [local][unique][uuid][object] */ 


EXTERN_C const IID IID_ICC_DNxHD_VideoEncoder;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("495AE5F1-C244-42F5-B41C-9C2FB86F97DE")
    ICC_DNxHD_VideoEncoder : public ICC_VideoEncoder
    {
    public:
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_DNxHD_VideoEncoderVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_DNxHD_VideoEncoder * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_DNxHD_VideoEncoder * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_DNxHD_VideoEncoder * This);
        
        HRESULT ( STDMETHODCALLTYPE *Init )( 
            ICC_DNxHD_VideoEncoder * This,
            /* [defaultvalue][in] */ ICC_Settings *pSettings);
        
        HRESULT ( STDMETHODCALLTYPE *InitByXml )( 
            ICC_DNxHD_VideoEncoder * This,
            /* [in] */ CC_STRING strXML);
        
        HRESULT ( STDMETHODCALLTYPE *Done )( 
            ICC_DNxHD_VideoEncoder * This,
            /* [in] */ CC_BOOL bFlush,
            /* [defaultvalue][retval][out] */ CC_BOOL *pbDone);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_IsActive )( 
            ICC_DNxHD_VideoEncoder * This,
            /* [defaultvalue][retval][out] */ CC_BOOL *__MIDL__ICC_StreamProcessor0000);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_TimeBase )( 
            ICC_DNxHD_VideoEncoder * This,
            /* [retval][out] */ CC_TIMEBASE *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_TimeBase )( 
            ICC_DNxHD_VideoEncoder * This,
            /* [in] */ CC_TIMEBASE p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_BitRate )( 
            ICC_DNxHD_VideoEncoder * This,
            /* [retval][out] */ CC_BITRATE *p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_StreamInfo )( 
            ICC_DNxHD_VideoEncoder * This,
            /* [retval][out] */ ICC_Settings **p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_IsDataReady )( 
            ICC_DNxHD_VideoEncoder * This,
            /* [defaultvalue][retval][out] */ CC_BOOL *p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_DataInfo )( 
            ICC_DNxHD_VideoEncoder * This,
            /* [retval][out] */ ICC_Settings **s);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_OutputCallback )( 
            ICC_DNxHD_VideoEncoder * This,
            /* [retval][out] */ IUnknown **p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_OutputCallback )( 
            ICC_DNxHD_VideoEncoder * This,
            /* [in] */ IUnknown *p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_DataSize )( 
            ICC_DNxHD_VideoEncoder * This,
            /* [retval][out] */ CC_UINT *__MIDL__ICC_Encoder0000);
        
        HRESULT ( STDMETHODCALLTYPE *GetData )( 
            ICC_DNxHD_VideoEncoder * This,
            /* [size_is][out] */ CC_PBYTE pbData,
            /* [in] */ CC_UINT cbBufSize,
            /* [defaultvalue][retval][out] */ CC_UINT *pcbRetSize);
        
        HRESULT ( STDMETHODCALLTYPE *AddFrame )( 
            ICC_DNxHD_VideoEncoder * This,
            /* [in] */ CC_COLOR_FMT Format,
            /* [size_is][in] */ const BYTE *pData,
            /* [in] */ DWORD cbSize,
            /* [defaultvalue][in] */ INT stride,
            /* [defaultvalue][retval][out] */ CC_BOOL *pResult);
        
        HRESULT ( STDMETHODCALLTYPE *GetStride )( 
            ICC_DNxHD_VideoEncoder * This,
            /* [in] */ CC_COLOR_FMT fmt,
            /* [retval][out] */ DWORD *pNumBytes);
        
        HRESULT ( STDMETHODCALLTYPE *IsFormatSupported )( 
            ICC_DNxHD_VideoEncoder * This,
            /* [in] */ CC_COLOR_FMT fmt,
            /* [defaultvalue][retval][out] */ CC_BOOL *pResult);
        
        HRESULT ( STDMETHODCALLTYPE *AddScaleFrame )( 
            ICC_DNxHD_VideoEncoder * This,
            /* [size_is][in] */ const BYTE *pData,
            /* [in] */ DWORD cbSize,
            /* [in] */ CC_ADD_VIDEO_FRAME_PARAMS *pParams,
            /* [defaultvalue][retval][out] */ CC_BOOL *pResult);
        
        HRESULT ( STDMETHODCALLTYPE *IsScaleAvailable )( 
            ICC_DNxHD_VideoEncoder * This,
            /* [in] */ CC_ADD_VIDEO_FRAME_PARAMS *pParams,
            /* [defaultvalue][retval][out] */ CC_BOOL *__MIDL__ICC_VideoEncoder0000);
        
        HRESULT ( STDMETHODCALLTYPE *GetVideoStreamInfo )( 
            ICC_DNxHD_VideoEncoder * This,
            /* [retval][out] */ ICC_VideoStreamInfo **pDescr);
        
        HRESULT ( STDMETHODCALLTYPE *GetVideoFrameInfo )( 
            ICC_DNxHD_VideoEncoder * This,
            /* [retval][out] */ ICC_VideoFrameInfo **pDescr);
        
        END_INTERFACE
    } ICC_DNxHD_VideoEncoderVtbl;

    interface ICC_DNxHD_VideoEncoder
    {
        CONST_VTBL struct ICC_DNxHD_VideoEncoderVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_DNxHD_VideoEncoder_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_DNxHD_VideoEncoder_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_DNxHD_VideoEncoder_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ICC_DNxHD_VideoEncoder_Init(This,pSettings)	\
    ( (This)->lpVtbl -> Init(This,pSettings) ) 

#define ICC_DNxHD_VideoEncoder_InitByXml(This,strXML)	\
    ( (This)->lpVtbl -> InitByXml(This,strXML) ) 

#define ICC_DNxHD_VideoEncoder_Done(This,bFlush,pbDone)	\
    ( (This)->lpVtbl -> Done(This,bFlush,pbDone) ) 

#define ICC_DNxHD_VideoEncoder_get_IsActive(This,__MIDL__ICC_StreamProcessor0000)	\
    ( (This)->lpVtbl -> get_IsActive(This,__MIDL__ICC_StreamProcessor0000) ) 

#define ICC_DNxHD_VideoEncoder_get_TimeBase(This,p)	\
    ( (This)->lpVtbl -> get_TimeBase(This,p) ) 

#define ICC_DNxHD_VideoEncoder_put_TimeBase(This,p)	\
    ( (This)->lpVtbl -> put_TimeBase(This,p) ) 

#define ICC_DNxHD_VideoEncoder_get_BitRate(This,p)	\
    ( (This)->lpVtbl -> get_BitRate(This,p) ) 

#define ICC_DNxHD_VideoEncoder_get_StreamInfo(This,p)	\
    ( (This)->lpVtbl -> get_StreamInfo(This,p) ) 

#define ICC_DNxHD_VideoEncoder_get_IsDataReady(This,p)	\
    ( (This)->lpVtbl -> get_IsDataReady(This,p) ) 

#define ICC_DNxHD_VideoEncoder_get_DataInfo(This,s)	\
    ( (This)->lpVtbl -> get_DataInfo(This,s) ) 

#define ICC_DNxHD_VideoEncoder_get_OutputCallback(This,p)	\
    ( (This)->lpVtbl -> get_OutputCallback(This,p) ) 

#define ICC_DNxHD_VideoEncoder_put_OutputCallback(This,p)	\
    ( (This)->lpVtbl -> put_OutputCallback(This,p) ) 


#define ICC_DNxHD_VideoEncoder_get_DataSize(This,__MIDL__ICC_Encoder0000)	\
    ( (This)->lpVtbl -> get_DataSize(This,__MIDL__ICC_Encoder0000) ) 

#define ICC_DNxHD_VideoEncoder_GetData(This,pbData,cbBufSize,pcbRetSize)	\
    ( (This)->lpVtbl -> GetData(This,pbData,cbBufSize,pcbRetSize) ) 


#define ICC_DNxHD_VideoEncoder_AddFrame(This,Format,pData,cbSize,stride,pResult)	\
    ( (This)->lpVtbl -> AddFrame(This,Format,pData,cbSize,stride,pResult) ) 

#define ICC_DNxHD_VideoEncoder_GetStride(This,fmt,pNumBytes)	\
    ( (This)->lpVtbl -> GetStride(This,fmt,pNumBytes) ) 

#define ICC_DNxHD_VideoEncoder_IsFormatSupported(This,fmt,pResult)	\
    ( (This)->lpVtbl -> IsFormatSupported(This,fmt,pResult) ) 

#define ICC_DNxHD_VideoEncoder_AddScaleFrame(This,pData,cbSize,pParams,pResult)	\
    ( (This)->lpVtbl -> AddScaleFrame(This,pData,cbSize,pParams,pResult) ) 

#define ICC_DNxHD_VideoEncoder_IsScaleAvailable(This,pParams,__MIDL__ICC_VideoEncoder0000)	\
    ( (This)->lpVtbl -> IsScaleAvailable(This,pParams,__MIDL__ICC_VideoEncoder0000) ) 

#define ICC_DNxHD_VideoEncoder_GetVideoStreamInfo(This,pDescr)	\
    ( (This)->lpVtbl -> GetVideoStreamInfo(This,pDescr) ) 

#define ICC_DNxHD_VideoEncoder_GetVideoFrameInfo(This,pDescr)	\
    ( (This)->lpVtbl -> GetVideoFrameInfo(This,pDescr) ) 


#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_DNxHD_VideoEncoder_INTERFACE_DEFINED__ */


#ifndef __ICC_DNxHD_VideoStreamInfo_INTERFACE_DEFINED__
#define __ICC_DNxHD_VideoStreamInfo_INTERFACE_DEFINED__

/* interface ICC_DNxHD_VideoStreamInfo */
/* [object][uuid] */ 


EXTERN_C const IID IID_ICC_DNxHD_VideoStreamInfo;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("449E6E5A-BE8C-4900-8910-0083C6E2DD42")
    ICC_DNxHD_VideoStreamInfo : public ICC_VideoStreamInfo
    {
    public:
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_DNxHD_VideoStreamInfoVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_DNxHD_VideoStreamInfo * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_DNxHD_VideoStreamInfo * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_DNxHD_VideoStreamInfo * This);
        
        HRESULT ( STDMETHODCALLTYPE *Clear )( 
            ICC_DNxHD_VideoStreamInfo * This,
            /* [in] */ LPCSTR strVarName);
        
        HRESULT ( STDMETHODCALLTYPE *Assigned )( 
            ICC_DNxHD_VideoStreamInfo * This,
            /* [in] */ LPCSTR strVarName,
            /* [defaultvalue][retval][out] */ CC_BOOL *__MIDL__ICC_Settings0000);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_XML )( 
            ICC_DNxHD_VideoStreamInfo * This,
            /* [retval][out] */ CC_STRING *pstrXml);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_XML )( 
            ICC_DNxHD_VideoStreamInfo * This,
            /* [in] */ CC_STRING strXml);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_StreamType )( 
            ICC_DNxHD_VideoStreamInfo * This,
            /* [retval][out] */ CC_ELEMENTARY_STREAM_TYPE *p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_BitRate )( 
            ICC_DNxHD_VideoStreamInfo * This,
            /* [retval][out] */ CC_BITRATE *p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_FrameRate )( 
            ICC_DNxHD_VideoStreamInfo * This,
            /* [retval][out] */ CC_FRAME_RATE *p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_FrameSize )( 
            ICC_DNxHD_VideoStreamInfo * This,
            /* [retval][out] */ CC_SIZE *s);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_AspectRatio )( 
            ICC_DNxHD_VideoStreamInfo * This,
            /* [retval][out] */ CC_RATIONAL *a);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_ProgressiveSequence )( 
            ICC_DNxHD_VideoStreamInfo * This,
            /* [retval][out] */ CC_BOOL *x);
        
        END_INTERFACE
    } ICC_DNxHD_VideoStreamInfoVtbl;

    interface ICC_DNxHD_VideoStreamInfo
    {
        CONST_VTBL struct ICC_DNxHD_VideoStreamInfoVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_DNxHD_VideoStreamInfo_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_DNxHD_VideoStreamInfo_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_DNxHD_VideoStreamInfo_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ICC_DNxHD_VideoStreamInfo_Clear(This,strVarName)	\
    ( (This)->lpVtbl -> Clear(This,strVarName) ) 

#define ICC_DNxHD_VideoStreamInfo_Assigned(This,strVarName,__MIDL__ICC_Settings0000)	\
    ( (This)->lpVtbl -> Assigned(This,strVarName,__MIDL__ICC_Settings0000) ) 

#define ICC_DNxHD_VideoStreamInfo_get_XML(This,pstrXml)	\
    ( (This)->lpVtbl -> get_XML(This,pstrXml) ) 

#define ICC_DNxHD_VideoStreamInfo_put_XML(This,strXml)	\
    ( (This)->lpVtbl -> put_XML(This,strXml) ) 


#define ICC_DNxHD_VideoStreamInfo_get_StreamType(This,p)	\
    ( (This)->lpVtbl -> get_StreamType(This,p) ) 

#define ICC_DNxHD_VideoStreamInfo_get_BitRate(This,p)	\
    ( (This)->lpVtbl -> get_BitRate(This,p) ) 

#define ICC_DNxHD_VideoStreamInfo_get_FrameRate(This,p)	\
    ( (This)->lpVtbl -> get_FrameRate(This,p) ) 


#define ICC_DNxHD_VideoStreamInfo_get_FrameSize(This,s)	\
    ( (This)->lpVtbl -> get_FrameSize(This,s) ) 

#define ICC_DNxHD_VideoStreamInfo_get_AspectRatio(This,a)	\
    ( (This)->lpVtbl -> get_AspectRatio(This,a) ) 

#define ICC_DNxHD_VideoStreamInfo_get_ProgressiveSequence(This,x)	\
    ( (This)->lpVtbl -> get_ProgressiveSequence(This,x) ) 


#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_DNxHD_VideoStreamInfo_INTERFACE_DEFINED__ */


#ifndef __ICC_DNxHD_VideoEncoderSettings_INTERFACE_DEFINED__
#define __ICC_DNxHD_VideoEncoderSettings_INTERFACE_DEFINED__

/* interface ICC_DNxHD_VideoEncoderSettings */
/* [object][uuid] */ 


EXTERN_C const IID IID_ICC_DNxHD_VideoEncoderSettings;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("A2846808-3782-4E20-8727-3285447C2D73")
    ICC_DNxHD_VideoEncoderSettings : public ICC_Settings
    {
    public:
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_CompressionID( 
            /* [retval][out] */ CC_DNxHD_CID *pCID) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_CompressionID( 
            /* [in] */ CC_DNxHD_CID CID) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_FrameRate( 
            /* [retval][out] */ CC_RATIONAL *p) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_FrameRate( 
            /* [in] */ CC_RATIONAL v) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_DNxHD_VideoEncoderSettingsVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_DNxHD_VideoEncoderSettings * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_DNxHD_VideoEncoderSettings * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_DNxHD_VideoEncoderSettings * This);
        
        HRESULT ( STDMETHODCALLTYPE *Clear )( 
            ICC_DNxHD_VideoEncoderSettings * This,
            /* [in] */ LPCSTR strVarName);
        
        HRESULT ( STDMETHODCALLTYPE *Assigned )( 
            ICC_DNxHD_VideoEncoderSettings * This,
            /* [in] */ LPCSTR strVarName,
            /* [defaultvalue][retval][out] */ CC_BOOL *__MIDL__ICC_Settings0000);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_XML )( 
            ICC_DNxHD_VideoEncoderSettings * This,
            /* [retval][out] */ CC_STRING *pstrXml);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_XML )( 
            ICC_DNxHD_VideoEncoderSettings * This,
            /* [in] */ CC_STRING strXml);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_CompressionID )( 
            ICC_DNxHD_VideoEncoderSettings * This,
            /* [retval][out] */ CC_DNxHD_CID *pCID);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_CompressionID )( 
            ICC_DNxHD_VideoEncoderSettings * This,
            /* [in] */ CC_DNxHD_CID CID);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_FrameRate )( 
            ICC_DNxHD_VideoEncoderSettings * This,
            /* [retval][out] */ CC_RATIONAL *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_FrameRate )( 
            ICC_DNxHD_VideoEncoderSettings * This,
            /* [in] */ CC_RATIONAL v);
        
        END_INTERFACE
    } ICC_DNxHD_VideoEncoderSettingsVtbl;

    interface ICC_DNxHD_VideoEncoderSettings
    {
        CONST_VTBL struct ICC_DNxHD_VideoEncoderSettingsVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_DNxHD_VideoEncoderSettings_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_DNxHD_VideoEncoderSettings_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_DNxHD_VideoEncoderSettings_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ICC_DNxHD_VideoEncoderSettings_Clear(This,strVarName)	\
    ( (This)->lpVtbl -> Clear(This,strVarName) ) 

#define ICC_DNxHD_VideoEncoderSettings_Assigned(This,strVarName,__MIDL__ICC_Settings0000)	\
    ( (This)->lpVtbl -> Assigned(This,strVarName,__MIDL__ICC_Settings0000) ) 

#define ICC_DNxHD_VideoEncoderSettings_get_XML(This,pstrXml)	\
    ( (This)->lpVtbl -> get_XML(This,pstrXml) ) 

#define ICC_DNxHD_VideoEncoderSettings_put_XML(This,strXml)	\
    ( (This)->lpVtbl -> put_XML(This,strXml) ) 


#define ICC_DNxHD_VideoEncoderSettings_get_CompressionID(This,pCID)	\
    ( (This)->lpVtbl -> get_CompressionID(This,pCID) ) 

#define ICC_DNxHD_VideoEncoderSettings_put_CompressionID(This,CID)	\
    ( (This)->lpVtbl -> put_CompressionID(This,CID) ) 

#define ICC_DNxHD_VideoEncoderSettings_get_FrameRate(This,p)	\
    ( (This)->lpVtbl -> get_FrameRate(This,p) ) 

#define ICC_DNxHD_VideoEncoderSettings_put_FrameRate(This,v)	\
    ( (This)->lpVtbl -> put_FrameRate(This,v) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_DNxHD_VideoEncoderSettings_INTERFACE_DEFINED__ */


#ifndef __ICC_DNX_VideoEncoderSettings_INTERFACE_DEFINED__
#define __ICC_DNX_VideoEncoderSettings_INTERFACE_DEFINED__

/* interface ICC_DNX_VideoEncoderSettings */
/* [object][uuid] */ 


EXTERN_C const IID IID_ICC_DNX_VideoEncoderSettings;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("19FBA4DA-50DD-4321-8803-9150DB1558F1")
    ICC_DNX_VideoEncoderSettings : public ICC_Settings
    {
    public:
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_CompressionID( 
            /* [retval][out] */ CC_DNX_COMPRESSION_ID *p) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_CompressionID( 
            /* [in] */ CC_DNX_COMPRESSION_ID v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_FrameRate( 
            /* [retval][out] */ CC_RATIONAL *p) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_FrameRate( 
            /* [in] */ CC_RATIONAL v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_FrameSize( 
            /* [retval][out] */ CC_SIZE *p) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_FrameSize( 
            /* [in] */ CC_SIZE v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_ColorVolume( 
            /* [retval][out] */ CC_DNX_COLOR_VOLUME *p) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_ColorVolume( 
            /* [in] */ CC_DNX_COLOR_VOLUME v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_ChromaFormat( 
            /* [retval][out] */ CC_CHROMA_FORMAT *p) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_ChromaFormat( 
            /* [in] */ CC_CHROMA_FORMAT v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_BitDepth( 
            /* [retval][out] */ CC_UINT *p) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_BitDepth( 
            /* [in] */ CC_UINT v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_PARC( 
            /* [retval][out] */ CC_UINT *p) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_PARC( 
            /* [in] */ CC_UINT v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_PARN( 
            /* [retval][out] */ CC_UINT *p) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_PARN( 
            /* [in] */ CC_UINT v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_CRCPresence( 
            /* [retval][out] */ CC_BOOL *p) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_CRCPresence( 
            /* [in] */ CC_BOOL v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_AlphaPresence( 
            /* [retval][out] */ CC_BOOL *p) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_AlphaPresence( 
            /* [in] */ CC_BOOL v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_LosslessAlpha( 
            /* [retval][out] */ CC_BOOL *p) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_LosslessAlpha( 
            /* [in] */ CC_BOOL v) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_DNX_VideoEncoderSettingsVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_DNX_VideoEncoderSettings * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_DNX_VideoEncoderSettings * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_DNX_VideoEncoderSettings * This);
        
        HRESULT ( STDMETHODCALLTYPE *Clear )( 
            ICC_DNX_VideoEncoderSettings * This,
            /* [in] */ LPCSTR strVarName);
        
        HRESULT ( STDMETHODCALLTYPE *Assigned )( 
            ICC_DNX_VideoEncoderSettings * This,
            /* [in] */ LPCSTR strVarName,
            /* [defaultvalue][retval][out] */ CC_BOOL *__MIDL__ICC_Settings0000);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_XML )( 
            ICC_DNX_VideoEncoderSettings * This,
            /* [retval][out] */ CC_STRING *pstrXml);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_XML )( 
            ICC_DNX_VideoEncoderSettings * This,
            /* [in] */ CC_STRING strXml);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_CompressionID )( 
            ICC_DNX_VideoEncoderSettings * This,
            /* [retval][out] */ CC_DNX_COMPRESSION_ID *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_CompressionID )( 
            ICC_DNX_VideoEncoderSettings * This,
            /* [in] */ CC_DNX_COMPRESSION_ID v);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_FrameRate )( 
            ICC_DNX_VideoEncoderSettings * This,
            /* [retval][out] */ CC_RATIONAL *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_FrameRate )( 
            ICC_DNX_VideoEncoderSettings * This,
            /* [in] */ CC_RATIONAL v);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_FrameSize )( 
            ICC_DNX_VideoEncoderSettings * This,
            /* [retval][out] */ CC_SIZE *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_FrameSize )( 
            ICC_DNX_VideoEncoderSettings * This,
            /* [in] */ CC_SIZE v);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_ColorVolume )( 
            ICC_DNX_VideoEncoderSettings * This,
            /* [retval][out] */ CC_DNX_COLOR_VOLUME *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_ColorVolume )( 
            ICC_DNX_VideoEncoderSettings * This,
            /* [in] */ CC_DNX_COLOR_VOLUME v);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_ChromaFormat )( 
            ICC_DNX_VideoEncoderSettings * This,
            /* [retval][out] */ CC_CHROMA_FORMAT *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_ChromaFormat )( 
            ICC_DNX_VideoEncoderSettings * This,
            /* [in] */ CC_CHROMA_FORMAT v);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_BitDepth )( 
            ICC_DNX_VideoEncoderSettings * This,
            /* [retval][out] */ CC_UINT *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_BitDepth )( 
            ICC_DNX_VideoEncoderSettings * This,
            /* [in] */ CC_UINT v);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_PARC )( 
            ICC_DNX_VideoEncoderSettings * This,
            /* [retval][out] */ CC_UINT *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_PARC )( 
            ICC_DNX_VideoEncoderSettings * This,
            /* [in] */ CC_UINT v);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_PARN )( 
            ICC_DNX_VideoEncoderSettings * This,
            /* [retval][out] */ CC_UINT *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_PARN )( 
            ICC_DNX_VideoEncoderSettings * This,
            /* [in] */ CC_UINT v);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_CRCPresence )( 
            ICC_DNX_VideoEncoderSettings * This,
            /* [retval][out] */ CC_BOOL *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_CRCPresence )( 
            ICC_DNX_VideoEncoderSettings * This,
            /* [in] */ CC_BOOL v);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_AlphaPresence )( 
            ICC_DNX_VideoEncoderSettings * This,
            /* [retval][out] */ CC_BOOL *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_AlphaPresence )( 
            ICC_DNX_VideoEncoderSettings * This,
            /* [in] */ CC_BOOL v);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_LosslessAlpha )( 
            ICC_DNX_VideoEncoderSettings * This,
            /* [retval][out] */ CC_BOOL *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_LosslessAlpha )( 
            ICC_DNX_VideoEncoderSettings * This,
            /* [in] */ CC_BOOL v);
        
        END_INTERFACE
    } ICC_DNX_VideoEncoderSettingsVtbl;

    interface ICC_DNX_VideoEncoderSettings
    {
        CONST_VTBL struct ICC_DNX_VideoEncoderSettingsVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_DNX_VideoEncoderSettings_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_DNX_VideoEncoderSettings_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_DNX_VideoEncoderSettings_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ICC_DNX_VideoEncoderSettings_Clear(This,strVarName)	\
    ( (This)->lpVtbl -> Clear(This,strVarName) ) 

#define ICC_DNX_VideoEncoderSettings_Assigned(This,strVarName,__MIDL__ICC_Settings0000)	\
    ( (This)->lpVtbl -> Assigned(This,strVarName,__MIDL__ICC_Settings0000) ) 

#define ICC_DNX_VideoEncoderSettings_get_XML(This,pstrXml)	\
    ( (This)->lpVtbl -> get_XML(This,pstrXml) ) 

#define ICC_DNX_VideoEncoderSettings_put_XML(This,strXml)	\
    ( (This)->lpVtbl -> put_XML(This,strXml) ) 


#define ICC_DNX_VideoEncoderSettings_get_CompressionID(This,p)	\
    ( (This)->lpVtbl -> get_CompressionID(This,p) ) 

#define ICC_DNX_VideoEncoderSettings_put_CompressionID(This,v)	\
    ( (This)->lpVtbl -> put_CompressionID(This,v) ) 

#define ICC_DNX_VideoEncoderSettings_get_FrameRate(This,p)	\
    ( (This)->lpVtbl -> get_FrameRate(This,p) ) 

#define ICC_DNX_VideoEncoderSettings_put_FrameRate(This,v)	\
    ( (This)->lpVtbl -> put_FrameRate(This,v) ) 

#define ICC_DNX_VideoEncoderSettings_get_FrameSize(This,p)	\
    ( (This)->lpVtbl -> get_FrameSize(This,p) ) 

#define ICC_DNX_VideoEncoderSettings_put_FrameSize(This,v)	\
    ( (This)->lpVtbl -> put_FrameSize(This,v) ) 

#define ICC_DNX_VideoEncoderSettings_get_ColorVolume(This,p)	\
    ( (This)->lpVtbl -> get_ColorVolume(This,p) ) 

#define ICC_DNX_VideoEncoderSettings_put_ColorVolume(This,v)	\
    ( (This)->lpVtbl -> put_ColorVolume(This,v) ) 

#define ICC_DNX_VideoEncoderSettings_get_ChromaFormat(This,p)	\
    ( (This)->lpVtbl -> get_ChromaFormat(This,p) ) 

#define ICC_DNX_VideoEncoderSettings_put_ChromaFormat(This,v)	\
    ( (This)->lpVtbl -> put_ChromaFormat(This,v) ) 

#define ICC_DNX_VideoEncoderSettings_get_BitDepth(This,p)	\
    ( (This)->lpVtbl -> get_BitDepth(This,p) ) 

#define ICC_DNX_VideoEncoderSettings_put_BitDepth(This,v)	\
    ( (This)->lpVtbl -> put_BitDepth(This,v) ) 

#define ICC_DNX_VideoEncoderSettings_get_PARC(This,p)	\
    ( (This)->lpVtbl -> get_PARC(This,p) ) 

#define ICC_DNX_VideoEncoderSettings_put_PARC(This,v)	\
    ( (This)->lpVtbl -> put_PARC(This,v) ) 

#define ICC_DNX_VideoEncoderSettings_get_PARN(This,p)	\
    ( (This)->lpVtbl -> get_PARN(This,p) ) 

#define ICC_DNX_VideoEncoderSettings_put_PARN(This,v)	\
    ( (This)->lpVtbl -> put_PARN(This,v) ) 

#define ICC_DNX_VideoEncoderSettings_get_CRCPresence(This,p)	\
    ( (This)->lpVtbl -> get_CRCPresence(This,p) ) 

#define ICC_DNX_VideoEncoderSettings_put_CRCPresence(This,v)	\
    ( (This)->lpVtbl -> put_CRCPresence(This,v) ) 

#define ICC_DNX_VideoEncoderSettings_get_AlphaPresence(This,p)	\
    ( (This)->lpVtbl -> get_AlphaPresence(This,p) ) 

#define ICC_DNX_VideoEncoderSettings_put_AlphaPresence(This,v)	\
    ( (This)->lpVtbl -> put_AlphaPresence(This,v) ) 

#define ICC_DNX_VideoEncoderSettings_get_LosslessAlpha(This,p)	\
    ( (This)->lpVtbl -> get_LosslessAlpha(This,p) ) 

#define ICC_DNX_VideoEncoderSettings_put_LosslessAlpha(This,v)	\
    ( (This)->lpVtbl -> put_LosslessAlpha(This,v) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_DNX_VideoEncoderSettings_INTERFACE_DEFINED__ */


#ifndef __ICC_DNX_VideoStreamInfo_INTERFACE_DEFINED__
#define __ICC_DNX_VideoStreamInfo_INTERFACE_DEFINED__

/* interface ICC_DNX_VideoStreamInfo */
/* [object][uuid] */ 


EXTERN_C const IID IID_ICC_DNX_VideoStreamInfo;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("34B25A9C-5F78-43CF-844A-03EBC35FB099")
    ICC_DNX_VideoStreamInfo : public ICC_VideoStreamInfo
    {
    public:
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_DNX_VideoStreamInfoVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_DNX_VideoStreamInfo * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_DNX_VideoStreamInfo * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_DNX_VideoStreamInfo * This);
        
        HRESULT ( STDMETHODCALLTYPE *Clear )( 
            ICC_DNX_VideoStreamInfo * This,
            /* [in] */ LPCSTR strVarName);
        
        HRESULT ( STDMETHODCALLTYPE *Assigned )( 
            ICC_DNX_VideoStreamInfo * This,
            /* [in] */ LPCSTR strVarName,
            /* [defaultvalue][retval][out] */ CC_BOOL *__MIDL__ICC_Settings0000);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_XML )( 
            ICC_DNX_VideoStreamInfo * This,
            /* [retval][out] */ CC_STRING *pstrXml);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_XML )( 
            ICC_DNX_VideoStreamInfo * This,
            /* [in] */ CC_STRING strXml);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_StreamType )( 
            ICC_DNX_VideoStreamInfo * This,
            /* [retval][out] */ CC_ELEMENTARY_STREAM_TYPE *p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_BitRate )( 
            ICC_DNX_VideoStreamInfo * This,
            /* [retval][out] */ CC_BITRATE *p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_FrameRate )( 
            ICC_DNX_VideoStreamInfo * This,
            /* [retval][out] */ CC_FRAME_RATE *p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_FrameSize )( 
            ICC_DNX_VideoStreamInfo * This,
            /* [retval][out] */ CC_SIZE *s);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_AspectRatio )( 
            ICC_DNX_VideoStreamInfo * This,
            /* [retval][out] */ CC_RATIONAL *a);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_ProgressiveSequence )( 
            ICC_DNX_VideoStreamInfo * This,
            /* [retval][out] */ CC_BOOL *x);
        
        END_INTERFACE
    } ICC_DNX_VideoStreamInfoVtbl;

    interface ICC_DNX_VideoStreamInfo
    {
        CONST_VTBL struct ICC_DNX_VideoStreamInfoVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_DNX_VideoStreamInfo_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_DNX_VideoStreamInfo_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_DNX_VideoStreamInfo_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ICC_DNX_VideoStreamInfo_Clear(This,strVarName)	\
    ( (This)->lpVtbl -> Clear(This,strVarName) ) 

#define ICC_DNX_VideoStreamInfo_Assigned(This,strVarName,__MIDL__ICC_Settings0000)	\
    ( (This)->lpVtbl -> Assigned(This,strVarName,__MIDL__ICC_Settings0000) ) 

#define ICC_DNX_VideoStreamInfo_get_XML(This,pstrXml)	\
    ( (This)->lpVtbl -> get_XML(This,pstrXml) ) 

#define ICC_DNX_VideoStreamInfo_put_XML(This,strXml)	\
    ( (This)->lpVtbl -> put_XML(This,strXml) ) 


#define ICC_DNX_VideoStreamInfo_get_StreamType(This,p)	\
    ( (This)->lpVtbl -> get_StreamType(This,p) ) 

#define ICC_DNX_VideoStreamInfo_get_BitRate(This,p)	\
    ( (This)->lpVtbl -> get_BitRate(This,p) ) 

#define ICC_DNX_VideoStreamInfo_get_FrameRate(This,p)	\
    ( (This)->lpVtbl -> get_FrameRate(This,p) ) 


#define ICC_DNX_VideoStreamInfo_get_FrameSize(This,s)	\
    ( (This)->lpVtbl -> get_FrameSize(This,s) ) 

#define ICC_DNX_VideoStreamInfo_get_AspectRatio(This,a)	\
    ( (This)->lpVtbl -> get_AspectRatio(This,a) ) 

#define ICC_DNX_VideoStreamInfo_get_ProgressiveSequence(This,x)	\
    ( (This)->lpVtbl -> get_ProgressiveSequence(This,x) ) 


#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_DNX_VideoStreamInfo_INTERFACE_DEFINED__ */


#ifndef __ICC_DNX_VideoEncoder_INTERFACE_DEFINED__
#define __ICC_DNX_VideoEncoder_INTERFACE_DEFINED__

/* interface ICC_DNX_VideoEncoder */
/* [local][unique][uuid][object] */ 


EXTERN_C const IID IID_ICC_DNX_VideoEncoder;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("83A4BEC2-19C4-493E-A8DB-07BF62CA62F8")
    ICC_DNX_VideoEncoder : public ICC_VideoEncoder
    {
    public:
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_DNX_VideoEncoderVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_DNX_VideoEncoder * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_DNX_VideoEncoder * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_DNX_VideoEncoder * This);
        
        HRESULT ( STDMETHODCALLTYPE *Init )( 
            ICC_DNX_VideoEncoder * This,
            /* [defaultvalue][in] */ ICC_Settings *pSettings);
        
        HRESULT ( STDMETHODCALLTYPE *InitByXml )( 
            ICC_DNX_VideoEncoder * This,
            /* [in] */ CC_STRING strXML);
        
        HRESULT ( STDMETHODCALLTYPE *Done )( 
            ICC_DNX_VideoEncoder * This,
            /* [in] */ CC_BOOL bFlush,
            /* [defaultvalue][retval][out] */ CC_BOOL *pbDone);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_IsActive )( 
            ICC_DNX_VideoEncoder * This,
            /* [defaultvalue][retval][out] */ CC_BOOL *__MIDL__ICC_StreamProcessor0000);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_TimeBase )( 
            ICC_DNX_VideoEncoder * This,
            /* [retval][out] */ CC_TIMEBASE *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_TimeBase )( 
            ICC_DNX_VideoEncoder * This,
            /* [in] */ CC_TIMEBASE p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_BitRate )( 
            ICC_DNX_VideoEncoder * This,
            /* [retval][out] */ CC_BITRATE *p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_StreamInfo )( 
            ICC_DNX_VideoEncoder * This,
            /* [retval][out] */ ICC_Settings **p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_IsDataReady )( 
            ICC_DNX_VideoEncoder * This,
            /* [defaultvalue][retval][out] */ CC_BOOL *p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_DataInfo )( 
            ICC_DNX_VideoEncoder * This,
            /* [retval][out] */ ICC_Settings **s);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_OutputCallback )( 
            ICC_DNX_VideoEncoder * This,
            /* [retval][out] */ IUnknown **p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_OutputCallback )( 
            ICC_DNX_VideoEncoder * This,
            /* [in] */ IUnknown *p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_DataSize )( 
            ICC_DNX_VideoEncoder * This,
            /* [retval][out] */ CC_UINT *__MIDL__ICC_Encoder0000);
        
        HRESULT ( STDMETHODCALLTYPE *GetData )( 
            ICC_DNX_VideoEncoder * This,
            /* [size_is][out] */ CC_PBYTE pbData,
            /* [in] */ CC_UINT cbBufSize,
            /* [defaultvalue][retval][out] */ CC_UINT *pcbRetSize);
        
        HRESULT ( STDMETHODCALLTYPE *AddFrame )( 
            ICC_DNX_VideoEncoder * This,
            /* [in] */ CC_COLOR_FMT Format,
            /* [size_is][in] */ const BYTE *pData,
            /* [in] */ DWORD cbSize,
            /* [defaultvalue][in] */ INT stride,
            /* [defaultvalue][retval][out] */ CC_BOOL *pResult);
        
        HRESULT ( STDMETHODCALLTYPE *GetStride )( 
            ICC_DNX_VideoEncoder * This,
            /* [in] */ CC_COLOR_FMT fmt,
            /* [retval][out] */ DWORD *pNumBytes);
        
        HRESULT ( STDMETHODCALLTYPE *IsFormatSupported )( 
            ICC_DNX_VideoEncoder * This,
            /* [in] */ CC_COLOR_FMT fmt,
            /* [defaultvalue][retval][out] */ CC_BOOL *pResult);
        
        HRESULT ( STDMETHODCALLTYPE *AddScaleFrame )( 
            ICC_DNX_VideoEncoder * This,
            /* [size_is][in] */ const BYTE *pData,
            /* [in] */ DWORD cbSize,
            /* [in] */ CC_ADD_VIDEO_FRAME_PARAMS *pParams,
            /* [defaultvalue][retval][out] */ CC_BOOL *pResult);
        
        HRESULT ( STDMETHODCALLTYPE *IsScaleAvailable )( 
            ICC_DNX_VideoEncoder * This,
            /* [in] */ CC_ADD_VIDEO_FRAME_PARAMS *pParams,
            /* [defaultvalue][retval][out] */ CC_BOOL *__MIDL__ICC_VideoEncoder0000);
        
        HRESULT ( STDMETHODCALLTYPE *GetVideoStreamInfo )( 
            ICC_DNX_VideoEncoder * This,
            /* [retval][out] */ ICC_VideoStreamInfo **pDescr);
        
        HRESULT ( STDMETHODCALLTYPE *GetVideoFrameInfo )( 
            ICC_DNX_VideoEncoder * This,
            /* [retval][out] */ ICC_VideoFrameInfo **pDescr);
        
        END_INTERFACE
    } ICC_DNX_VideoEncoderVtbl;

    interface ICC_DNX_VideoEncoder
    {
        CONST_VTBL struct ICC_DNX_VideoEncoderVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_DNX_VideoEncoder_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_DNX_VideoEncoder_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_DNX_VideoEncoder_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ICC_DNX_VideoEncoder_Init(This,pSettings)	\
    ( (This)->lpVtbl -> Init(This,pSettings) ) 

#define ICC_DNX_VideoEncoder_InitByXml(This,strXML)	\
    ( (This)->lpVtbl -> InitByXml(This,strXML) ) 

#define ICC_DNX_VideoEncoder_Done(This,bFlush,pbDone)	\
    ( (This)->lpVtbl -> Done(This,bFlush,pbDone) ) 

#define ICC_DNX_VideoEncoder_get_IsActive(This,__MIDL__ICC_StreamProcessor0000)	\
    ( (This)->lpVtbl -> get_IsActive(This,__MIDL__ICC_StreamProcessor0000) ) 

#define ICC_DNX_VideoEncoder_get_TimeBase(This,p)	\
    ( (This)->lpVtbl -> get_TimeBase(This,p) ) 

#define ICC_DNX_VideoEncoder_put_TimeBase(This,p)	\
    ( (This)->lpVtbl -> put_TimeBase(This,p) ) 

#define ICC_DNX_VideoEncoder_get_BitRate(This,p)	\
    ( (This)->lpVtbl -> get_BitRate(This,p) ) 

#define ICC_DNX_VideoEncoder_get_StreamInfo(This,p)	\
    ( (This)->lpVtbl -> get_StreamInfo(This,p) ) 

#define ICC_DNX_VideoEncoder_get_IsDataReady(This,p)	\
    ( (This)->lpVtbl -> get_IsDataReady(This,p) ) 

#define ICC_DNX_VideoEncoder_get_DataInfo(This,s)	\
    ( (This)->lpVtbl -> get_DataInfo(This,s) ) 

#define ICC_DNX_VideoEncoder_get_OutputCallback(This,p)	\
    ( (This)->lpVtbl -> get_OutputCallback(This,p) ) 

#define ICC_DNX_VideoEncoder_put_OutputCallback(This,p)	\
    ( (This)->lpVtbl -> put_OutputCallback(This,p) ) 


#define ICC_DNX_VideoEncoder_get_DataSize(This,__MIDL__ICC_Encoder0000)	\
    ( (This)->lpVtbl -> get_DataSize(This,__MIDL__ICC_Encoder0000) ) 

#define ICC_DNX_VideoEncoder_GetData(This,pbData,cbBufSize,pcbRetSize)	\
    ( (This)->lpVtbl -> GetData(This,pbData,cbBufSize,pcbRetSize) ) 


#define ICC_DNX_VideoEncoder_AddFrame(This,Format,pData,cbSize,stride,pResult)	\
    ( (This)->lpVtbl -> AddFrame(This,Format,pData,cbSize,stride,pResult) ) 

#define ICC_DNX_VideoEncoder_GetStride(This,fmt,pNumBytes)	\
    ( (This)->lpVtbl -> GetStride(This,fmt,pNumBytes) ) 

#define ICC_DNX_VideoEncoder_IsFormatSupported(This,fmt,pResult)	\
    ( (This)->lpVtbl -> IsFormatSupported(This,fmt,pResult) ) 

#define ICC_DNX_VideoEncoder_AddScaleFrame(This,pData,cbSize,pParams,pResult)	\
    ( (This)->lpVtbl -> AddScaleFrame(This,pData,cbSize,pParams,pResult) ) 

#define ICC_DNX_VideoEncoder_IsScaleAvailable(This,pParams,__MIDL__ICC_VideoEncoder0000)	\
    ( (This)->lpVtbl -> IsScaleAvailable(This,pParams,__MIDL__ICC_VideoEncoder0000) ) 

#define ICC_DNX_VideoEncoder_GetVideoStreamInfo(This,pDescr)	\
    ( (This)->lpVtbl -> GetVideoStreamInfo(This,pDescr) ) 

#define ICC_DNX_VideoEncoder_GetVideoFrameInfo(This,pDescr)	\
    ( (This)->lpVtbl -> GetVideoFrameInfo(This,pDescr) ) 


#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_DNX_VideoEncoder_INTERFACE_DEFINED__ */



#ifndef __Cinecoder_Plugin_Codecs_DNxHD_LIBRARY_DEFINED__
#define __Cinecoder_Plugin_Codecs_DNxHD_LIBRARY_DEFINED__

/* library Cinecoder_Plugin_Codecs_DNxHD */
/* [uuid] */ 


EXTERN_C const IID LIBID_Cinecoder_Plugin_Codecs_DNxHD;

EXTERN_C const CLSID CLSID_CC_DNxHD_VideoEncoder;

#ifdef __cplusplus

class DECLSPEC_UUID("31D1E4DA-4130-497D-8336-50016F8EB652")
CC_DNxHD_VideoEncoder;
#endif

EXTERN_C const CLSID CLSID_CC_DNxHD_VideoEncoderSettings;

#ifdef __cplusplus

class DECLSPEC_UUID("A9141A3B-42BD-4B96-BDF7-F5B4781E0FA2")
CC_DNxHD_VideoEncoderSettings;
#endif

EXTERN_C const CLSID CLSID_CC_DNX_VideoEncoderSettings;

#ifdef __cplusplus

class DECLSPEC_UUID("5FFC519A-D567-4180-AC27-4C3A8A9B167D")
CC_DNX_VideoEncoderSettings;
#endif

EXTERN_C const CLSID CLSID_CC_DNX_VideoEncoder;

#ifdef __cplusplus

class DECLSPEC_UUID("0B53BDBD-5F4D-4E14-8B64-32CC1AC01861")
CC_DNX_VideoEncoder;
#endif
#endif /* __Cinecoder_Plugin_Codecs_DNxHD_LIBRARY_DEFINED__ */

/* Additional Prototypes for ALL interfaces */

/* end of Additional Prototypes */

#ifdef __cplusplus
}
#endif

#endif


