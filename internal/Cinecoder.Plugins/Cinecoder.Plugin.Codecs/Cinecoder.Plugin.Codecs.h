

/* this ALWAYS GENERATED file contains the definitions for the interfaces */


 /* File created by MIDL compiler version 8.00.0603 */
/* at Wed Jun 16 18:29:08 2021
 */
/* Compiler settings for Cinecoder.Plugin.Codecs.idl:
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

#ifndef __Cinecoder2EPlugin2ECodecs_h__
#define __Cinecoder2EPlugin2ECodecs_h__

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
#pragma once
#endif

/* Forward Declarations */ 

#ifndef __ICC_DV_VideoEncoder_FWD_DEFINED__
#define __ICC_DV_VideoEncoder_FWD_DEFINED__
typedef interface ICC_DV_VideoEncoder ICC_DV_VideoEncoder;

#endif 	/* __ICC_DV_VideoEncoder_FWD_DEFINED__ */


#ifndef __ICC_DV_VideoStreamInfo_FWD_DEFINED__
#define __ICC_DV_VideoStreamInfo_FWD_DEFINED__
typedef interface ICC_DV_VideoStreamInfo ICC_DV_VideoStreamInfo;

#endif 	/* __ICC_DV_VideoStreamInfo_FWD_DEFINED__ */


#ifndef __ICC_DV_VideoEncoderSettings_FWD_DEFINED__
#define __ICC_DV_VideoEncoderSettings_FWD_DEFINED__
typedef interface ICC_DV_VideoEncoderSettings ICC_DV_VideoEncoderSettings;

#endif 	/* __ICC_DV_VideoEncoderSettings_FWD_DEFINED__ */


#ifndef __ICC_DV_VideoDecoder_FWD_DEFINED__
#define __ICC_DV_VideoDecoder_FWD_DEFINED__
typedef interface ICC_DV_VideoDecoder ICC_DV_VideoDecoder;

#endif 	/* __ICC_DV_VideoDecoder_FWD_DEFINED__ */


#ifndef __ICC_DV_VideoDecoderSettings_FWD_DEFINED__
#define __ICC_DV_VideoDecoderSettings_FWD_DEFINED__
typedef interface ICC_DV_VideoDecoderSettings ICC_DV_VideoDecoderSettings;

#endif 	/* __ICC_DV_VideoDecoderSettings_FWD_DEFINED__ */


#ifndef __ICC_ProRes_VideoStreamInfo_FWD_DEFINED__
#define __ICC_ProRes_VideoStreamInfo_FWD_DEFINED__
typedef interface ICC_ProRes_VideoStreamInfo ICC_ProRes_VideoStreamInfo;

#endif 	/* __ICC_ProRes_VideoStreamInfo_FWD_DEFINED__ */


#ifndef __ICC_ProRes_VideoFrameInfo_FWD_DEFINED__
#define __ICC_ProRes_VideoFrameInfo_FWD_DEFINED__
typedef interface ICC_ProRes_VideoFrameInfo ICC_ProRes_VideoFrameInfo;

#endif 	/* __ICC_ProRes_VideoFrameInfo_FWD_DEFINED__ */


#ifndef __ICC_ProRes_VideoDecoderSettings_FWD_DEFINED__
#define __ICC_ProRes_VideoDecoderSettings_FWD_DEFINED__
typedef interface ICC_ProRes_VideoDecoderSettings ICC_ProRes_VideoDecoderSettings;

#endif 	/* __ICC_ProRes_VideoDecoderSettings_FWD_DEFINED__ */


#ifndef __ICC_ProRes_VideoDecoder_FWD_DEFINED__
#define __ICC_ProRes_VideoDecoder_FWD_DEFINED__
typedef interface ICC_ProRes_VideoDecoder ICC_ProRes_VideoDecoder;

#endif 	/* __ICC_ProRes_VideoDecoder_FWD_DEFINED__ */


#ifndef __ICC_ProRes_VideoEncoderSettings_FWD_DEFINED__
#define __ICC_ProRes_VideoEncoderSettings_FWD_DEFINED__
typedef interface ICC_ProRes_VideoEncoderSettings ICC_ProRes_VideoEncoderSettings;

#endif 	/* __ICC_ProRes_VideoEncoderSettings_FWD_DEFINED__ */


#ifndef __ICC_ProRes_VideoEncoder_FWD_DEFINED__
#define __ICC_ProRes_VideoEncoder_FWD_DEFINED__
typedef interface ICC_ProRes_VideoEncoder ICC_ProRes_VideoEncoder;

#endif 	/* __ICC_ProRes_VideoEncoder_FWD_DEFINED__ */


#ifndef __ICC_PCM_AudioMapperInputPinSettings_FWD_DEFINED__
#define __ICC_PCM_AudioMapperInputPinSettings_FWD_DEFINED__
typedef interface ICC_PCM_AudioMapperInputPinSettings ICC_PCM_AudioMapperInputPinSettings;

#endif 	/* __ICC_PCM_AudioMapperInputPinSettings_FWD_DEFINED__ */


#ifndef __ICC_PCM_AudioMapperSettings_FWD_DEFINED__
#define __ICC_PCM_AudioMapperSettings_FWD_DEFINED__
typedef interface ICC_PCM_AudioMapperSettings ICC_PCM_AudioMapperSettings;

#endif 	/* __ICC_PCM_AudioMapperSettings_FWD_DEFINED__ */


#ifndef __ICC_PCM_AudioMapperLinkSettings_FWD_DEFINED__
#define __ICC_PCM_AudioMapperLinkSettings_FWD_DEFINED__
typedef interface ICC_PCM_AudioMapperLinkSettings ICC_PCM_AudioMapperLinkSettings;

#endif 	/* __ICC_PCM_AudioMapperLinkSettings_FWD_DEFINED__ */


#ifndef __ICC_PCM_AudioMapperOutputStreamSettings_FWD_DEFINED__
#define __ICC_PCM_AudioMapperOutputStreamSettings_FWD_DEFINED__
typedef interface ICC_PCM_AudioMapperOutputStreamSettings ICC_PCM_AudioMapperOutputStreamSettings;

#endif 	/* __ICC_PCM_AudioMapperOutputStreamSettings_FWD_DEFINED__ */


#ifndef __ICC_PCM_AudioMapperOutputStreamInfo_FWD_DEFINED__
#define __ICC_PCM_AudioMapperOutputStreamInfo_FWD_DEFINED__
typedef interface ICC_PCM_AudioMapperOutputStreamInfo ICC_PCM_AudioMapperOutputStreamInfo;

#endif 	/* __ICC_PCM_AudioMapperOutputStreamInfo_FWD_DEFINED__ */


#ifndef __ICC_PCM_AudioMapperOutputPin_FWD_DEFINED__
#define __ICC_PCM_AudioMapperOutputPin_FWD_DEFINED__
typedef interface ICC_PCM_AudioMapperOutputPin ICC_PCM_AudioMapperOutputPin;

#endif 	/* __ICC_PCM_AudioMapperOutputPin_FWD_DEFINED__ */


#ifndef __ICC_PCM_AudioMapper_FWD_DEFINED__
#define __ICC_PCM_AudioMapper_FWD_DEFINED__
typedef interface ICC_PCM_AudioMapper ICC_PCM_AudioMapper;

#endif 	/* __ICC_PCM_AudioMapper_FWD_DEFINED__ */


#ifndef __ICC_PCM_AudioMapperProducer_FWD_DEFINED__
#define __ICC_PCM_AudioMapperProducer_FWD_DEFINED__
typedef interface ICC_PCM_AudioMapperProducer ICC_PCM_AudioMapperProducer;

#endif 	/* __ICC_PCM_AudioMapperProducer_FWD_DEFINED__ */


#ifndef __ICC_Audio_Resampler_FWD_DEFINED__
#define __ICC_Audio_Resampler_FWD_DEFINED__
typedef interface ICC_Audio_Resampler ICC_Audio_Resampler;

#endif 	/* __ICC_Audio_Resampler_FWD_DEFINED__ */


#ifndef __ICC_Audio_Resampler_Settings_FWD_DEFINED__
#define __ICC_Audio_Resampler_Settings_FWD_DEFINED__
typedef interface ICC_Audio_Resampler_Settings ICC_Audio_Resampler_Settings;

#endif 	/* __ICC_Audio_Resampler_Settings_FWD_DEFINED__ */


#ifndef __ICC_Audio_Resampler_OutputPin_FWD_DEFINED__
#define __ICC_Audio_Resampler_OutputPin_FWD_DEFINED__
typedef interface ICC_Audio_Resampler_OutputPin ICC_Audio_Resampler_OutputPin;

#endif 	/* __ICC_Audio_Resampler_OutputPin_FWD_DEFINED__ */


#ifndef __ICC_Audio_Resampler_OutputPin_Internal_FWD_DEFINED__
#define __ICC_Audio_Resampler_OutputPin_Internal_FWD_DEFINED__
typedef interface ICC_Audio_Resampler_OutputPin_Internal ICC_Audio_Resampler_OutputPin_Internal;

#endif 	/* __ICC_Audio_Resampler_OutputPin_Internal_FWD_DEFINED__ */


#ifndef __ICC_Audio_Resampler_OutputPin_Producer_FWD_DEFINED__
#define __ICC_Audio_Resampler_OutputPin_Producer_FWD_DEFINED__
typedef interface ICC_Audio_Resampler_OutputPin_Producer ICC_Audio_Resampler_OutputPin_Producer;

#endif 	/* __ICC_Audio_Resampler_OutputPin_Producer_FWD_DEFINED__ */


#ifndef __ICC_Audio_Resampler_InputPin_Settings_FWD_DEFINED__
#define __ICC_Audio_Resampler_InputPin_Settings_FWD_DEFINED__
typedef interface ICC_Audio_Resampler_InputPin_Settings ICC_Audio_Resampler_InputPin_Settings;

#endif 	/* __ICC_Audio_Resampler_InputPin_Settings_FWD_DEFINED__ */


#ifndef __ICC_Audio_Resampler_InputPin_FWD_DEFINED__
#define __ICC_Audio_Resampler_InputPin_FWD_DEFINED__
typedef interface ICC_Audio_Resampler_InputPin ICC_Audio_Resampler_InputPin;

#endif 	/* __ICC_Audio_Resampler_InputPin_FWD_DEFINED__ */


#ifndef __ICC_Audio_Resampler_InputPin_Internal_FWD_DEFINED__
#define __ICC_Audio_Resampler_InputPin_Internal_FWD_DEFINED__
typedef interface ICC_Audio_Resampler_InputPin_Internal ICC_Audio_Resampler_InputPin_Internal;

#endif 	/* __ICC_Audio_Resampler_InputPin_Internal_FWD_DEFINED__ */


#ifndef __ICC_PCM_AudioMapperOutputPin_FWD_DEFINED__
#define __ICC_PCM_AudioMapperOutputPin_FWD_DEFINED__
typedef interface ICC_PCM_AudioMapperOutputPin ICC_PCM_AudioMapperOutputPin;

#endif 	/* __ICC_PCM_AudioMapperOutputPin_FWD_DEFINED__ */


#ifndef __ICC_PCM_AudioMapperProducer_FWD_DEFINED__
#define __ICC_PCM_AudioMapperProducer_FWD_DEFINED__
typedef interface ICC_PCM_AudioMapperProducer ICC_PCM_AudioMapperProducer;

#endif 	/* __ICC_PCM_AudioMapperProducer_FWD_DEFINED__ */


#ifndef __ICC_Audio_Resampler_InputPin_FWD_DEFINED__
#define __ICC_Audio_Resampler_InputPin_FWD_DEFINED__
typedef interface ICC_Audio_Resampler_InputPin ICC_Audio_Resampler_InputPin;

#endif 	/* __ICC_Audio_Resampler_InputPin_FWD_DEFINED__ */


#ifndef __ICC_Audio_Resampler_OutputPin_FWD_DEFINED__
#define __ICC_Audio_Resampler_OutputPin_FWD_DEFINED__
typedef interface ICC_Audio_Resampler_OutputPin ICC_Audio_Resampler_OutputPin;

#endif 	/* __ICC_Audio_Resampler_OutputPin_FWD_DEFINED__ */


#ifndef __ICC_Audio_Resampler_OutputPin_Producer_FWD_DEFINED__
#define __ICC_Audio_Resampler_OutputPin_Producer_FWD_DEFINED__
typedef interface ICC_Audio_Resampler_OutputPin_Producer ICC_Audio_Resampler_OutputPin_Producer;

#endif 	/* __ICC_Audio_Resampler_OutputPin_Producer_FWD_DEFINED__ */


#ifndef __CC_DV_VideoEncoder_FWD_DEFINED__
#define __CC_DV_VideoEncoder_FWD_DEFINED__

#ifdef __cplusplus
typedef class CC_DV_VideoEncoder CC_DV_VideoEncoder;
#else
typedef struct CC_DV_VideoEncoder CC_DV_VideoEncoder;
#endif /* __cplusplus */

#endif 	/* __CC_DV_VideoEncoder_FWD_DEFINED__ */


#ifndef __CC_DV_VideoEncoderSettings_FWD_DEFINED__
#define __CC_DV_VideoEncoderSettings_FWD_DEFINED__

#ifdef __cplusplus
typedef class CC_DV_VideoEncoderSettings CC_DV_VideoEncoderSettings;
#else
typedef struct CC_DV_VideoEncoderSettings CC_DV_VideoEncoderSettings;
#endif /* __cplusplus */

#endif 	/* __CC_DV_VideoEncoderSettings_FWD_DEFINED__ */


#ifndef __CC_DV_VideoDecoder_FWD_DEFINED__
#define __CC_DV_VideoDecoder_FWD_DEFINED__

#ifdef __cplusplus
typedef class CC_DV_VideoDecoder CC_DV_VideoDecoder;
#else
typedef struct CC_DV_VideoDecoder CC_DV_VideoDecoder;
#endif /* __cplusplus */

#endif 	/* __CC_DV_VideoDecoder_FWD_DEFINED__ */


#ifndef __CC_DV_VideoDecoderSettings_FWD_DEFINED__
#define __CC_DV_VideoDecoderSettings_FWD_DEFINED__

#ifdef __cplusplus
typedef class CC_DV_VideoDecoderSettings CC_DV_VideoDecoderSettings;
#else
typedef struct CC_DV_VideoDecoderSettings CC_DV_VideoDecoderSettings;
#endif /* __cplusplus */

#endif 	/* __CC_DV_VideoDecoderSettings_FWD_DEFINED__ */


#ifndef __CC_ProRes_VideoDecoder_FWD_DEFINED__
#define __CC_ProRes_VideoDecoder_FWD_DEFINED__

#ifdef __cplusplus
typedef class CC_ProRes_VideoDecoder CC_ProRes_VideoDecoder;
#else
typedef struct CC_ProRes_VideoDecoder CC_ProRes_VideoDecoder;
#endif /* __cplusplus */

#endif 	/* __CC_ProRes_VideoDecoder_FWD_DEFINED__ */


#ifndef __CC_ProRes_VideoDecoderSettings_FWD_DEFINED__
#define __CC_ProRes_VideoDecoderSettings_FWD_DEFINED__

#ifdef __cplusplus
typedef class CC_ProRes_VideoDecoderSettings CC_ProRes_VideoDecoderSettings;
#else
typedef struct CC_ProRes_VideoDecoderSettings CC_ProRes_VideoDecoderSettings;
#endif /* __cplusplus */

#endif 	/* __CC_ProRes_VideoDecoderSettings_FWD_DEFINED__ */


#ifndef __CC_ProRes_VideoEncoder_FWD_DEFINED__
#define __CC_ProRes_VideoEncoder_FWD_DEFINED__

#ifdef __cplusplus
typedef class CC_ProRes_VideoEncoder CC_ProRes_VideoEncoder;
#else
typedef struct CC_ProRes_VideoEncoder CC_ProRes_VideoEncoder;
#endif /* __cplusplus */

#endif 	/* __CC_ProRes_VideoEncoder_FWD_DEFINED__ */


#ifndef __CC_ProRes_VideoEncoderSettings_FWD_DEFINED__
#define __CC_ProRes_VideoEncoderSettings_FWD_DEFINED__

#ifdef __cplusplus
typedef class CC_ProRes_VideoEncoderSettings CC_ProRes_VideoEncoderSettings;
#else
typedef struct CC_ProRes_VideoEncoderSettings CC_ProRes_VideoEncoderSettings;
#endif /* __cplusplus */

#endif 	/* __CC_ProRes_VideoEncoderSettings_FWD_DEFINED__ */


#ifndef __CC_PCM_AudioMapperInputPinSettings_FWD_DEFINED__
#define __CC_PCM_AudioMapperInputPinSettings_FWD_DEFINED__

#ifdef __cplusplus
typedef class CC_PCM_AudioMapperInputPinSettings CC_PCM_AudioMapperInputPinSettings;
#else
typedef struct CC_PCM_AudioMapperInputPinSettings CC_PCM_AudioMapperInputPinSettings;
#endif /* __cplusplus */

#endif 	/* __CC_PCM_AudioMapperInputPinSettings_FWD_DEFINED__ */


#ifndef __CC_PCM_AudioMapperSettings_FWD_DEFINED__
#define __CC_PCM_AudioMapperSettings_FWD_DEFINED__

#ifdef __cplusplus
typedef class CC_PCM_AudioMapperSettings CC_PCM_AudioMapperSettings;
#else
typedef struct CC_PCM_AudioMapperSettings CC_PCM_AudioMapperSettings;
#endif /* __cplusplus */

#endif 	/* __CC_PCM_AudioMapperSettings_FWD_DEFINED__ */


#ifndef __CC_PCM_AudioMapperLinkSettings_FWD_DEFINED__
#define __CC_PCM_AudioMapperLinkSettings_FWD_DEFINED__

#ifdef __cplusplus
typedef class CC_PCM_AudioMapperLinkSettings CC_PCM_AudioMapperLinkSettings;
#else
typedef struct CC_PCM_AudioMapperLinkSettings CC_PCM_AudioMapperLinkSettings;
#endif /* __cplusplus */

#endif 	/* __CC_PCM_AudioMapperLinkSettings_FWD_DEFINED__ */


#ifndef __CC_PCM_AudioMapperOutputStreamSettings_FWD_DEFINED__
#define __CC_PCM_AudioMapperOutputStreamSettings_FWD_DEFINED__

#ifdef __cplusplus
typedef class CC_PCM_AudioMapperOutputStreamSettings CC_PCM_AudioMapperOutputStreamSettings;
#else
typedef struct CC_PCM_AudioMapperOutputStreamSettings CC_PCM_AudioMapperOutputStreamSettings;
#endif /* __cplusplus */

#endif 	/* __CC_PCM_AudioMapperOutputStreamSettings_FWD_DEFINED__ */


#ifndef __CC_PCM_AudioMapper_FWD_DEFINED__
#define __CC_PCM_AudioMapper_FWD_DEFINED__

#ifdef __cplusplus
typedef class CC_PCM_AudioMapper CC_PCM_AudioMapper;
#else
typedef struct CC_PCM_AudioMapper CC_PCM_AudioMapper;
#endif /* __cplusplus */

#endif 	/* __CC_PCM_AudioMapper_FWD_DEFINED__ */


#ifndef __CC_Audio_Resampler_FWD_DEFINED__
#define __CC_Audio_Resampler_FWD_DEFINED__

#ifdef __cplusplus
typedef class CC_Audio_Resampler CC_Audio_Resampler;
#else
typedef struct CC_Audio_Resampler CC_Audio_Resampler;
#endif /* __cplusplus */

#endif 	/* __CC_Audio_Resampler_FWD_DEFINED__ */


#ifndef __CC_Audio_Resampler_Settings_FWD_DEFINED__
#define __CC_Audio_Resampler_Settings_FWD_DEFINED__

#ifdef __cplusplus
typedef class CC_Audio_Resampler_Settings CC_Audio_Resampler_Settings;
#else
typedef struct CC_Audio_Resampler_Settings CC_Audio_Resampler_Settings;
#endif /* __cplusplus */

#endif 	/* __CC_Audio_Resampler_Settings_FWD_DEFINED__ */


#ifndef __CC_Audio_Resampler_InputPin_Settings_FWD_DEFINED__
#define __CC_Audio_Resampler_InputPin_Settings_FWD_DEFINED__

#ifdef __cplusplus
typedef class CC_Audio_Resampler_InputPin_Settings CC_Audio_Resampler_InputPin_Settings;
#else
typedef struct CC_Audio_Resampler_InputPin_Settings CC_Audio_Resampler_InputPin_Settings;
#endif /* __cplusplus */

#endif 	/* __CC_Audio_Resampler_InputPin_Settings_FWD_DEFINED__ */


/* header files for imported files */
#include "oaidl.h"
#include "ocidl.h"

#ifdef __cplusplus
extern "C"{
#endif 


/* interface __MIDL_itf_Cinecoder2EPlugin2ECodecs_0000_0000 */
/* [local] */ 

typedef /* [v1_enum] */ 
enum CC_DV_TYPE
    {
        CC_DV_TYPE_UNKNOWN	= 0,
        CC_DV_TYPE_SD_525	= ( CC_DV_TYPE_UNKNOWN + 1 ) ,
        CC_DV_TYPE_SD_625	= ( CC_DV_TYPE_SD_525 + 1 ) ,
        CC_DV_TYPE_25_525	= ( CC_DV_TYPE_SD_625 + 1 ) ,
        CC_DV_TYPE_25_625	= ( CC_DV_TYPE_25_525 + 1 ) ,
        CC_DV_TYPE_50_525	= ( CC_DV_TYPE_25_625 + 1 ) ,
        CC_DV_TYPE_50_625	= ( CC_DV_TYPE_50_525 + 1 ) ,
        CC_DV_TYPE_100_720_50p	= ( CC_DV_TYPE_50_625 + 1 ) ,
        CC_DV_TYPE_100_720_60p	= ( CC_DV_TYPE_100_720_50p + 1 ) ,
        CC_DV_TYPE_100_1080_50i	= ( CC_DV_TYPE_100_720_60p + 1 ) ,
        CC_DV_TYPE_100_1080_60i	= ( CC_DV_TYPE_100_1080_50i + 1 ) 
    } 	CC_DV_TYPE;



extern RPC_IF_HANDLE __MIDL_itf_Cinecoder2EPlugin2ECodecs_0000_0000_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_Cinecoder2EPlugin2ECodecs_0000_0000_v0_0_s_ifspec;

#ifndef __ICC_DV_VideoEncoder_INTERFACE_DEFINED__
#define __ICC_DV_VideoEncoder_INTERFACE_DEFINED__

/* interface ICC_DV_VideoEncoder */
/* [local][unique][uuid][object] */ 


EXTERN_C const IID IID_ICC_DV_VideoEncoder;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("98A11893-37A9-46E4-A6B8-4D7C24A7CCFD")
    ICC_DV_VideoEncoder : public ICC_VideoEncoder
    {
    public:
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_DV_VideoEncoderVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_DV_VideoEncoder * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_DV_VideoEncoder * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_DV_VideoEncoder * This);
        
        HRESULT ( STDMETHODCALLTYPE *Init )( 
            ICC_DV_VideoEncoder * This,
            /* [defaultvalue][in] */ ICC_Settings *pSettings);
        
        HRESULT ( STDMETHODCALLTYPE *InitByXml )( 
            ICC_DV_VideoEncoder * This,
            /* [in] */ CC_STRING strXML);
        
        HRESULT ( STDMETHODCALLTYPE *Done )( 
            ICC_DV_VideoEncoder * This,
            /* [in] */ CC_BOOL bFlush,
            /* [defaultvalue][retval][out] */ CC_BOOL *pbDone);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_IsActive )( 
            ICC_DV_VideoEncoder * This,
            /* [defaultvalue][retval][out] */ CC_BOOL *__MIDL__ICC_StreamProcessor0000);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_TimeBase )( 
            ICC_DV_VideoEncoder * This,
            /* [retval][out] */ CC_TIMEBASE *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_TimeBase )( 
            ICC_DV_VideoEncoder * This,
            /* [in] */ CC_TIMEBASE p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_BitRate )( 
            ICC_DV_VideoEncoder * This,
            /* [retval][out] */ CC_BITRATE *p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_StreamInfo )( 
            ICC_DV_VideoEncoder * This,
            /* [retval][out] */ ICC_Settings **p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_IsDataReady )( 
            ICC_DV_VideoEncoder * This,
            /* [defaultvalue][retval][out] */ CC_BOOL *p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_DataInfo )( 
            ICC_DV_VideoEncoder * This,
            /* [retval][out] */ ICC_Settings **s);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_OutputCallback )( 
            ICC_DV_VideoEncoder * This,
            /* [retval][out] */ IUnknown **p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_OutputCallback )( 
            ICC_DV_VideoEncoder * This,
            /* [in] */ IUnknown *p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_DataSize )( 
            ICC_DV_VideoEncoder * This,
            /* [retval][out] */ CC_UINT *__MIDL__ICC_Encoder0000);
        
        HRESULT ( STDMETHODCALLTYPE *GetData )( 
            ICC_DV_VideoEncoder * This,
            /* [size_is][out] */ CC_PBYTE pbData,
            /* [in] */ CC_UINT cbBufSize,
            /* [defaultvalue][retval][out] */ CC_UINT *pcbRetSize);
        
        HRESULT ( STDMETHODCALLTYPE *AddFrame )( 
            ICC_DV_VideoEncoder * This,
            /* [in] */ CC_COLOR_FMT Format,
            /* [size_is][in] */ const BYTE *pData,
            /* [in] */ DWORD cbSize,
            /* [defaultvalue][in] */ INT stride,
            /* [defaultvalue][retval][out] */ CC_BOOL *pResult);
        
        HRESULT ( STDMETHODCALLTYPE *GetStride )( 
            ICC_DV_VideoEncoder * This,
            /* [in] */ CC_COLOR_FMT fmt,
            /* [retval][out] */ DWORD *pNumBytes);
        
        HRESULT ( STDMETHODCALLTYPE *IsFormatSupported )( 
            ICC_DV_VideoEncoder * This,
            /* [in] */ CC_COLOR_FMT fmt,
            /* [defaultvalue][retval][out] */ CC_BOOL *pResult);
        
        HRESULT ( STDMETHODCALLTYPE *AddScaleFrame )( 
            ICC_DV_VideoEncoder * This,
            /* [size_is][in] */ const BYTE *pData,
            /* [in] */ DWORD cbSize,
            /* [in] */ CC_ADD_VIDEO_FRAME_PARAMS *pParams,
            /* [defaultvalue][retval][out] */ CC_BOOL *pResult);
        
        HRESULT ( STDMETHODCALLTYPE *IsScaleAvailable )( 
            ICC_DV_VideoEncoder * This,
            /* [in] */ CC_ADD_VIDEO_FRAME_PARAMS *pParams,
            /* [defaultvalue][retval][out] */ CC_BOOL *__MIDL__ICC_VideoEncoder0000);
        
        HRESULT ( STDMETHODCALLTYPE *GetVideoStreamInfo )( 
            ICC_DV_VideoEncoder * This,
            /* [retval][out] */ ICC_VideoStreamInfo **pDescr);
        
        HRESULT ( STDMETHODCALLTYPE *GetVideoFrameInfo )( 
            ICC_DV_VideoEncoder * This,
            /* [retval][out] */ ICC_VideoFrameInfo **pDescr);
        
        END_INTERFACE
    } ICC_DV_VideoEncoderVtbl;

    interface ICC_DV_VideoEncoder
    {
        CONST_VTBL struct ICC_DV_VideoEncoderVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_DV_VideoEncoder_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_DV_VideoEncoder_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_DV_VideoEncoder_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ICC_DV_VideoEncoder_Init(This,pSettings)	\
    ( (This)->lpVtbl -> Init(This,pSettings) ) 

#define ICC_DV_VideoEncoder_InitByXml(This,strXML)	\
    ( (This)->lpVtbl -> InitByXml(This,strXML) ) 

#define ICC_DV_VideoEncoder_Done(This,bFlush,pbDone)	\
    ( (This)->lpVtbl -> Done(This,bFlush,pbDone) ) 

#define ICC_DV_VideoEncoder_get_IsActive(This,__MIDL__ICC_StreamProcessor0000)	\
    ( (This)->lpVtbl -> get_IsActive(This,__MIDL__ICC_StreamProcessor0000) ) 

#define ICC_DV_VideoEncoder_get_TimeBase(This,p)	\
    ( (This)->lpVtbl -> get_TimeBase(This,p) ) 

#define ICC_DV_VideoEncoder_put_TimeBase(This,p)	\
    ( (This)->lpVtbl -> put_TimeBase(This,p) ) 

#define ICC_DV_VideoEncoder_get_BitRate(This,p)	\
    ( (This)->lpVtbl -> get_BitRate(This,p) ) 

#define ICC_DV_VideoEncoder_get_StreamInfo(This,p)	\
    ( (This)->lpVtbl -> get_StreamInfo(This,p) ) 

#define ICC_DV_VideoEncoder_get_IsDataReady(This,p)	\
    ( (This)->lpVtbl -> get_IsDataReady(This,p) ) 

#define ICC_DV_VideoEncoder_get_DataInfo(This,s)	\
    ( (This)->lpVtbl -> get_DataInfo(This,s) ) 

#define ICC_DV_VideoEncoder_get_OutputCallback(This,p)	\
    ( (This)->lpVtbl -> get_OutputCallback(This,p) ) 

#define ICC_DV_VideoEncoder_put_OutputCallback(This,p)	\
    ( (This)->lpVtbl -> put_OutputCallback(This,p) ) 


#define ICC_DV_VideoEncoder_get_DataSize(This,__MIDL__ICC_Encoder0000)	\
    ( (This)->lpVtbl -> get_DataSize(This,__MIDL__ICC_Encoder0000) ) 

#define ICC_DV_VideoEncoder_GetData(This,pbData,cbBufSize,pcbRetSize)	\
    ( (This)->lpVtbl -> GetData(This,pbData,cbBufSize,pcbRetSize) ) 


#define ICC_DV_VideoEncoder_AddFrame(This,Format,pData,cbSize,stride,pResult)	\
    ( (This)->lpVtbl -> AddFrame(This,Format,pData,cbSize,stride,pResult) ) 

#define ICC_DV_VideoEncoder_GetStride(This,fmt,pNumBytes)	\
    ( (This)->lpVtbl -> GetStride(This,fmt,pNumBytes) ) 

#define ICC_DV_VideoEncoder_IsFormatSupported(This,fmt,pResult)	\
    ( (This)->lpVtbl -> IsFormatSupported(This,fmt,pResult) ) 

#define ICC_DV_VideoEncoder_AddScaleFrame(This,pData,cbSize,pParams,pResult)	\
    ( (This)->lpVtbl -> AddScaleFrame(This,pData,cbSize,pParams,pResult) ) 

#define ICC_DV_VideoEncoder_IsScaleAvailable(This,pParams,__MIDL__ICC_VideoEncoder0000)	\
    ( (This)->lpVtbl -> IsScaleAvailable(This,pParams,__MIDL__ICC_VideoEncoder0000) ) 

#define ICC_DV_VideoEncoder_GetVideoStreamInfo(This,pDescr)	\
    ( (This)->lpVtbl -> GetVideoStreamInfo(This,pDescr) ) 

#define ICC_DV_VideoEncoder_GetVideoFrameInfo(This,pDescr)	\
    ( (This)->lpVtbl -> GetVideoFrameInfo(This,pDescr) ) 


#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_DV_VideoEncoder_INTERFACE_DEFINED__ */


#ifndef __ICC_DV_VideoStreamInfo_INTERFACE_DEFINED__
#define __ICC_DV_VideoStreamInfo_INTERFACE_DEFINED__

/* interface ICC_DV_VideoStreamInfo */
/* [object][uuid] */ 


EXTERN_C const IID IID_ICC_DV_VideoStreamInfo;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("BE590E74-F1AA-424A-9A35-183FD5A5B799")
    ICC_DV_VideoStreamInfo : public ICC_VideoStreamInfo
    {
    public:
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_DV_VideoStreamInfoVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_DV_VideoStreamInfo * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_DV_VideoStreamInfo * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_DV_VideoStreamInfo * This);
        
        HRESULT ( STDMETHODCALLTYPE *Clear )( 
            ICC_DV_VideoStreamInfo * This,
            /* [in] */ LPCSTR strVarName);
        
        HRESULT ( STDMETHODCALLTYPE *Assigned )( 
            ICC_DV_VideoStreamInfo * This,
            /* [in] */ LPCSTR strVarName,
            /* [defaultvalue][retval][out] */ CC_BOOL *__MIDL__ICC_Settings0000);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_XML )( 
            ICC_DV_VideoStreamInfo * This,
            /* [retval][out] */ CC_STRING *pstrXml);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_XML )( 
            ICC_DV_VideoStreamInfo * This,
            /* [in] */ CC_STRING strXml);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_StreamType )( 
            ICC_DV_VideoStreamInfo * This,
            /* [retval][out] */ CC_ELEMENTARY_STREAM_TYPE *p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_BitRate )( 
            ICC_DV_VideoStreamInfo * This,
            /* [retval][out] */ CC_BITRATE *p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_FrameRate )( 
            ICC_DV_VideoStreamInfo * This,
            /* [retval][out] */ CC_FRAME_RATE *p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_FrameSize )( 
            ICC_DV_VideoStreamInfo * This,
            /* [retval][out] */ CC_SIZE *s);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_AspectRatio )( 
            ICC_DV_VideoStreamInfo * This,
            /* [retval][out] */ CC_RATIONAL *a);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_ProgressiveSequence )( 
            ICC_DV_VideoStreamInfo * This,
            /* [retval][out] */ CC_BOOL *x);
        
        END_INTERFACE
    } ICC_DV_VideoStreamInfoVtbl;

    interface ICC_DV_VideoStreamInfo
    {
        CONST_VTBL struct ICC_DV_VideoStreamInfoVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_DV_VideoStreamInfo_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_DV_VideoStreamInfo_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_DV_VideoStreamInfo_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ICC_DV_VideoStreamInfo_Clear(This,strVarName)	\
    ( (This)->lpVtbl -> Clear(This,strVarName) ) 

#define ICC_DV_VideoStreamInfo_Assigned(This,strVarName,__MIDL__ICC_Settings0000)	\
    ( (This)->lpVtbl -> Assigned(This,strVarName,__MIDL__ICC_Settings0000) ) 

#define ICC_DV_VideoStreamInfo_get_XML(This,pstrXml)	\
    ( (This)->lpVtbl -> get_XML(This,pstrXml) ) 

#define ICC_DV_VideoStreamInfo_put_XML(This,strXml)	\
    ( (This)->lpVtbl -> put_XML(This,strXml) ) 


#define ICC_DV_VideoStreamInfo_get_StreamType(This,p)	\
    ( (This)->lpVtbl -> get_StreamType(This,p) ) 

#define ICC_DV_VideoStreamInfo_get_BitRate(This,p)	\
    ( (This)->lpVtbl -> get_BitRate(This,p) ) 

#define ICC_DV_VideoStreamInfo_get_FrameRate(This,p)	\
    ( (This)->lpVtbl -> get_FrameRate(This,p) ) 


#define ICC_DV_VideoStreamInfo_get_FrameSize(This,s)	\
    ( (This)->lpVtbl -> get_FrameSize(This,s) ) 

#define ICC_DV_VideoStreamInfo_get_AspectRatio(This,a)	\
    ( (This)->lpVtbl -> get_AspectRatio(This,a) ) 

#define ICC_DV_VideoStreamInfo_get_ProgressiveSequence(This,x)	\
    ( (This)->lpVtbl -> get_ProgressiveSequence(This,x) ) 


#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_DV_VideoStreamInfo_INTERFACE_DEFINED__ */


#ifndef __ICC_DV_VideoEncoderSettings_INTERFACE_DEFINED__
#define __ICC_DV_VideoEncoderSettings_INTERFACE_DEFINED__

/* interface ICC_DV_VideoEncoderSettings */
/* [object][uuid] */ 


EXTERN_C const IID IID_ICC_DV_VideoEncoderSettings;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("8730E0B2-E8FE-48C4-B9DE-59EEE7E2C958")
    ICC_DV_VideoEncoderSettings : public ICC_Settings
    {
    public:
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_Type( 
            /* [retval][out] */ CC_DV_TYPE *p) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_Type( 
            /* [in] */ CC_DV_TYPE p) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_AspectRatio( 
            /* [retval][out] */ CC_RATIONAL *par) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_AspectRatio( 
            /* [in] */ CC_RATIONAL ar) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_DV_VideoEncoderSettingsVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_DV_VideoEncoderSettings * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_DV_VideoEncoderSettings * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_DV_VideoEncoderSettings * This);
        
        HRESULT ( STDMETHODCALLTYPE *Clear )( 
            ICC_DV_VideoEncoderSettings * This,
            /* [in] */ LPCSTR strVarName);
        
        HRESULT ( STDMETHODCALLTYPE *Assigned )( 
            ICC_DV_VideoEncoderSettings * This,
            /* [in] */ LPCSTR strVarName,
            /* [defaultvalue][retval][out] */ CC_BOOL *__MIDL__ICC_Settings0000);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_XML )( 
            ICC_DV_VideoEncoderSettings * This,
            /* [retval][out] */ CC_STRING *pstrXml);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_XML )( 
            ICC_DV_VideoEncoderSettings * This,
            /* [in] */ CC_STRING strXml);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Type )( 
            ICC_DV_VideoEncoderSettings * This,
            /* [retval][out] */ CC_DV_TYPE *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_Type )( 
            ICC_DV_VideoEncoderSettings * This,
            /* [in] */ CC_DV_TYPE p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_AspectRatio )( 
            ICC_DV_VideoEncoderSettings * This,
            /* [retval][out] */ CC_RATIONAL *par);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_AspectRatio )( 
            ICC_DV_VideoEncoderSettings * This,
            /* [in] */ CC_RATIONAL ar);
        
        END_INTERFACE
    } ICC_DV_VideoEncoderSettingsVtbl;

    interface ICC_DV_VideoEncoderSettings
    {
        CONST_VTBL struct ICC_DV_VideoEncoderSettingsVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_DV_VideoEncoderSettings_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_DV_VideoEncoderSettings_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_DV_VideoEncoderSettings_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ICC_DV_VideoEncoderSettings_Clear(This,strVarName)	\
    ( (This)->lpVtbl -> Clear(This,strVarName) ) 

#define ICC_DV_VideoEncoderSettings_Assigned(This,strVarName,__MIDL__ICC_Settings0000)	\
    ( (This)->lpVtbl -> Assigned(This,strVarName,__MIDL__ICC_Settings0000) ) 

#define ICC_DV_VideoEncoderSettings_get_XML(This,pstrXml)	\
    ( (This)->lpVtbl -> get_XML(This,pstrXml) ) 

#define ICC_DV_VideoEncoderSettings_put_XML(This,strXml)	\
    ( (This)->lpVtbl -> put_XML(This,strXml) ) 


#define ICC_DV_VideoEncoderSettings_get_Type(This,p)	\
    ( (This)->lpVtbl -> get_Type(This,p) ) 

#define ICC_DV_VideoEncoderSettings_put_Type(This,p)	\
    ( (This)->lpVtbl -> put_Type(This,p) ) 

#define ICC_DV_VideoEncoderSettings_get_AspectRatio(This,par)	\
    ( (This)->lpVtbl -> get_AspectRatio(This,par) ) 

#define ICC_DV_VideoEncoderSettings_put_AspectRatio(This,ar)	\
    ( (This)->lpVtbl -> put_AspectRatio(This,ar) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_DV_VideoEncoderSettings_INTERFACE_DEFINED__ */


#ifndef __ICC_DV_VideoDecoder_INTERFACE_DEFINED__
#define __ICC_DV_VideoDecoder_INTERFACE_DEFINED__

/* interface ICC_DV_VideoDecoder */
/* [local][unique][uuid][object] */ 


EXTERN_C const IID IID_ICC_DV_VideoDecoder;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("F119F7C2-DFF3-4ECB-9CEB-1DE4B2025E7E")
    ICC_DV_VideoDecoder : public ICC_VideoDecoder
    {
    public:
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_DV_VideoDecoderVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_DV_VideoDecoder * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_DV_VideoDecoder * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_DV_VideoDecoder * This);
        
        HRESULT ( STDMETHODCALLTYPE *Init )( 
            ICC_DV_VideoDecoder * This,
            /* [defaultvalue][in] */ ICC_Settings *pSettings);
        
        HRESULT ( STDMETHODCALLTYPE *InitByXml )( 
            ICC_DV_VideoDecoder * This,
            /* [in] */ CC_STRING strXML);
        
        HRESULT ( STDMETHODCALLTYPE *Done )( 
            ICC_DV_VideoDecoder * This,
            /* [in] */ CC_BOOL bFlush,
            /* [defaultvalue][retval][out] */ CC_BOOL *pbDone);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_IsActive )( 
            ICC_DV_VideoDecoder * This,
            /* [defaultvalue][retval][out] */ CC_BOOL *__MIDL__ICC_StreamProcessor0000);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_TimeBase )( 
            ICC_DV_VideoDecoder * This,
            /* [retval][out] */ CC_TIMEBASE *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_TimeBase )( 
            ICC_DV_VideoDecoder * This,
            /* [in] */ CC_TIMEBASE p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_BitRate )( 
            ICC_DV_VideoDecoder * This,
            /* [retval][out] */ CC_BITRATE *p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_StreamInfo )( 
            ICC_DV_VideoDecoder * This,
            /* [retval][out] */ ICC_Settings **p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_IsDataReady )( 
            ICC_DV_VideoDecoder * This,
            /* [defaultvalue][retval][out] */ CC_BOOL *p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_DataInfo )( 
            ICC_DV_VideoDecoder * This,
            /* [retval][out] */ ICC_Settings **s);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_OutputCallback )( 
            ICC_DV_VideoDecoder * This,
            /* [retval][out] */ IUnknown **p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_OutputCallback )( 
            ICC_DV_VideoDecoder * This,
            /* [in] */ IUnknown *p);
        
        HRESULT ( STDMETHODCALLTYPE *ProcessData )( 
            ICC_DV_VideoDecoder * This,
            /* [size_is][in] */ CC_PCBYTE pbData,
            /* [in] */ CC_UINT cbSize,
            /* [defaultvalue][in] */ CC_UINT cbOffset,
            /* [defaultvalue][in] */ CC_TIME pts,
            /* [defaultvalue][retval][out] */ CC_UINT *pcbProcessed);
        
        HRESULT ( STDMETHODCALLTYPE *Break )( 
            ICC_DV_VideoDecoder * This,
            /* [in] */ CC_BOOL bFlush,
            /* [defaultvalue][retval][out] */ CC_BOOL *pbDone);
        
        HRESULT ( STDMETHODCALLTYPE *GetFrame )( 
            ICC_DV_VideoDecoder * This,
            /* [in] */ CC_COLOR_FMT Format,
            /* [size_is][out] */ BYTE *pbVideoData,
            /* [in] */ DWORD cbSize,
            /* [defaultvalue][in] */ INT stride,
            /* [defaultvalue][retval][out] */ DWORD *pcbRetSize);
        
        HRESULT ( STDMETHODCALLTYPE *GetStride )( 
            ICC_DV_VideoDecoder * This,
            /* [in] */ CC_COLOR_FMT fmt,
            /* [retval][out] */ DWORD *pNumBytes);
        
        HRESULT ( STDMETHODCALLTYPE *IsFormatSupported )( 
            ICC_DV_VideoDecoder * This,
            /* [in] */ CC_COLOR_FMT fmt,
            /* [defaultvalue][retval][out] */ CC_BOOL *pResult);
        
        HRESULT ( STDMETHODCALLTYPE *GetVideoStreamInfo )( 
            ICC_DV_VideoDecoder * This,
            /* [retval][out] */ ICC_VideoStreamInfo **pDescr);
        
        HRESULT ( STDMETHODCALLTYPE *GetVideoFrameInfo )( 
            ICC_DV_VideoDecoder * This,
            /* [retval][out] */ ICC_VideoFrameInfo **pDescr);
        
        END_INTERFACE
    } ICC_DV_VideoDecoderVtbl;

    interface ICC_DV_VideoDecoder
    {
        CONST_VTBL struct ICC_DV_VideoDecoderVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_DV_VideoDecoder_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_DV_VideoDecoder_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_DV_VideoDecoder_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ICC_DV_VideoDecoder_Init(This,pSettings)	\
    ( (This)->lpVtbl -> Init(This,pSettings) ) 

#define ICC_DV_VideoDecoder_InitByXml(This,strXML)	\
    ( (This)->lpVtbl -> InitByXml(This,strXML) ) 

#define ICC_DV_VideoDecoder_Done(This,bFlush,pbDone)	\
    ( (This)->lpVtbl -> Done(This,bFlush,pbDone) ) 

#define ICC_DV_VideoDecoder_get_IsActive(This,__MIDL__ICC_StreamProcessor0000)	\
    ( (This)->lpVtbl -> get_IsActive(This,__MIDL__ICC_StreamProcessor0000) ) 

#define ICC_DV_VideoDecoder_get_TimeBase(This,p)	\
    ( (This)->lpVtbl -> get_TimeBase(This,p) ) 

#define ICC_DV_VideoDecoder_put_TimeBase(This,p)	\
    ( (This)->lpVtbl -> put_TimeBase(This,p) ) 

#define ICC_DV_VideoDecoder_get_BitRate(This,p)	\
    ( (This)->lpVtbl -> get_BitRate(This,p) ) 

#define ICC_DV_VideoDecoder_get_StreamInfo(This,p)	\
    ( (This)->lpVtbl -> get_StreamInfo(This,p) ) 

#define ICC_DV_VideoDecoder_get_IsDataReady(This,p)	\
    ( (This)->lpVtbl -> get_IsDataReady(This,p) ) 

#define ICC_DV_VideoDecoder_get_DataInfo(This,s)	\
    ( (This)->lpVtbl -> get_DataInfo(This,s) ) 

#define ICC_DV_VideoDecoder_get_OutputCallback(This,p)	\
    ( (This)->lpVtbl -> get_OutputCallback(This,p) ) 

#define ICC_DV_VideoDecoder_put_OutputCallback(This,p)	\
    ( (This)->lpVtbl -> put_OutputCallback(This,p) ) 


#define ICC_DV_VideoDecoder_ProcessData(This,pbData,cbSize,cbOffset,pts,pcbProcessed)	\
    ( (This)->lpVtbl -> ProcessData(This,pbData,cbSize,cbOffset,pts,pcbProcessed) ) 

#define ICC_DV_VideoDecoder_Break(This,bFlush,pbDone)	\
    ( (This)->lpVtbl -> Break(This,bFlush,pbDone) ) 


#define ICC_DV_VideoDecoder_GetFrame(This,Format,pbVideoData,cbSize,stride,pcbRetSize)	\
    ( (This)->lpVtbl -> GetFrame(This,Format,pbVideoData,cbSize,stride,pcbRetSize) ) 

#define ICC_DV_VideoDecoder_GetStride(This,fmt,pNumBytes)	\
    ( (This)->lpVtbl -> GetStride(This,fmt,pNumBytes) ) 

#define ICC_DV_VideoDecoder_IsFormatSupported(This,fmt,pResult)	\
    ( (This)->lpVtbl -> IsFormatSupported(This,fmt,pResult) ) 

#define ICC_DV_VideoDecoder_GetVideoStreamInfo(This,pDescr)	\
    ( (This)->lpVtbl -> GetVideoStreamInfo(This,pDescr) ) 

#define ICC_DV_VideoDecoder_GetVideoFrameInfo(This,pDescr)	\
    ( (This)->lpVtbl -> GetVideoFrameInfo(This,pDescr) ) 


#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_DV_VideoDecoder_INTERFACE_DEFINED__ */


#ifndef __ICC_DV_VideoDecoderSettings_INTERFACE_DEFINED__
#define __ICC_DV_VideoDecoderSettings_INTERFACE_DEFINED__

/* interface ICC_DV_VideoDecoderSettings */
/* [object][uuid] */ 


EXTERN_C const IID IID_ICC_DV_VideoDecoderSettings;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("28D7DB30-2594-48F5-93D6-AE43D61876CD")
    ICC_DV_VideoDecoderSettings : public ICC_Settings
    {
    public:
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_DV_VideoDecoderSettingsVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_DV_VideoDecoderSettings * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_DV_VideoDecoderSettings * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_DV_VideoDecoderSettings * This);
        
        HRESULT ( STDMETHODCALLTYPE *Clear )( 
            ICC_DV_VideoDecoderSettings * This,
            /* [in] */ LPCSTR strVarName);
        
        HRESULT ( STDMETHODCALLTYPE *Assigned )( 
            ICC_DV_VideoDecoderSettings * This,
            /* [in] */ LPCSTR strVarName,
            /* [defaultvalue][retval][out] */ CC_BOOL *__MIDL__ICC_Settings0000);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_XML )( 
            ICC_DV_VideoDecoderSettings * This,
            /* [retval][out] */ CC_STRING *pstrXml);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_XML )( 
            ICC_DV_VideoDecoderSettings * This,
            /* [in] */ CC_STRING strXml);
        
        END_INTERFACE
    } ICC_DV_VideoDecoderSettingsVtbl;

    interface ICC_DV_VideoDecoderSettings
    {
        CONST_VTBL struct ICC_DV_VideoDecoderSettingsVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_DV_VideoDecoderSettings_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_DV_VideoDecoderSettings_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_DV_VideoDecoderSettings_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ICC_DV_VideoDecoderSettings_Clear(This,strVarName)	\
    ( (This)->lpVtbl -> Clear(This,strVarName) ) 

#define ICC_DV_VideoDecoderSettings_Assigned(This,strVarName,__MIDL__ICC_Settings0000)	\
    ( (This)->lpVtbl -> Assigned(This,strVarName,__MIDL__ICC_Settings0000) ) 

#define ICC_DV_VideoDecoderSettings_get_XML(This,pstrXml)	\
    ( (This)->lpVtbl -> get_XML(This,pstrXml) ) 

#define ICC_DV_VideoDecoderSettings_put_XML(This,strXml)	\
    ( (This)->lpVtbl -> put_XML(This,strXml) ) 


#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_DV_VideoDecoderSettings_INTERFACE_DEFINED__ */


#ifndef __ICC_ProRes_VideoStreamInfo_INTERFACE_DEFINED__
#define __ICC_ProRes_VideoStreamInfo_INTERFACE_DEFINED__

/* interface ICC_ProRes_VideoStreamInfo */
/* [object][uuid] */ 


EXTERN_C const IID IID_ICC_ProRes_VideoStreamInfo;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("7DBC7F3A-C002-4B49-A3B7-1089B8773C86")
    ICC_ProRes_VideoStreamInfo : public ICC_VideoStreamInfo
    {
    public:
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_Type( 
            /* [retval][out] */ CC_PRORES_TYPE *p) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_Type( 
            /* [in] */ CC_PRORES_TYPE p) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_ProRes_VideoStreamInfoVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_ProRes_VideoStreamInfo * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_ProRes_VideoStreamInfo * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_ProRes_VideoStreamInfo * This);
        
        HRESULT ( STDMETHODCALLTYPE *Clear )( 
            ICC_ProRes_VideoStreamInfo * This,
            /* [in] */ LPCSTR strVarName);
        
        HRESULT ( STDMETHODCALLTYPE *Assigned )( 
            ICC_ProRes_VideoStreamInfo * This,
            /* [in] */ LPCSTR strVarName,
            /* [defaultvalue][retval][out] */ CC_BOOL *__MIDL__ICC_Settings0000);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_XML )( 
            ICC_ProRes_VideoStreamInfo * This,
            /* [retval][out] */ CC_STRING *pstrXml);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_XML )( 
            ICC_ProRes_VideoStreamInfo * This,
            /* [in] */ CC_STRING strXml);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_StreamType )( 
            ICC_ProRes_VideoStreamInfo * This,
            /* [retval][out] */ CC_ELEMENTARY_STREAM_TYPE *p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_BitRate )( 
            ICC_ProRes_VideoStreamInfo * This,
            /* [retval][out] */ CC_BITRATE *p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_FrameRate )( 
            ICC_ProRes_VideoStreamInfo * This,
            /* [retval][out] */ CC_FRAME_RATE *p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_FrameSize )( 
            ICC_ProRes_VideoStreamInfo * This,
            /* [retval][out] */ CC_SIZE *s);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_AspectRatio )( 
            ICC_ProRes_VideoStreamInfo * This,
            /* [retval][out] */ CC_RATIONAL *a);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_ProgressiveSequence )( 
            ICC_ProRes_VideoStreamInfo * This,
            /* [retval][out] */ CC_BOOL *x);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Type )( 
            ICC_ProRes_VideoStreamInfo * This,
            /* [retval][out] */ CC_PRORES_TYPE *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_Type )( 
            ICC_ProRes_VideoStreamInfo * This,
            /* [in] */ CC_PRORES_TYPE p);
        
        END_INTERFACE
    } ICC_ProRes_VideoStreamInfoVtbl;

    interface ICC_ProRes_VideoStreamInfo
    {
        CONST_VTBL struct ICC_ProRes_VideoStreamInfoVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_ProRes_VideoStreamInfo_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_ProRes_VideoStreamInfo_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_ProRes_VideoStreamInfo_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ICC_ProRes_VideoStreamInfo_Clear(This,strVarName)	\
    ( (This)->lpVtbl -> Clear(This,strVarName) ) 

#define ICC_ProRes_VideoStreamInfo_Assigned(This,strVarName,__MIDL__ICC_Settings0000)	\
    ( (This)->lpVtbl -> Assigned(This,strVarName,__MIDL__ICC_Settings0000) ) 

#define ICC_ProRes_VideoStreamInfo_get_XML(This,pstrXml)	\
    ( (This)->lpVtbl -> get_XML(This,pstrXml) ) 

#define ICC_ProRes_VideoStreamInfo_put_XML(This,strXml)	\
    ( (This)->lpVtbl -> put_XML(This,strXml) ) 


#define ICC_ProRes_VideoStreamInfo_get_StreamType(This,p)	\
    ( (This)->lpVtbl -> get_StreamType(This,p) ) 

#define ICC_ProRes_VideoStreamInfo_get_BitRate(This,p)	\
    ( (This)->lpVtbl -> get_BitRate(This,p) ) 

#define ICC_ProRes_VideoStreamInfo_get_FrameRate(This,p)	\
    ( (This)->lpVtbl -> get_FrameRate(This,p) ) 


#define ICC_ProRes_VideoStreamInfo_get_FrameSize(This,s)	\
    ( (This)->lpVtbl -> get_FrameSize(This,s) ) 

#define ICC_ProRes_VideoStreamInfo_get_AspectRatio(This,a)	\
    ( (This)->lpVtbl -> get_AspectRatio(This,a) ) 

#define ICC_ProRes_VideoStreamInfo_get_ProgressiveSequence(This,x)	\
    ( (This)->lpVtbl -> get_ProgressiveSequence(This,x) ) 


#define ICC_ProRes_VideoStreamInfo_get_Type(This,p)	\
    ( (This)->lpVtbl -> get_Type(This,p) ) 

#define ICC_ProRes_VideoStreamInfo_put_Type(This,p)	\
    ( (This)->lpVtbl -> put_Type(This,p) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_ProRes_VideoStreamInfo_INTERFACE_DEFINED__ */


#ifndef __ICC_ProRes_VideoFrameInfo_INTERFACE_DEFINED__
#define __ICC_ProRes_VideoFrameInfo_INTERFACE_DEFINED__

/* interface ICC_ProRes_VideoFrameInfo */
/* [object][uuid] */ 


EXTERN_C const IID IID_ICC_ProRes_VideoFrameInfo;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("1C727CE7-518C-49FA-AA85-5EE5F3068448")
    ICC_ProRes_VideoFrameInfo : public ICC_VideoFrameInfo
    {
    public:
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_ProRes_VideoFrameInfoVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_ProRes_VideoFrameInfo * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_ProRes_VideoFrameInfo * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_ProRes_VideoFrameInfo * This);
        
        HRESULT ( STDMETHODCALLTYPE *Clear )( 
            ICC_ProRes_VideoFrameInfo * This,
            /* [in] */ LPCSTR strVarName);
        
        HRESULT ( STDMETHODCALLTYPE *Assigned )( 
            ICC_ProRes_VideoFrameInfo * This,
            /* [in] */ LPCSTR strVarName,
            /* [defaultvalue][retval][out] */ CC_BOOL *__MIDL__ICC_Settings0000);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_XML )( 
            ICC_ProRes_VideoFrameInfo * This,
            /* [retval][out] */ CC_STRING *pstrXml);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_XML )( 
            ICC_ProRes_VideoFrameInfo * This,
            /* [in] */ CC_STRING strXml);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_DataSize )( 
            ICC_ProRes_VideoFrameInfo * This,
            /* [retval][out] */ CC_UINT *__MIDL__ICC_ByteStreamDataInfo0000);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_ByteOffset )( 
            ICC_ProRes_VideoFrameInfo * This,
            /* [retval][out] */ CC_OFFSET *__MIDL__ICC_ByteStreamDataInfo0001);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_NumSamples )( 
            ICC_ProRes_VideoFrameInfo * This,
            /* [retval][out] */ CC_UINT *__MIDL__ICC_ElementaryDataInfo0000);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_SampleOffset )( 
            ICC_ProRes_VideoFrameInfo * This,
            /* [retval][out] */ CC_OFFSET *__MIDL__ICC_ElementaryDataInfo0001);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Duration )( 
            ICC_ProRes_VideoFrameInfo * This,
            /* [retval][out] */ CC_TIME *__MIDL__ICC_ElementaryDataInfo0002);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_PresentationDelta )( 
            ICC_ProRes_VideoFrameInfo * This,
            /* [retval][out] */ CC_INT *__MIDL__ICC_ElementaryDataInfo0003);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_PTS )( 
            ICC_ProRes_VideoFrameInfo * This,
            /* [retval][out] */ CC_TIME *__MIDL__ICC_ElementaryDataInfo0004);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_DTS )( 
            ICC_ProRes_VideoFrameInfo * This,
            /* [retval][out] */ CC_TIME *__MIDL__ICC_ElementaryDataInfo0005);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_SequenceEntryFlag )( 
            ICC_ProRes_VideoFrameInfo * This,
            /* [retval][out] */ CC_BOOL *__MIDL__ICC_ElementaryDataInfo0006);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Number )( 
            ICC_ProRes_VideoFrameInfo * This,
            /* [retval][out] */ CC_UINT *n);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_CodingNumber )( 
            ICC_ProRes_VideoFrameInfo * This,
            /* [retval][out] */ CC_UINT *c);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_TimeCode )( 
            ICC_ProRes_VideoFrameInfo * This,
            /* [retval][out] */ CC_TIMECODE *t);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_InterlaceType )( 
            ICC_ProRes_VideoFrameInfo * This,
            /* [retval][out] */ CC_INTERLACE_TYPE *i);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_FrameType )( 
            ICC_ProRes_VideoFrameInfo * This,
            /* [retval][out] */ CC_FRAME_TYPE *x);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_PictStruct )( 
            ICC_ProRes_VideoFrameInfo * This,
            /* [retval][out] */ CC_PICTURE_STRUCTURE *x);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Flags )( 
            ICC_ProRes_VideoFrameInfo * This,
            /* [retval][out] */ DWORD *x);
        
        END_INTERFACE
    } ICC_ProRes_VideoFrameInfoVtbl;

    interface ICC_ProRes_VideoFrameInfo
    {
        CONST_VTBL struct ICC_ProRes_VideoFrameInfoVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_ProRes_VideoFrameInfo_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_ProRes_VideoFrameInfo_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_ProRes_VideoFrameInfo_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ICC_ProRes_VideoFrameInfo_Clear(This,strVarName)	\
    ( (This)->lpVtbl -> Clear(This,strVarName) ) 

#define ICC_ProRes_VideoFrameInfo_Assigned(This,strVarName,__MIDL__ICC_Settings0000)	\
    ( (This)->lpVtbl -> Assigned(This,strVarName,__MIDL__ICC_Settings0000) ) 

#define ICC_ProRes_VideoFrameInfo_get_XML(This,pstrXml)	\
    ( (This)->lpVtbl -> get_XML(This,pstrXml) ) 

#define ICC_ProRes_VideoFrameInfo_put_XML(This,strXml)	\
    ( (This)->lpVtbl -> put_XML(This,strXml) ) 


#define ICC_ProRes_VideoFrameInfo_get_DataSize(This,__MIDL__ICC_ByteStreamDataInfo0000)	\
    ( (This)->lpVtbl -> get_DataSize(This,__MIDL__ICC_ByteStreamDataInfo0000) ) 

#define ICC_ProRes_VideoFrameInfo_get_ByteOffset(This,__MIDL__ICC_ByteStreamDataInfo0001)	\
    ( (This)->lpVtbl -> get_ByteOffset(This,__MIDL__ICC_ByteStreamDataInfo0001) ) 


#define ICC_ProRes_VideoFrameInfo_get_NumSamples(This,__MIDL__ICC_ElementaryDataInfo0000)	\
    ( (This)->lpVtbl -> get_NumSamples(This,__MIDL__ICC_ElementaryDataInfo0000) ) 

#define ICC_ProRes_VideoFrameInfo_get_SampleOffset(This,__MIDL__ICC_ElementaryDataInfo0001)	\
    ( (This)->lpVtbl -> get_SampleOffset(This,__MIDL__ICC_ElementaryDataInfo0001) ) 

#define ICC_ProRes_VideoFrameInfo_get_Duration(This,__MIDL__ICC_ElementaryDataInfo0002)	\
    ( (This)->lpVtbl -> get_Duration(This,__MIDL__ICC_ElementaryDataInfo0002) ) 

#define ICC_ProRes_VideoFrameInfo_get_PresentationDelta(This,__MIDL__ICC_ElementaryDataInfo0003)	\
    ( (This)->lpVtbl -> get_PresentationDelta(This,__MIDL__ICC_ElementaryDataInfo0003) ) 

#define ICC_ProRes_VideoFrameInfo_get_PTS(This,__MIDL__ICC_ElementaryDataInfo0004)	\
    ( (This)->lpVtbl -> get_PTS(This,__MIDL__ICC_ElementaryDataInfo0004) ) 

#define ICC_ProRes_VideoFrameInfo_get_DTS(This,__MIDL__ICC_ElementaryDataInfo0005)	\
    ( (This)->lpVtbl -> get_DTS(This,__MIDL__ICC_ElementaryDataInfo0005) ) 

#define ICC_ProRes_VideoFrameInfo_get_SequenceEntryFlag(This,__MIDL__ICC_ElementaryDataInfo0006)	\
    ( (This)->lpVtbl -> get_SequenceEntryFlag(This,__MIDL__ICC_ElementaryDataInfo0006) ) 


#define ICC_ProRes_VideoFrameInfo_get_Number(This,n)	\
    ( (This)->lpVtbl -> get_Number(This,n) ) 

#define ICC_ProRes_VideoFrameInfo_get_CodingNumber(This,c)	\
    ( (This)->lpVtbl -> get_CodingNumber(This,c) ) 

#define ICC_ProRes_VideoFrameInfo_get_TimeCode(This,t)	\
    ( (This)->lpVtbl -> get_TimeCode(This,t) ) 

#define ICC_ProRes_VideoFrameInfo_get_InterlaceType(This,i)	\
    ( (This)->lpVtbl -> get_InterlaceType(This,i) ) 

#define ICC_ProRes_VideoFrameInfo_get_FrameType(This,x)	\
    ( (This)->lpVtbl -> get_FrameType(This,x) ) 

#define ICC_ProRes_VideoFrameInfo_get_PictStruct(This,x)	\
    ( (This)->lpVtbl -> get_PictStruct(This,x) ) 

#define ICC_ProRes_VideoFrameInfo_get_Flags(This,x)	\
    ( (This)->lpVtbl -> get_Flags(This,x) ) 


#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_ProRes_VideoFrameInfo_INTERFACE_DEFINED__ */


#ifndef __ICC_ProRes_VideoDecoderSettings_INTERFACE_DEFINED__
#define __ICC_ProRes_VideoDecoderSettings_INTERFACE_DEFINED__

/* interface ICC_ProRes_VideoDecoderSettings */
/* [object][uuid] */ 


EXTERN_C const IID IID_ICC_ProRes_VideoDecoderSettings;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("3F08F282-34D1-4034-9CC9-70BCE2D7FA6E")
    ICC_ProRes_VideoDecoderSettings : public ICC_Settings
    {
    public:
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_FrameSize( 
            /* [retval][out] */ CC_SIZE *x) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_FrameSize( 
            /* [in] */ CC_SIZE x) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_ProRes_VideoDecoderSettingsVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_ProRes_VideoDecoderSettings * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_ProRes_VideoDecoderSettings * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_ProRes_VideoDecoderSettings * This);
        
        HRESULT ( STDMETHODCALLTYPE *Clear )( 
            ICC_ProRes_VideoDecoderSettings * This,
            /* [in] */ LPCSTR strVarName);
        
        HRESULT ( STDMETHODCALLTYPE *Assigned )( 
            ICC_ProRes_VideoDecoderSettings * This,
            /* [in] */ LPCSTR strVarName,
            /* [defaultvalue][retval][out] */ CC_BOOL *__MIDL__ICC_Settings0000);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_XML )( 
            ICC_ProRes_VideoDecoderSettings * This,
            /* [retval][out] */ CC_STRING *pstrXml);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_XML )( 
            ICC_ProRes_VideoDecoderSettings * This,
            /* [in] */ CC_STRING strXml);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_FrameSize )( 
            ICC_ProRes_VideoDecoderSettings * This,
            /* [retval][out] */ CC_SIZE *x);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_FrameSize )( 
            ICC_ProRes_VideoDecoderSettings * This,
            /* [in] */ CC_SIZE x);
        
        END_INTERFACE
    } ICC_ProRes_VideoDecoderSettingsVtbl;

    interface ICC_ProRes_VideoDecoderSettings
    {
        CONST_VTBL struct ICC_ProRes_VideoDecoderSettingsVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_ProRes_VideoDecoderSettings_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_ProRes_VideoDecoderSettings_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_ProRes_VideoDecoderSettings_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ICC_ProRes_VideoDecoderSettings_Clear(This,strVarName)	\
    ( (This)->lpVtbl -> Clear(This,strVarName) ) 

#define ICC_ProRes_VideoDecoderSettings_Assigned(This,strVarName,__MIDL__ICC_Settings0000)	\
    ( (This)->lpVtbl -> Assigned(This,strVarName,__MIDL__ICC_Settings0000) ) 

#define ICC_ProRes_VideoDecoderSettings_get_XML(This,pstrXml)	\
    ( (This)->lpVtbl -> get_XML(This,pstrXml) ) 

#define ICC_ProRes_VideoDecoderSettings_put_XML(This,strXml)	\
    ( (This)->lpVtbl -> put_XML(This,strXml) ) 


#define ICC_ProRes_VideoDecoderSettings_get_FrameSize(This,x)	\
    ( (This)->lpVtbl -> get_FrameSize(This,x) ) 

#define ICC_ProRes_VideoDecoderSettings_put_FrameSize(This,x)	\
    ( (This)->lpVtbl -> put_FrameSize(This,x) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_ProRes_VideoDecoderSettings_INTERFACE_DEFINED__ */


#ifndef __ICC_ProRes_VideoDecoder_INTERFACE_DEFINED__
#define __ICC_ProRes_VideoDecoder_INTERFACE_DEFINED__

/* interface ICC_ProRes_VideoDecoder */
/* [local][unique][uuid][object] */ 


EXTERN_C const IID IID_ICC_ProRes_VideoDecoder;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("10269D3C-63C3-40FA-92FA-43138BEB1CA4")
    ICC_ProRes_VideoDecoder : public ICC_VideoDecoder
    {
    public:
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_ProRes_VideoDecoderVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_ProRes_VideoDecoder * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_ProRes_VideoDecoder * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_ProRes_VideoDecoder * This);
        
        HRESULT ( STDMETHODCALLTYPE *Init )( 
            ICC_ProRes_VideoDecoder * This,
            /* [defaultvalue][in] */ ICC_Settings *pSettings);
        
        HRESULT ( STDMETHODCALLTYPE *InitByXml )( 
            ICC_ProRes_VideoDecoder * This,
            /* [in] */ CC_STRING strXML);
        
        HRESULT ( STDMETHODCALLTYPE *Done )( 
            ICC_ProRes_VideoDecoder * This,
            /* [in] */ CC_BOOL bFlush,
            /* [defaultvalue][retval][out] */ CC_BOOL *pbDone);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_IsActive )( 
            ICC_ProRes_VideoDecoder * This,
            /* [defaultvalue][retval][out] */ CC_BOOL *__MIDL__ICC_StreamProcessor0000);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_TimeBase )( 
            ICC_ProRes_VideoDecoder * This,
            /* [retval][out] */ CC_TIMEBASE *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_TimeBase )( 
            ICC_ProRes_VideoDecoder * This,
            /* [in] */ CC_TIMEBASE p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_BitRate )( 
            ICC_ProRes_VideoDecoder * This,
            /* [retval][out] */ CC_BITRATE *p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_StreamInfo )( 
            ICC_ProRes_VideoDecoder * This,
            /* [retval][out] */ ICC_Settings **p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_IsDataReady )( 
            ICC_ProRes_VideoDecoder * This,
            /* [defaultvalue][retval][out] */ CC_BOOL *p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_DataInfo )( 
            ICC_ProRes_VideoDecoder * This,
            /* [retval][out] */ ICC_Settings **s);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_OutputCallback )( 
            ICC_ProRes_VideoDecoder * This,
            /* [retval][out] */ IUnknown **p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_OutputCallback )( 
            ICC_ProRes_VideoDecoder * This,
            /* [in] */ IUnknown *p);
        
        HRESULT ( STDMETHODCALLTYPE *ProcessData )( 
            ICC_ProRes_VideoDecoder * This,
            /* [size_is][in] */ CC_PCBYTE pbData,
            /* [in] */ CC_UINT cbSize,
            /* [defaultvalue][in] */ CC_UINT cbOffset,
            /* [defaultvalue][in] */ CC_TIME pts,
            /* [defaultvalue][retval][out] */ CC_UINT *pcbProcessed);
        
        HRESULT ( STDMETHODCALLTYPE *Break )( 
            ICC_ProRes_VideoDecoder * This,
            /* [in] */ CC_BOOL bFlush,
            /* [defaultvalue][retval][out] */ CC_BOOL *pbDone);
        
        HRESULT ( STDMETHODCALLTYPE *GetFrame )( 
            ICC_ProRes_VideoDecoder * This,
            /* [in] */ CC_COLOR_FMT Format,
            /* [size_is][out] */ BYTE *pbVideoData,
            /* [in] */ DWORD cbSize,
            /* [defaultvalue][in] */ INT stride,
            /* [defaultvalue][retval][out] */ DWORD *pcbRetSize);
        
        HRESULT ( STDMETHODCALLTYPE *GetStride )( 
            ICC_ProRes_VideoDecoder * This,
            /* [in] */ CC_COLOR_FMT fmt,
            /* [retval][out] */ DWORD *pNumBytes);
        
        HRESULT ( STDMETHODCALLTYPE *IsFormatSupported )( 
            ICC_ProRes_VideoDecoder * This,
            /* [in] */ CC_COLOR_FMT fmt,
            /* [defaultvalue][retval][out] */ CC_BOOL *pResult);
        
        HRESULT ( STDMETHODCALLTYPE *GetVideoStreamInfo )( 
            ICC_ProRes_VideoDecoder * This,
            /* [retval][out] */ ICC_VideoStreamInfo **pDescr);
        
        HRESULT ( STDMETHODCALLTYPE *GetVideoFrameInfo )( 
            ICC_ProRes_VideoDecoder * This,
            /* [retval][out] */ ICC_VideoFrameInfo **pDescr);
        
        END_INTERFACE
    } ICC_ProRes_VideoDecoderVtbl;

    interface ICC_ProRes_VideoDecoder
    {
        CONST_VTBL struct ICC_ProRes_VideoDecoderVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_ProRes_VideoDecoder_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_ProRes_VideoDecoder_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_ProRes_VideoDecoder_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ICC_ProRes_VideoDecoder_Init(This,pSettings)	\
    ( (This)->lpVtbl -> Init(This,pSettings) ) 

#define ICC_ProRes_VideoDecoder_InitByXml(This,strXML)	\
    ( (This)->lpVtbl -> InitByXml(This,strXML) ) 

#define ICC_ProRes_VideoDecoder_Done(This,bFlush,pbDone)	\
    ( (This)->lpVtbl -> Done(This,bFlush,pbDone) ) 

#define ICC_ProRes_VideoDecoder_get_IsActive(This,__MIDL__ICC_StreamProcessor0000)	\
    ( (This)->lpVtbl -> get_IsActive(This,__MIDL__ICC_StreamProcessor0000) ) 

#define ICC_ProRes_VideoDecoder_get_TimeBase(This,p)	\
    ( (This)->lpVtbl -> get_TimeBase(This,p) ) 

#define ICC_ProRes_VideoDecoder_put_TimeBase(This,p)	\
    ( (This)->lpVtbl -> put_TimeBase(This,p) ) 

#define ICC_ProRes_VideoDecoder_get_BitRate(This,p)	\
    ( (This)->lpVtbl -> get_BitRate(This,p) ) 

#define ICC_ProRes_VideoDecoder_get_StreamInfo(This,p)	\
    ( (This)->lpVtbl -> get_StreamInfo(This,p) ) 

#define ICC_ProRes_VideoDecoder_get_IsDataReady(This,p)	\
    ( (This)->lpVtbl -> get_IsDataReady(This,p) ) 

#define ICC_ProRes_VideoDecoder_get_DataInfo(This,s)	\
    ( (This)->lpVtbl -> get_DataInfo(This,s) ) 

#define ICC_ProRes_VideoDecoder_get_OutputCallback(This,p)	\
    ( (This)->lpVtbl -> get_OutputCallback(This,p) ) 

#define ICC_ProRes_VideoDecoder_put_OutputCallback(This,p)	\
    ( (This)->lpVtbl -> put_OutputCallback(This,p) ) 


#define ICC_ProRes_VideoDecoder_ProcessData(This,pbData,cbSize,cbOffset,pts,pcbProcessed)	\
    ( (This)->lpVtbl -> ProcessData(This,pbData,cbSize,cbOffset,pts,pcbProcessed) ) 

#define ICC_ProRes_VideoDecoder_Break(This,bFlush,pbDone)	\
    ( (This)->lpVtbl -> Break(This,bFlush,pbDone) ) 


#define ICC_ProRes_VideoDecoder_GetFrame(This,Format,pbVideoData,cbSize,stride,pcbRetSize)	\
    ( (This)->lpVtbl -> GetFrame(This,Format,pbVideoData,cbSize,stride,pcbRetSize) ) 

#define ICC_ProRes_VideoDecoder_GetStride(This,fmt,pNumBytes)	\
    ( (This)->lpVtbl -> GetStride(This,fmt,pNumBytes) ) 

#define ICC_ProRes_VideoDecoder_IsFormatSupported(This,fmt,pResult)	\
    ( (This)->lpVtbl -> IsFormatSupported(This,fmt,pResult) ) 

#define ICC_ProRes_VideoDecoder_GetVideoStreamInfo(This,pDescr)	\
    ( (This)->lpVtbl -> GetVideoStreamInfo(This,pDescr) ) 

#define ICC_ProRes_VideoDecoder_GetVideoFrameInfo(This,pDescr)	\
    ( (This)->lpVtbl -> GetVideoFrameInfo(This,pDescr) ) 


#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_ProRes_VideoDecoder_INTERFACE_DEFINED__ */


#ifndef __ICC_ProRes_VideoEncoderSettings_INTERFACE_DEFINED__
#define __ICC_ProRes_VideoEncoderSettings_INTERFACE_DEFINED__

/* interface ICC_ProRes_VideoEncoderSettings */
/* [object][uuid] */ 


EXTERN_C const IID IID_ICC_ProRes_VideoEncoderSettings;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("DA007EDD-349D-4DEF-BD4D-A2E255784BBE")
    ICC_ProRes_VideoEncoderSettings : public ICC_Settings
    {
    public:
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_Type( 
            /* [retval][out] */ CC_PRORES_TYPE *type) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_Type( 
            /* [in] */ CC_PRORES_TYPE type) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_FrameRate( 
            /* [retval][out] */ CC_RATIONAL *fr) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_FrameRate( 
            /* [in] */ CC_RATIONAL fr) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_FrameSize( 
            /* [retval][out] */ CC_SIZE *size) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_FrameSize( 
            /* [in] */ CC_SIZE size) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_InterlaceType( 
            /* [retval][out] */ CC_INTERLACE_TYPE *type) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_InterlaceType( 
            /* [in] */ CC_INTERLACE_TYPE type) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_PreserveAlpha( 
            /* [retval][out] */ CC_BOOL *preserve) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_PreserveAlpha( 
            /* [in] */ CC_BOOL preserve) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_ColorCoefs( 
            /* [retval][out] */ CC_COLOUR_DESCRIPTION *__MIDL__ICC_ProRes_VideoEncoderSettings0000) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_ColorCoefs( 
            /* [in] */ CC_COLOUR_DESCRIPTION __MIDL__ICC_ProRes_VideoEncoderSettings0001) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_AspectRatio( 
            /* [retval][out] */ CC_RATIONAL *par) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_AspectRatio( 
            /* [in] */ CC_RATIONAL ar) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_ProRes_VideoEncoderSettingsVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_ProRes_VideoEncoderSettings * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_ProRes_VideoEncoderSettings * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_ProRes_VideoEncoderSettings * This);
        
        HRESULT ( STDMETHODCALLTYPE *Clear )( 
            ICC_ProRes_VideoEncoderSettings * This,
            /* [in] */ LPCSTR strVarName);
        
        HRESULT ( STDMETHODCALLTYPE *Assigned )( 
            ICC_ProRes_VideoEncoderSettings * This,
            /* [in] */ LPCSTR strVarName,
            /* [defaultvalue][retval][out] */ CC_BOOL *__MIDL__ICC_Settings0000);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_XML )( 
            ICC_ProRes_VideoEncoderSettings * This,
            /* [retval][out] */ CC_STRING *pstrXml);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_XML )( 
            ICC_ProRes_VideoEncoderSettings * This,
            /* [in] */ CC_STRING strXml);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Type )( 
            ICC_ProRes_VideoEncoderSettings * This,
            /* [retval][out] */ CC_PRORES_TYPE *type);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_Type )( 
            ICC_ProRes_VideoEncoderSettings * This,
            /* [in] */ CC_PRORES_TYPE type);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_FrameRate )( 
            ICC_ProRes_VideoEncoderSettings * This,
            /* [retval][out] */ CC_RATIONAL *fr);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_FrameRate )( 
            ICC_ProRes_VideoEncoderSettings * This,
            /* [in] */ CC_RATIONAL fr);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_FrameSize )( 
            ICC_ProRes_VideoEncoderSettings * This,
            /* [retval][out] */ CC_SIZE *size);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_FrameSize )( 
            ICC_ProRes_VideoEncoderSettings * This,
            /* [in] */ CC_SIZE size);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_InterlaceType )( 
            ICC_ProRes_VideoEncoderSettings * This,
            /* [retval][out] */ CC_INTERLACE_TYPE *type);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_InterlaceType )( 
            ICC_ProRes_VideoEncoderSettings * This,
            /* [in] */ CC_INTERLACE_TYPE type);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_PreserveAlpha )( 
            ICC_ProRes_VideoEncoderSettings * This,
            /* [retval][out] */ CC_BOOL *preserve);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_PreserveAlpha )( 
            ICC_ProRes_VideoEncoderSettings * This,
            /* [in] */ CC_BOOL preserve);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_ColorCoefs )( 
            ICC_ProRes_VideoEncoderSettings * This,
            /* [retval][out] */ CC_COLOUR_DESCRIPTION *__MIDL__ICC_ProRes_VideoEncoderSettings0000);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_ColorCoefs )( 
            ICC_ProRes_VideoEncoderSettings * This,
            /* [in] */ CC_COLOUR_DESCRIPTION __MIDL__ICC_ProRes_VideoEncoderSettings0001);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_AspectRatio )( 
            ICC_ProRes_VideoEncoderSettings * This,
            /* [retval][out] */ CC_RATIONAL *par);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_AspectRatio )( 
            ICC_ProRes_VideoEncoderSettings * This,
            /* [in] */ CC_RATIONAL ar);
        
        END_INTERFACE
    } ICC_ProRes_VideoEncoderSettingsVtbl;

    interface ICC_ProRes_VideoEncoderSettings
    {
        CONST_VTBL struct ICC_ProRes_VideoEncoderSettingsVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_ProRes_VideoEncoderSettings_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_ProRes_VideoEncoderSettings_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_ProRes_VideoEncoderSettings_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ICC_ProRes_VideoEncoderSettings_Clear(This,strVarName)	\
    ( (This)->lpVtbl -> Clear(This,strVarName) ) 

#define ICC_ProRes_VideoEncoderSettings_Assigned(This,strVarName,__MIDL__ICC_Settings0000)	\
    ( (This)->lpVtbl -> Assigned(This,strVarName,__MIDL__ICC_Settings0000) ) 

#define ICC_ProRes_VideoEncoderSettings_get_XML(This,pstrXml)	\
    ( (This)->lpVtbl -> get_XML(This,pstrXml) ) 

#define ICC_ProRes_VideoEncoderSettings_put_XML(This,strXml)	\
    ( (This)->lpVtbl -> put_XML(This,strXml) ) 


#define ICC_ProRes_VideoEncoderSettings_get_Type(This,type)	\
    ( (This)->lpVtbl -> get_Type(This,type) ) 

#define ICC_ProRes_VideoEncoderSettings_put_Type(This,type)	\
    ( (This)->lpVtbl -> put_Type(This,type) ) 

#define ICC_ProRes_VideoEncoderSettings_get_FrameRate(This,fr)	\
    ( (This)->lpVtbl -> get_FrameRate(This,fr) ) 

#define ICC_ProRes_VideoEncoderSettings_put_FrameRate(This,fr)	\
    ( (This)->lpVtbl -> put_FrameRate(This,fr) ) 

#define ICC_ProRes_VideoEncoderSettings_get_FrameSize(This,size)	\
    ( (This)->lpVtbl -> get_FrameSize(This,size) ) 

#define ICC_ProRes_VideoEncoderSettings_put_FrameSize(This,size)	\
    ( (This)->lpVtbl -> put_FrameSize(This,size) ) 

#define ICC_ProRes_VideoEncoderSettings_get_InterlaceType(This,type)	\
    ( (This)->lpVtbl -> get_InterlaceType(This,type) ) 

#define ICC_ProRes_VideoEncoderSettings_put_InterlaceType(This,type)	\
    ( (This)->lpVtbl -> put_InterlaceType(This,type) ) 

#define ICC_ProRes_VideoEncoderSettings_get_PreserveAlpha(This,preserve)	\
    ( (This)->lpVtbl -> get_PreserveAlpha(This,preserve) ) 

#define ICC_ProRes_VideoEncoderSettings_put_PreserveAlpha(This,preserve)	\
    ( (This)->lpVtbl -> put_PreserveAlpha(This,preserve) ) 

#define ICC_ProRes_VideoEncoderSettings_get_ColorCoefs(This,__MIDL__ICC_ProRes_VideoEncoderSettings0000)	\
    ( (This)->lpVtbl -> get_ColorCoefs(This,__MIDL__ICC_ProRes_VideoEncoderSettings0000) ) 

#define ICC_ProRes_VideoEncoderSettings_put_ColorCoefs(This,__MIDL__ICC_ProRes_VideoEncoderSettings0001)	\
    ( (This)->lpVtbl -> put_ColorCoefs(This,__MIDL__ICC_ProRes_VideoEncoderSettings0001) ) 

#define ICC_ProRes_VideoEncoderSettings_get_AspectRatio(This,par)	\
    ( (This)->lpVtbl -> get_AspectRatio(This,par) ) 

#define ICC_ProRes_VideoEncoderSettings_put_AspectRatio(This,ar)	\
    ( (This)->lpVtbl -> put_AspectRatio(This,ar) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_ProRes_VideoEncoderSettings_INTERFACE_DEFINED__ */


#ifndef __ICC_ProRes_VideoEncoder_INTERFACE_DEFINED__
#define __ICC_ProRes_VideoEncoder_INTERFACE_DEFINED__

/* interface ICC_ProRes_VideoEncoder */
/* [local][unique][uuid][object] */ 


EXTERN_C const IID IID_ICC_ProRes_VideoEncoder;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("F4E9D858-4E1E-4DAE-95AD-2FBE502B87D4")
    ICC_ProRes_VideoEncoder : public ICC_VideoEncoder
    {
    public:
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_ProRes_VideoEncoderVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_ProRes_VideoEncoder * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_ProRes_VideoEncoder * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_ProRes_VideoEncoder * This);
        
        HRESULT ( STDMETHODCALLTYPE *Init )( 
            ICC_ProRes_VideoEncoder * This,
            /* [defaultvalue][in] */ ICC_Settings *pSettings);
        
        HRESULT ( STDMETHODCALLTYPE *InitByXml )( 
            ICC_ProRes_VideoEncoder * This,
            /* [in] */ CC_STRING strXML);
        
        HRESULT ( STDMETHODCALLTYPE *Done )( 
            ICC_ProRes_VideoEncoder * This,
            /* [in] */ CC_BOOL bFlush,
            /* [defaultvalue][retval][out] */ CC_BOOL *pbDone);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_IsActive )( 
            ICC_ProRes_VideoEncoder * This,
            /* [defaultvalue][retval][out] */ CC_BOOL *__MIDL__ICC_StreamProcessor0000);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_TimeBase )( 
            ICC_ProRes_VideoEncoder * This,
            /* [retval][out] */ CC_TIMEBASE *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_TimeBase )( 
            ICC_ProRes_VideoEncoder * This,
            /* [in] */ CC_TIMEBASE p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_BitRate )( 
            ICC_ProRes_VideoEncoder * This,
            /* [retval][out] */ CC_BITRATE *p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_StreamInfo )( 
            ICC_ProRes_VideoEncoder * This,
            /* [retval][out] */ ICC_Settings **p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_IsDataReady )( 
            ICC_ProRes_VideoEncoder * This,
            /* [defaultvalue][retval][out] */ CC_BOOL *p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_DataInfo )( 
            ICC_ProRes_VideoEncoder * This,
            /* [retval][out] */ ICC_Settings **s);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_OutputCallback )( 
            ICC_ProRes_VideoEncoder * This,
            /* [retval][out] */ IUnknown **p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_OutputCallback )( 
            ICC_ProRes_VideoEncoder * This,
            /* [in] */ IUnknown *p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_DataSize )( 
            ICC_ProRes_VideoEncoder * This,
            /* [retval][out] */ CC_UINT *__MIDL__ICC_Encoder0000);
        
        HRESULT ( STDMETHODCALLTYPE *GetData )( 
            ICC_ProRes_VideoEncoder * This,
            /* [size_is][out] */ CC_PBYTE pbData,
            /* [in] */ CC_UINT cbBufSize,
            /* [defaultvalue][retval][out] */ CC_UINT *pcbRetSize);
        
        HRESULT ( STDMETHODCALLTYPE *AddFrame )( 
            ICC_ProRes_VideoEncoder * This,
            /* [in] */ CC_COLOR_FMT Format,
            /* [size_is][in] */ const BYTE *pData,
            /* [in] */ DWORD cbSize,
            /* [defaultvalue][in] */ INT stride,
            /* [defaultvalue][retval][out] */ CC_BOOL *pResult);
        
        HRESULT ( STDMETHODCALLTYPE *GetStride )( 
            ICC_ProRes_VideoEncoder * This,
            /* [in] */ CC_COLOR_FMT fmt,
            /* [retval][out] */ DWORD *pNumBytes);
        
        HRESULT ( STDMETHODCALLTYPE *IsFormatSupported )( 
            ICC_ProRes_VideoEncoder * This,
            /* [in] */ CC_COLOR_FMT fmt,
            /* [defaultvalue][retval][out] */ CC_BOOL *pResult);
        
        HRESULT ( STDMETHODCALLTYPE *AddScaleFrame )( 
            ICC_ProRes_VideoEncoder * This,
            /* [size_is][in] */ const BYTE *pData,
            /* [in] */ DWORD cbSize,
            /* [in] */ CC_ADD_VIDEO_FRAME_PARAMS *pParams,
            /* [defaultvalue][retval][out] */ CC_BOOL *pResult);
        
        HRESULT ( STDMETHODCALLTYPE *IsScaleAvailable )( 
            ICC_ProRes_VideoEncoder * This,
            /* [in] */ CC_ADD_VIDEO_FRAME_PARAMS *pParams,
            /* [defaultvalue][retval][out] */ CC_BOOL *__MIDL__ICC_VideoEncoder0000);
        
        HRESULT ( STDMETHODCALLTYPE *GetVideoStreamInfo )( 
            ICC_ProRes_VideoEncoder * This,
            /* [retval][out] */ ICC_VideoStreamInfo **pDescr);
        
        HRESULT ( STDMETHODCALLTYPE *GetVideoFrameInfo )( 
            ICC_ProRes_VideoEncoder * This,
            /* [retval][out] */ ICC_VideoFrameInfo **pDescr);
        
        END_INTERFACE
    } ICC_ProRes_VideoEncoderVtbl;

    interface ICC_ProRes_VideoEncoder
    {
        CONST_VTBL struct ICC_ProRes_VideoEncoderVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_ProRes_VideoEncoder_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_ProRes_VideoEncoder_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_ProRes_VideoEncoder_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ICC_ProRes_VideoEncoder_Init(This,pSettings)	\
    ( (This)->lpVtbl -> Init(This,pSettings) ) 

#define ICC_ProRes_VideoEncoder_InitByXml(This,strXML)	\
    ( (This)->lpVtbl -> InitByXml(This,strXML) ) 

#define ICC_ProRes_VideoEncoder_Done(This,bFlush,pbDone)	\
    ( (This)->lpVtbl -> Done(This,bFlush,pbDone) ) 

#define ICC_ProRes_VideoEncoder_get_IsActive(This,__MIDL__ICC_StreamProcessor0000)	\
    ( (This)->lpVtbl -> get_IsActive(This,__MIDL__ICC_StreamProcessor0000) ) 

#define ICC_ProRes_VideoEncoder_get_TimeBase(This,p)	\
    ( (This)->lpVtbl -> get_TimeBase(This,p) ) 

#define ICC_ProRes_VideoEncoder_put_TimeBase(This,p)	\
    ( (This)->lpVtbl -> put_TimeBase(This,p) ) 

#define ICC_ProRes_VideoEncoder_get_BitRate(This,p)	\
    ( (This)->lpVtbl -> get_BitRate(This,p) ) 

#define ICC_ProRes_VideoEncoder_get_StreamInfo(This,p)	\
    ( (This)->lpVtbl -> get_StreamInfo(This,p) ) 

#define ICC_ProRes_VideoEncoder_get_IsDataReady(This,p)	\
    ( (This)->lpVtbl -> get_IsDataReady(This,p) ) 

#define ICC_ProRes_VideoEncoder_get_DataInfo(This,s)	\
    ( (This)->lpVtbl -> get_DataInfo(This,s) ) 

#define ICC_ProRes_VideoEncoder_get_OutputCallback(This,p)	\
    ( (This)->lpVtbl -> get_OutputCallback(This,p) ) 

#define ICC_ProRes_VideoEncoder_put_OutputCallback(This,p)	\
    ( (This)->lpVtbl -> put_OutputCallback(This,p) ) 


#define ICC_ProRes_VideoEncoder_get_DataSize(This,__MIDL__ICC_Encoder0000)	\
    ( (This)->lpVtbl -> get_DataSize(This,__MIDL__ICC_Encoder0000) ) 

#define ICC_ProRes_VideoEncoder_GetData(This,pbData,cbBufSize,pcbRetSize)	\
    ( (This)->lpVtbl -> GetData(This,pbData,cbBufSize,pcbRetSize) ) 


#define ICC_ProRes_VideoEncoder_AddFrame(This,Format,pData,cbSize,stride,pResult)	\
    ( (This)->lpVtbl -> AddFrame(This,Format,pData,cbSize,stride,pResult) ) 

#define ICC_ProRes_VideoEncoder_GetStride(This,fmt,pNumBytes)	\
    ( (This)->lpVtbl -> GetStride(This,fmt,pNumBytes) ) 

#define ICC_ProRes_VideoEncoder_IsFormatSupported(This,fmt,pResult)	\
    ( (This)->lpVtbl -> IsFormatSupported(This,fmt,pResult) ) 

#define ICC_ProRes_VideoEncoder_AddScaleFrame(This,pData,cbSize,pParams,pResult)	\
    ( (This)->lpVtbl -> AddScaleFrame(This,pData,cbSize,pParams,pResult) ) 

#define ICC_ProRes_VideoEncoder_IsScaleAvailable(This,pParams,__MIDL__ICC_VideoEncoder0000)	\
    ( (This)->lpVtbl -> IsScaleAvailable(This,pParams,__MIDL__ICC_VideoEncoder0000) ) 

#define ICC_ProRes_VideoEncoder_GetVideoStreamInfo(This,pDescr)	\
    ( (This)->lpVtbl -> GetVideoStreamInfo(This,pDescr) ) 

#define ICC_ProRes_VideoEncoder_GetVideoFrameInfo(This,pDescr)	\
    ( (This)->lpVtbl -> GetVideoFrameInfo(This,pDescr) ) 


#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_ProRes_VideoEncoder_INTERFACE_DEFINED__ */


#ifndef __ICC_PCM_AudioMapperInputPinSettings_INTERFACE_DEFINED__
#define __ICC_PCM_AudioMapperInputPinSettings_INTERFACE_DEFINED__

/* interface ICC_PCM_AudioMapperInputPinSettings */
/* [object][uuid] */ 


EXTERN_C const IID IID_ICC_PCM_AudioMapperInputPinSettings;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("1AF5C7B5-EEDD-46C9-B2F7-B154243AE834")
    ICC_PCM_AudioMapperInputPinSettings : public ICC_Settings
    {
    public:
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_SampleRate( 
            /* [retval][out] */ CC_UINT *p) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_SampleRate( 
            /* [in] */ CC_UINT v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_NumChannels( 
            /* [retval][out] */ CC_UINT *p) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_NumChannels( 
            /* [in] */ CC_UINT v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_BitsPerSample( 
            /* [retval][out] */ CC_UINT *p) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_BitsPerSample( 
            /* [in] */ CC_UINT v) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_PCM_AudioMapperInputPinSettingsVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_PCM_AudioMapperInputPinSettings * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_PCM_AudioMapperInputPinSettings * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_PCM_AudioMapperInputPinSettings * This);
        
        HRESULT ( STDMETHODCALLTYPE *Clear )( 
            ICC_PCM_AudioMapperInputPinSettings * This,
            /* [in] */ LPCSTR strVarName);
        
        HRESULT ( STDMETHODCALLTYPE *Assigned )( 
            ICC_PCM_AudioMapperInputPinSettings * This,
            /* [in] */ LPCSTR strVarName,
            /* [defaultvalue][retval][out] */ CC_BOOL *__MIDL__ICC_Settings0000);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_XML )( 
            ICC_PCM_AudioMapperInputPinSettings * This,
            /* [retval][out] */ CC_STRING *pstrXml);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_XML )( 
            ICC_PCM_AudioMapperInputPinSettings * This,
            /* [in] */ CC_STRING strXml);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_SampleRate )( 
            ICC_PCM_AudioMapperInputPinSettings * This,
            /* [retval][out] */ CC_UINT *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_SampleRate )( 
            ICC_PCM_AudioMapperInputPinSettings * This,
            /* [in] */ CC_UINT v);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_NumChannels )( 
            ICC_PCM_AudioMapperInputPinSettings * This,
            /* [retval][out] */ CC_UINT *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_NumChannels )( 
            ICC_PCM_AudioMapperInputPinSettings * This,
            /* [in] */ CC_UINT v);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_BitsPerSample )( 
            ICC_PCM_AudioMapperInputPinSettings * This,
            /* [retval][out] */ CC_UINT *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_BitsPerSample )( 
            ICC_PCM_AudioMapperInputPinSettings * This,
            /* [in] */ CC_UINT v);
        
        END_INTERFACE
    } ICC_PCM_AudioMapperInputPinSettingsVtbl;

    interface ICC_PCM_AudioMapperInputPinSettings
    {
        CONST_VTBL struct ICC_PCM_AudioMapperInputPinSettingsVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_PCM_AudioMapperInputPinSettings_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_PCM_AudioMapperInputPinSettings_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_PCM_AudioMapperInputPinSettings_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ICC_PCM_AudioMapperInputPinSettings_Clear(This,strVarName)	\
    ( (This)->lpVtbl -> Clear(This,strVarName) ) 

#define ICC_PCM_AudioMapperInputPinSettings_Assigned(This,strVarName,__MIDL__ICC_Settings0000)	\
    ( (This)->lpVtbl -> Assigned(This,strVarName,__MIDL__ICC_Settings0000) ) 

#define ICC_PCM_AudioMapperInputPinSettings_get_XML(This,pstrXml)	\
    ( (This)->lpVtbl -> get_XML(This,pstrXml) ) 

#define ICC_PCM_AudioMapperInputPinSettings_put_XML(This,strXml)	\
    ( (This)->lpVtbl -> put_XML(This,strXml) ) 


#define ICC_PCM_AudioMapperInputPinSettings_get_SampleRate(This,p)	\
    ( (This)->lpVtbl -> get_SampleRate(This,p) ) 

#define ICC_PCM_AudioMapperInputPinSettings_put_SampleRate(This,v)	\
    ( (This)->lpVtbl -> put_SampleRate(This,v) ) 

#define ICC_PCM_AudioMapperInputPinSettings_get_NumChannels(This,p)	\
    ( (This)->lpVtbl -> get_NumChannels(This,p) ) 

#define ICC_PCM_AudioMapperInputPinSettings_put_NumChannels(This,v)	\
    ( (This)->lpVtbl -> put_NumChannels(This,v) ) 

#define ICC_PCM_AudioMapperInputPinSettings_get_BitsPerSample(This,p)	\
    ( (This)->lpVtbl -> get_BitsPerSample(This,p) ) 

#define ICC_PCM_AudioMapperInputPinSettings_put_BitsPerSample(This,v)	\
    ( (This)->lpVtbl -> put_BitsPerSample(This,v) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_PCM_AudioMapperInputPinSettings_INTERFACE_DEFINED__ */


#ifndef __ICC_PCM_AudioMapperSettings_INTERFACE_DEFINED__
#define __ICC_PCM_AudioMapperSettings_INTERFACE_DEFINED__

/* interface ICC_PCM_AudioMapperSettings */
/* [object][uuid] */ 


EXTERN_C const IID IID_ICC_PCM_AudioMapperSettings;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("5737F248-C614-4E80-97F9-0E9040344F06")
    ICC_PCM_AudioMapperSettings : public ICC_Settings
    {
    public:
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_Mapping( 
            /* [retval][out] */ ICC_Collection **p) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_OutputSettings( 
            /* [retval][out] */ ICC_Collection **p) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_PCM_AudioMapperSettingsVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_PCM_AudioMapperSettings * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_PCM_AudioMapperSettings * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_PCM_AudioMapperSettings * This);
        
        HRESULT ( STDMETHODCALLTYPE *Clear )( 
            ICC_PCM_AudioMapperSettings * This,
            /* [in] */ LPCSTR strVarName);
        
        HRESULT ( STDMETHODCALLTYPE *Assigned )( 
            ICC_PCM_AudioMapperSettings * This,
            /* [in] */ LPCSTR strVarName,
            /* [defaultvalue][retval][out] */ CC_BOOL *__MIDL__ICC_Settings0000);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_XML )( 
            ICC_PCM_AudioMapperSettings * This,
            /* [retval][out] */ CC_STRING *pstrXml);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_XML )( 
            ICC_PCM_AudioMapperSettings * This,
            /* [in] */ CC_STRING strXml);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Mapping )( 
            ICC_PCM_AudioMapperSettings * This,
            /* [retval][out] */ ICC_Collection **p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_OutputSettings )( 
            ICC_PCM_AudioMapperSettings * This,
            /* [retval][out] */ ICC_Collection **p);
        
        END_INTERFACE
    } ICC_PCM_AudioMapperSettingsVtbl;

    interface ICC_PCM_AudioMapperSettings
    {
        CONST_VTBL struct ICC_PCM_AudioMapperSettingsVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_PCM_AudioMapperSettings_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_PCM_AudioMapperSettings_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_PCM_AudioMapperSettings_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ICC_PCM_AudioMapperSettings_Clear(This,strVarName)	\
    ( (This)->lpVtbl -> Clear(This,strVarName) ) 

#define ICC_PCM_AudioMapperSettings_Assigned(This,strVarName,__MIDL__ICC_Settings0000)	\
    ( (This)->lpVtbl -> Assigned(This,strVarName,__MIDL__ICC_Settings0000) ) 

#define ICC_PCM_AudioMapperSettings_get_XML(This,pstrXml)	\
    ( (This)->lpVtbl -> get_XML(This,pstrXml) ) 

#define ICC_PCM_AudioMapperSettings_put_XML(This,strXml)	\
    ( (This)->lpVtbl -> put_XML(This,strXml) ) 


#define ICC_PCM_AudioMapperSettings_get_Mapping(This,p)	\
    ( (This)->lpVtbl -> get_Mapping(This,p) ) 

#define ICC_PCM_AudioMapperSettings_get_OutputSettings(This,p)	\
    ( (This)->lpVtbl -> get_OutputSettings(This,p) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_PCM_AudioMapperSettings_INTERFACE_DEFINED__ */


#ifndef __ICC_PCM_AudioMapperLinkSettings_INTERFACE_DEFINED__
#define __ICC_PCM_AudioMapperLinkSettings_INTERFACE_DEFINED__

/* interface ICC_PCM_AudioMapperLinkSettings */
/* [object][uuid] */ 


EXTERN_C const IID IID_ICC_PCM_AudioMapperLinkSettings;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("4F69127A-5051-4B88-98E3-69C4AA00FDDA")
    ICC_PCM_AudioMapperLinkSettings : public ICC_Settings
    {
    public:
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_SourceTrackNumber( 
            /* [retval][out] */ CC_UINT *p) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_SourceTrackNumber( 
            /* [in] */ CC_UINT v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_SourceChannelNumber( 
            /* [retval][out] */ CC_UINT *p) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_SourceChannelNumber( 
            /* [in] */ CC_UINT v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_TargetTrackNumber( 
            /* [retval][out] */ CC_UINT *p) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_TargetTrackNumber( 
            /* [in] */ CC_UINT v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_TargetChannelNumber( 
            /* [retval][out] */ CC_UINT *p) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_TargetChannelNumber( 
            /* [in] */ CC_UINT v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_Multiplier( 
            /* [retval][out] */ CC_FLOAT *p) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_Multiplier( 
            /* [in] */ CC_FLOAT v) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_PCM_AudioMapperLinkSettingsVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_PCM_AudioMapperLinkSettings * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_PCM_AudioMapperLinkSettings * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_PCM_AudioMapperLinkSettings * This);
        
        HRESULT ( STDMETHODCALLTYPE *Clear )( 
            ICC_PCM_AudioMapperLinkSettings * This,
            /* [in] */ LPCSTR strVarName);
        
        HRESULT ( STDMETHODCALLTYPE *Assigned )( 
            ICC_PCM_AudioMapperLinkSettings * This,
            /* [in] */ LPCSTR strVarName,
            /* [defaultvalue][retval][out] */ CC_BOOL *__MIDL__ICC_Settings0000);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_XML )( 
            ICC_PCM_AudioMapperLinkSettings * This,
            /* [retval][out] */ CC_STRING *pstrXml);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_XML )( 
            ICC_PCM_AudioMapperLinkSettings * This,
            /* [in] */ CC_STRING strXml);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_SourceTrackNumber )( 
            ICC_PCM_AudioMapperLinkSettings * This,
            /* [retval][out] */ CC_UINT *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_SourceTrackNumber )( 
            ICC_PCM_AudioMapperLinkSettings * This,
            /* [in] */ CC_UINT v);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_SourceChannelNumber )( 
            ICC_PCM_AudioMapperLinkSettings * This,
            /* [retval][out] */ CC_UINT *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_SourceChannelNumber )( 
            ICC_PCM_AudioMapperLinkSettings * This,
            /* [in] */ CC_UINT v);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_TargetTrackNumber )( 
            ICC_PCM_AudioMapperLinkSettings * This,
            /* [retval][out] */ CC_UINT *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_TargetTrackNumber )( 
            ICC_PCM_AudioMapperLinkSettings * This,
            /* [in] */ CC_UINT v);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_TargetChannelNumber )( 
            ICC_PCM_AudioMapperLinkSettings * This,
            /* [retval][out] */ CC_UINT *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_TargetChannelNumber )( 
            ICC_PCM_AudioMapperLinkSettings * This,
            /* [in] */ CC_UINT v);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Multiplier )( 
            ICC_PCM_AudioMapperLinkSettings * This,
            /* [retval][out] */ CC_FLOAT *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_Multiplier )( 
            ICC_PCM_AudioMapperLinkSettings * This,
            /* [in] */ CC_FLOAT v);
        
        END_INTERFACE
    } ICC_PCM_AudioMapperLinkSettingsVtbl;

    interface ICC_PCM_AudioMapperLinkSettings
    {
        CONST_VTBL struct ICC_PCM_AudioMapperLinkSettingsVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_PCM_AudioMapperLinkSettings_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_PCM_AudioMapperLinkSettings_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_PCM_AudioMapperLinkSettings_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ICC_PCM_AudioMapperLinkSettings_Clear(This,strVarName)	\
    ( (This)->lpVtbl -> Clear(This,strVarName) ) 

#define ICC_PCM_AudioMapperLinkSettings_Assigned(This,strVarName,__MIDL__ICC_Settings0000)	\
    ( (This)->lpVtbl -> Assigned(This,strVarName,__MIDL__ICC_Settings0000) ) 

#define ICC_PCM_AudioMapperLinkSettings_get_XML(This,pstrXml)	\
    ( (This)->lpVtbl -> get_XML(This,pstrXml) ) 

#define ICC_PCM_AudioMapperLinkSettings_put_XML(This,strXml)	\
    ( (This)->lpVtbl -> put_XML(This,strXml) ) 


#define ICC_PCM_AudioMapperLinkSettings_get_SourceTrackNumber(This,p)	\
    ( (This)->lpVtbl -> get_SourceTrackNumber(This,p) ) 

#define ICC_PCM_AudioMapperLinkSettings_put_SourceTrackNumber(This,v)	\
    ( (This)->lpVtbl -> put_SourceTrackNumber(This,v) ) 

#define ICC_PCM_AudioMapperLinkSettings_get_SourceChannelNumber(This,p)	\
    ( (This)->lpVtbl -> get_SourceChannelNumber(This,p) ) 

#define ICC_PCM_AudioMapperLinkSettings_put_SourceChannelNumber(This,v)	\
    ( (This)->lpVtbl -> put_SourceChannelNumber(This,v) ) 

#define ICC_PCM_AudioMapperLinkSettings_get_TargetTrackNumber(This,p)	\
    ( (This)->lpVtbl -> get_TargetTrackNumber(This,p) ) 

#define ICC_PCM_AudioMapperLinkSettings_put_TargetTrackNumber(This,v)	\
    ( (This)->lpVtbl -> put_TargetTrackNumber(This,v) ) 

#define ICC_PCM_AudioMapperLinkSettings_get_TargetChannelNumber(This,p)	\
    ( (This)->lpVtbl -> get_TargetChannelNumber(This,p) ) 

#define ICC_PCM_AudioMapperLinkSettings_put_TargetChannelNumber(This,v)	\
    ( (This)->lpVtbl -> put_TargetChannelNumber(This,v) ) 

#define ICC_PCM_AudioMapperLinkSettings_get_Multiplier(This,p)	\
    ( (This)->lpVtbl -> get_Multiplier(This,p) ) 

#define ICC_PCM_AudioMapperLinkSettings_put_Multiplier(This,v)	\
    ( (This)->lpVtbl -> put_Multiplier(This,v) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_PCM_AudioMapperLinkSettings_INTERFACE_DEFINED__ */


#ifndef __ICC_PCM_AudioMapperOutputStreamSettings_INTERFACE_DEFINED__
#define __ICC_PCM_AudioMapperOutputStreamSettings_INTERFACE_DEFINED__

/* interface ICC_PCM_AudioMapperOutputStreamSettings */
/* [object][uuid] */ 


EXTERN_C const IID IID_ICC_PCM_AudioMapperOutputStreamSettings;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("1BB3CF9F-7572-47BA-87FD-4F157F25E58E")
    ICC_PCM_AudioMapperOutputStreamSettings : public ICC_Settings
    {
    public:
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_NumChannels( 
            /* [retval][out] */ CC_UINT *p) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_NumChannels( 
            /* [in] */ CC_UINT v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_BitsPerSample( 
            /* [retval][out] */ CC_UINT *p) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_BitsPerSample( 
            /* [in] */ CC_UINT v) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_PCM_AudioMapperOutputStreamSettingsVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_PCM_AudioMapperOutputStreamSettings * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_PCM_AudioMapperOutputStreamSettings * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_PCM_AudioMapperOutputStreamSettings * This);
        
        HRESULT ( STDMETHODCALLTYPE *Clear )( 
            ICC_PCM_AudioMapperOutputStreamSettings * This,
            /* [in] */ LPCSTR strVarName);
        
        HRESULT ( STDMETHODCALLTYPE *Assigned )( 
            ICC_PCM_AudioMapperOutputStreamSettings * This,
            /* [in] */ LPCSTR strVarName,
            /* [defaultvalue][retval][out] */ CC_BOOL *__MIDL__ICC_Settings0000);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_XML )( 
            ICC_PCM_AudioMapperOutputStreamSettings * This,
            /* [retval][out] */ CC_STRING *pstrXml);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_XML )( 
            ICC_PCM_AudioMapperOutputStreamSettings * This,
            /* [in] */ CC_STRING strXml);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_NumChannels )( 
            ICC_PCM_AudioMapperOutputStreamSettings * This,
            /* [retval][out] */ CC_UINT *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_NumChannels )( 
            ICC_PCM_AudioMapperOutputStreamSettings * This,
            /* [in] */ CC_UINT v);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_BitsPerSample )( 
            ICC_PCM_AudioMapperOutputStreamSettings * This,
            /* [retval][out] */ CC_UINT *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_BitsPerSample )( 
            ICC_PCM_AudioMapperOutputStreamSettings * This,
            /* [in] */ CC_UINT v);
        
        END_INTERFACE
    } ICC_PCM_AudioMapperOutputStreamSettingsVtbl;

    interface ICC_PCM_AudioMapperOutputStreamSettings
    {
        CONST_VTBL struct ICC_PCM_AudioMapperOutputStreamSettingsVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_PCM_AudioMapperOutputStreamSettings_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_PCM_AudioMapperOutputStreamSettings_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_PCM_AudioMapperOutputStreamSettings_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ICC_PCM_AudioMapperOutputStreamSettings_Clear(This,strVarName)	\
    ( (This)->lpVtbl -> Clear(This,strVarName) ) 

#define ICC_PCM_AudioMapperOutputStreamSettings_Assigned(This,strVarName,__MIDL__ICC_Settings0000)	\
    ( (This)->lpVtbl -> Assigned(This,strVarName,__MIDL__ICC_Settings0000) ) 

#define ICC_PCM_AudioMapperOutputStreamSettings_get_XML(This,pstrXml)	\
    ( (This)->lpVtbl -> get_XML(This,pstrXml) ) 

#define ICC_PCM_AudioMapperOutputStreamSettings_put_XML(This,strXml)	\
    ( (This)->lpVtbl -> put_XML(This,strXml) ) 


#define ICC_PCM_AudioMapperOutputStreamSettings_get_NumChannels(This,p)	\
    ( (This)->lpVtbl -> get_NumChannels(This,p) ) 

#define ICC_PCM_AudioMapperOutputStreamSettings_put_NumChannels(This,v)	\
    ( (This)->lpVtbl -> put_NumChannels(This,v) ) 

#define ICC_PCM_AudioMapperOutputStreamSettings_get_BitsPerSample(This,p)	\
    ( (This)->lpVtbl -> get_BitsPerSample(This,p) ) 

#define ICC_PCM_AudioMapperOutputStreamSettings_put_BitsPerSample(This,v)	\
    ( (This)->lpVtbl -> put_BitsPerSample(This,v) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_PCM_AudioMapperOutputStreamSettings_INTERFACE_DEFINED__ */


#ifndef __ICC_PCM_AudioMapperOutputStreamInfo_INTERFACE_DEFINED__
#define __ICC_PCM_AudioMapperOutputStreamInfo_INTERFACE_DEFINED__

/* interface ICC_PCM_AudioMapperOutputStreamInfo */
/* [object][uuid] */ 


EXTERN_C const IID IID_ICC_PCM_AudioMapperOutputStreamInfo;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("67F1BBC7-8A92-45C8-9B44-5DAE91F6BA58")
    ICC_PCM_AudioMapperOutputStreamInfo : public ICC_Settings
    {
    public:
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_TrackID( 
            /* [retval][out] */ CC_UINT *p) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_TrackID( 
            /* [in] */ CC_UINT v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_SampleRate( 
            /* [retval][out] */ CC_UINT *p) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_SampleRate( 
            /* [in] */ CC_UINT v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_NumChannels( 
            /* [retval][out] */ CC_UINT *p) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_NumChannels( 
            /* [in] */ CC_UINT v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_BitsPerSample( 
            /* [retval][out] */ CC_UINT *p) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_BitsPerSample( 
            /* [in] */ CC_UINT v) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_PCM_AudioMapperOutputStreamInfoVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_PCM_AudioMapperOutputStreamInfo * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_PCM_AudioMapperOutputStreamInfo * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_PCM_AudioMapperOutputStreamInfo * This);
        
        HRESULT ( STDMETHODCALLTYPE *Clear )( 
            ICC_PCM_AudioMapperOutputStreamInfo * This,
            /* [in] */ LPCSTR strVarName);
        
        HRESULT ( STDMETHODCALLTYPE *Assigned )( 
            ICC_PCM_AudioMapperOutputStreamInfo * This,
            /* [in] */ LPCSTR strVarName,
            /* [defaultvalue][retval][out] */ CC_BOOL *__MIDL__ICC_Settings0000);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_XML )( 
            ICC_PCM_AudioMapperOutputStreamInfo * This,
            /* [retval][out] */ CC_STRING *pstrXml);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_XML )( 
            ICC_PCM_AudioMapperOutputStreamInfo * This,
            /* [in] */ CC_STRING strXml);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_TrackID )( 
            ICC_PCM_AudioMapperOutputStreamInfo * This,
            /* [retval][out] */ CC_UINT *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_TrackID )( 
            ICC_PCM_AudioMapperOutputStreamInfo * This,
            /* [in] */ CC_UINT v);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_SampleRate )( 
            ICC_PCM_AudioMapperOutputStreamInfo * This,
            /* [retval][out] */ CC_UINT *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_SampleRate )( 
            ICC_PCM_AudioMapperOutputStreamInfo * This,
            /* [in] */ CC_UINT v);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_NumChannels )( 
            ICC_PCM_AudioMapperOutputStreamInfo * This,
            /* [retval][out] */ CC_UINT *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_NumChannels )( 
            ICC_PCM_AudioMapperOutputStreamInfo * This,
            /* [in] */ CC_UINT v);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_BitsPerSample )( 
            ICC_PCM_AudioMapperOutputStreamInfo * This,
            /* [retval][out] */ CC_UINT *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_BitsPerSample )( 
            ICC_PCM_AudioMapperOutputStreamInfo * This,
            /* [in] */ CC_UINT v);
        
        END_INTERFACE
    } ICC_PCM_AudioMapperOutputStreamInfoVtbl;

    interface ICC_PCM_AudioMapperOutputStreamInfo
    {
        CONST_VTBL struct ICC_PCM_AudioMapperOutputStreamInfoVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_PCM_AudioMapperOutputStreamInfo_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_PCM_AudioMapperOutputStreamInfo_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_PCM_AudioMapperOutputStreamInfo_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ICC_PCM_AudioMapperOutputStreamInfo_Clear(This,strVarName)	\
    ( (This)->lpVtbl -> Clear(This,strVarName) ) 

#define ICC_PCM_AudioMapperOutputStreamInfo_Assigned(This,strVarName,__MIDL__ICC_Settings0000)	\
    ( (This)->lpVtbl -> Assigned(This,strVarName,__MIDL__ICC_Settings0000) ) 

#define ICC_PCM_AudioMapperOutputStreamInfo_get_XML(This,pstrXml)	\
    ( (This)->lpVtbl -> get_XML(This,pstrXml) ) 

#define ICC_PCM_AudioMapperOutputStreamInfo_put_XML(This,strXml)	\
    ( (This)->lpVtbl -> put_XML(This,strXml) ) 


#define ICC_PCM_AudioMapperOutputStreamInfo_get_TrackID(This,p)	\
    ( (This)->lpVtbl -> get_TrackID(This,p) ) 

#define ICC_PCM_AudioMapperOutputStreamInfo_put_TrackID(This,v)	\
    ( (This)->lpVtbl -> put_TrackID(This,v) ) 

#define ICC_PCM_AudioMapperOutputStreamInfo_get_SampleRate(This,p)	\
    ( (This)->lpVtbl -> get_SampleRate(This,p) ) 

#define ICC_PCM_AudioMapperOutputStreamInfo_put_SampleRate(This,v)	\
    ( (This)->lpVtbl -> put_SampleRate(This,v) ) 

#define ICC_PCM_AudioMapperOutputStreamInfo_get_NumChannels(This,p)	\
    ( (This)->lpVtbl -> get_NumChannels(This,p) ) 

#define ICC_PCM_AudioMapperOutputStreamInfo_put_NumChannels(This,v)	\
    ( (This)->lpVtbl -> put_NumChannels(This,v) ) 

#define ICC_PCM_AudioMapperOutputStreamInfo_get_BitsPerSample(This,p)	\
    ( (This)->lpVtbl -> get_BitsPerSample(This,p) ) 

#define ICC_PCM_AudioMapperOutputStreamInfo_put_BitsPerSample(This,v)	\
    ( (This)->lpVtbl -> put_BitsPerSample(This,v) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_PCM_AudioMapperOutputStreamInfo_INTERFACE_DEFINED__ */


#ifndef __ICC_PCM_AudioMapperOutputPin_INTERFACE_DEFINED__
#define __ICC_PCM_AudioMapperOutputPin_INTERFACE_DEFINED__

/* interface ICC_PCM_AudioMapperOutputPin */
/* [object][uuid] */ 


EXTERN_C const IID IID_ICC_PCM_AudioMapperOutputPin;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("4F72B312-1729-4AA0-89F8-0BCC82A711F2")
    ICC_PCM_AudioMapperOutputPin : public IUnknown
    {
    public:
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_TimeBase( 
            /* [retval][out] */ CC_TIMEBASE *p) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_TimeBase( 
            /* [in] */ CC_TIMEBASE v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_BitRate( 
            /* [retval][out] */ CC_BITRATE *p) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_StreamInfo( 
            /* [retval][out] */ ICC_Settings **pp) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_IsDataReady( 
            /* [defaultvalue][retval][out] */ CC_BOOL *p = 0) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_OutputCallback( 
            /* [retval][out] */ IUnknown **pp) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_OutputCallback( 
            /* [in] */ IUnknown *p) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_PCM_AudioMapperOutputPinVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_PCM_AudioMapperOutputPin * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_PCM_AudioMapperOutputPin * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_PCM_AudioMapperOutputPin * This);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_TimeBase )( 
            ICC_PCM_AudioMapperOutputPin * This,
            /* [retval][out] */ CC_TIMEBASE *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_TimeBase )( 
            ICC_PCM_AudioMapperOutputPin * This,
            /* [in] */ CC_TIMEBASE v);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_BitRate )( 
            ICC_PCM_AudioMapperOutputPin * This,
            /* [retval][out] */ CC_BITRATE *p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_StreamInfo )( 
            ICC_PCM_AudioMapperOutputPin * This,
            /* [retval][out] */ ICC_Settings **pp);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_IsDataReady )( 
            ICC_PCM_AudioMapperOutputPin * This,
            /* [defaultvalue][retval][out] */ CC_BOOL *p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_OutputCallback )( 
            ICC_PCM_AudioMapperOutputPin * This,
            /* [retval][out] */ IUnknown **pp);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_OutputCallback )( 
            ICC_PCM_AudioMapperOutputPin * This,
            /* [in] */ IUnknown *p);
        
        END_INTERFACE
    } ICC_PCM_AudioMapperOutputPinVtbl;

    interface ICC_PCM_AudioMapperOutputPin
    {
        CONST_VTBL struct ICC_PCM_AudioMapperOutputPinVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_PCM_AudioMapperOutputPin_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_PCM_AudioMapperOutputPin_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_PCM_AudioMapperOutputPin_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ICC_PCM_AudioMapperOutputPin_get_TimeBase(This,p)	\
    ( (This)->lpVtbl -> get_TimeBase(This,p) ) 

#define ICC_PCM_AudioMapperOutputPin_put_TimeBase(This,v)	\
    ( (This)->lpVtbl -> put_TimeBase(This,v) ) 

#define ICC_PCM_AudioMapperOutputPin_get_BitRate(This,p)	\
    ( (This)->lpVtbl -> get_BitRate(This,p) ) 

#define ICC_PCM_AudioMapperOutputPin_get_StreamInfo(This,pp)	\
    ( (This)->lpVtbl -> get_StreamInfo(This,pp) ) 

#define ICC_PCM_AudioMapperOutputPin_get_IsDataReady(This,p)	\
    ( (This)->lpVtbl -> get_IsDataReady(This,p) ) 

#define ICC_PCM_AudioMapperOutputPin_get_OutputCallback(This,pp)	\
    ( (This)->lpVtbl -> get_OutputCallback(This,pp) ) 

#define ICC_PCM_AudioMapperOutputPin_put_OutputCallback(This,p)	\
    ( (This)->lpVtbl -> put_OutputCallback(This,p) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_PCM_AudioMapperOutputPin_INTERFACE_DEFINED__ */


#ifndef __ICC_PCM_AudioMapper_INTERFACE_DEFINED__
#define __ICC_PCM_AudioMapper_INTERFACE_DEFINED__

/* interface ICC_PCM_AudioMapper */
/* [object][uuid] */ 


EXTERN_C const IID IID_ICC_PCM_AudioMapper;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("B00CC29A-06E5-4B7F-97C8-B588F55D523D")
    ICC_PCM_AudioMapper : public IUnknown
    {
    public:
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_PCM_AudioMapperVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_PCM_AudioMapper * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_PCM_AudioMapper * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_PCM_AudioMapper * This);
        
        END_INTERFACE
    } ICC_PCM_AudioMapperVtbl;

    interface ICC_PCM_AudioMapper
    {
        CONST_VTBL struct ICC_PCM_AudioMapperVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_PCM_AudioMapper_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_PCM_AudioMapper_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_PCM_AudioMapper_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_PCM_AudioMapper_INTERFACE_DEFINED__ */


#ifndef __ICC_PCM_AudioMapperProducer_INTERFACE_DEFINED__
#define __ICC_PCM_AudioMapperProducer_INTERFACE_DEFINED__

/* interface ICC_PCM_AudioMapperProducer */
/* [object][uuid] */ 


EXTERN_C const IID IID_ICC_PCM_AudioMapperProducer;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("46525E65-6C11-4437-997A-559881615F75")
    ICC_PCM_AudioMapperProducer : public IUnknown
    {
    public:
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_DataSize( 
            /* [retval][out] */ CC_UINT *v) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE GetData( 
            /* [size_is][out] */ CC_PBYTE pbData,
            /* [in] */ CC_UINT cbBufSize,
            /* [defaultvalue][retval][out] */ CC_UINT *pcbRetSize = 0) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_StreamInfo( 
            /* [retval][out] */ ICC_Settings **pp) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_PCM_AudioMapperProducerVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_PCM_AudioMapperProducer * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_PCM_AudioMapperProducer * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_PCM_AudioMapperProducer * This);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_DataSize )( 
            ICC_PCM_AudioMapperProducer * This,
            /* [retval][out] */ CC_UINT *v);
        
        HRESULT ( STDMETHODCALLTYPE *GetData )( 
            ICC_PCM_AudioMapperProducer * This,
            /* [size_is][out] */ CC_PBYTE pbData,
            /* [in] */ CC_UINT cbBufSize,
            /* [defaultvalue][retval][out] */ CC_UINT *pcbRetSize);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_StreamInfo )( 
            ICC_PCM_AudioMapperProducer * This,
            /* [retval][out] */ ICC_Settings **pp);
        
        END_INTERFACE
    } ICC_PCM_AudioMapperProducerVtbl;

    interface ICC_PCM_AudioMapperProducer
    {
        CONST_VTBL struct ICC_PCM_AudioMapperProducerVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_PCM_AudioMapperProducer_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_PCM_AudioMapperProducer_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_PCM_AudioMapperProducer_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ICC_PCM_AudioMapperProducer_get_DataSize(This,v)	\
    ( (This)->lpVtbl -> get_DataSize(This,v) ) 

#define ICC_PCM_AudioMapperProducer_GetData(This,pbData,cbBufSize,pcbRetSize)	\
    ( (This)->lpVtbl -> GetData(This,pbData,cbBufSize,pcbRetSize) ) 

#define ICC_PCM_AudioMapperProducer_get_StreamInfo(This,pp)	\
    ( (This)->lpVtbl -> get_StreamInfo(This,pp) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_PCM_AudioMapperProducer_INTERFACE_DEFINED__ */


#ifndef __ICC_Audio_Resampler_INTERFACE_DEFINED__
#define __ICC_Audio_Resampler_INTERFACE_DEFINED__

/* interface ICC_Audio_Resampler */
/* [object][uuid] */ 


EXTERN_C const IID IID_ICC_Audio_Resampler;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("5865DF43-F1A5-4CDE-86D3-6D66DE81BCAF")
    ICC_Audio_Resampler : public IUnknown
    {
    public:
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_Audio_ResamplerVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_Audio_Resampler * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_Audio_Resampler * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_Audio_Resampler * This);
        
        END_INTERFACE
    } ICC_Audio_ResamplerVtbl;

    interface ICC_Audio_Resampler
    {
        CONST_VTBL struct ICC_Audio_ResamplerVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_Audio_Resampler_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_Audio_Resampler_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_Audio_Resampler_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_Audio_Resampler_INTERFACE_DEFINED__ */


/* interface __MIDL_itf_Cinecoder2EPlugin2ECodecs_0000_0020 */
/* [local] */ 

typedef /* [v1_enum] */ 
enum CC_AUDIO_RESAMPLER_CONVERTER_TYPE
    {
        CC_SINC_BEST_QUALITY	= 0,
        CC_SINC_MEDIUM_QUALITY	= ( CC_SINC_BEST_QUALITY + 1 ) ,
        CC_SINC_FASTEST	= ( CC_SINC_MEDIUM_QUALITY + 1 ) ,
        CC_ZERO_ORDER_HOLD	= ( CC_SINC_FASTEST + 1 ) ,
        CC_LINEAR	= ( CC_ZERO_ORDER_HOLD + 1 ) 
    } 	CC_AUDIO_RESAMPLER_CONVERTER;



extern RPC_IF_HANDLE __MIDL_itf_Cinecoder2EPlugin2ECodecs_0000_0020_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_Cinecoder2EPlugin2ECodecs_0000_0020_v0_0_s_ifspec;

#ifndef __ICC_Audio_Resampler_Settings_INTERFACE_DEFINED__
#define __ICC_Audio_Resampler_Settings_INTERFACE_DEFINED__

/* interface ICC_Audio_Resampler_Settings */
/* [object][uuid] */ 


EXTERN_C const IID IID_ICC_Audio_Resampler_Settings;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("4E5A21DD-0F86-4A6D-8E2E-12498C70EE97")
    ICC_Audio_Resampler_Settings : public ICC_Settings
    {
    public:
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_OutputSampleRate( 
            /* [retval][out] */ CC_UINT *p) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_OutputSampleRate( 
            /* [in] */ CC_UINT v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_ConverterType( 
            /* [retval][out] */ CC_AUDIO_RESAMPLER_CONVERTER *p) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_ConverterType( 
            /* [in] */ CC_AUDIO_RESAMPLER_CONVERTER v) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_Audio_Resampler_SettingsVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_Audio_Resampler_Settings * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_Audio_Resampler_Settings * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_Audio_Resampler_Settings * This);
        
        HRESULT ( STDMETHODCALLTYPE *Clear )( 
            ICC_Audio_Resampler_Settings * This,
            /* [in] */ LPCSTR strVarName);
        
        HRESULT ( STDMETHODCALLTYPE *Assigned )( 
            ICC_Audio_Resampler_Settings * This,
            /* [in] */ LPCSTR strVarName,
            /* [defaultvalue][retval][out] */ CC_BOOL *__MIDL__ICC_Settings0000);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_XML )( 
            ICC_Audio_Resampler_Settings * This,
            /* [retval][out] */ CC_STRING *pstrXml);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_XML )( 
            ICC_Audio_Resampler_Settings * This,
            /* [in] */ CC_STRING strXml);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_OutputSampleRate )( 
            ICC_Audio_Resampler_Settings * This,
            /* [retval][out] */ CC_UINT *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_OutputSampleRate )( 
            ICC_Audio_Resampler_Settings * This,
            /* [in] */ CC_UINT v);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_ConverterType )( 
            ICC_Audio_Resampler_Settings * This,
            /* [retval][out] */ CC_AUDIO_RESAMPLER_CONVERTER *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_ConverterType )( 
            ICC_Audio_Resampler_Settings * This,
            /* [in] */ CC_AUDIO_RESAMPLER_CONVERTER v);
        
        END_INTERFACE
    } ICC_Audio_Resampler_SettingsVtbl;

    interface ICC_Audio_Resampler_Settings
    {
        CONST_VTBL struct ICC_Audio_Resampler_SettingsVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_Audio_Resampler_Settings_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_Audio_Resampler_Settings_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_Audio_Resampler_Settings_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ICC_Audio_Resampler_Settings_Clear(This,strVarName)	\
    ( (This)->lpVtbl -> Clear(This,strVarName) ) 

#define ICC_Audio_Resampler_Settings_Assigned(This,strVarName,__MIDL__ICC_Settings0000)	\
    ( (This)->lpVtbl -> Assigned(This,strVarName,__MIDL__ICC_Settings0000) ) 

#define ICC_Audio_Resampler_Settings_get_XML(This,pstrXml)	\
    ( (This)->lpVtbl -> get_XML(This,pstrXml) ) 

#define ICC_Audio_Resampler_Settings_put_XML(This,strXml)	\
    ( (This)->lpVtbl -> put_XML(This,strXml) ) 


#define ICC_Audio_Resampler_Settings_get_OutputSampleRate(This,p)	\
    ( (This)->lpVtbl -> get_OutputSampleRate(This,p) ) 

#define ICC_Audio_Resampler_Settings_put_OutputSampleRate(This,v)	\
    ( (This)->lpVtbl -> put_OutputSampleRate(This,v) ) 

#define ICC_Audio_Resampler_Settings_get_ConverterType(This,p)	\
    ( (This)->lpVtbl -> get_ConverterType(This,p) ) 

#define ICC_Audio_Resampler_Settings_put_ConverterType(This,v)	\
    ( (This)->lpVtbl -> put_ConverterType(This,v) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_Audio_Resampler_Settings_INTERFACE_DEFINED__ */


#ifndef __ICC_Audio_Resampler_OutputPin_INTERFACE_DEFINED__
#define __ICC_Audio_Resampler_OutputPin_INTERFACE_DEFINED__

/* interface ICC_Audio_Resampler_OutputPin */
/* [object][uuid] */ 


EXTERN_C const IID IID_ICC_Audio_Resampler_OutputPin;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("4AFD0EC8-1B58-4DCD-BF9E-5691EEC51B58")
    ICC_Audio_Resampler_OutputPin : public IUnknown
    {
    public:
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_OutputCallback( 
            /* [retval][out] */ IUnknown **pp) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_OutputCallback( 
            /* [in] */ IUnknown *p) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_SampleRate( 
            /* [retval][out] */ CC_UINT *p) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_NumChannels( 
            /* [retval][out] */ CC_UINT *p) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_BitsPerSample( 
            /* [retval][out] */ CC_UINT *p) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_Audio_Resampler_OutputPinVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_Audio_Resampler_OutputPin * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_Audio_Resampler_OutputPin * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_Audio_Resampler_OutputPin * This);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_OutputCallback )( 
            ICC_Audio_Resampler_OutputPin * This,
            /* [retval][out] */ IUnknown **pp);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_OutputCallback )( 
            ICC_Audio_Resampler_OutputPin * This,
            /* [in] */ IUnknown *p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_SampleRate )( 
            ICC_Audio_Resampler_OutputPin * This,
            /* [retval][out] */ CC_UINT *p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_NumChannels )( 
            ICC_Audio_Resampler_OutputPin * This,
            /* [retval][out] */ CC_UINT *p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_BitsPerSample )( 
            ICC_Audio_Resampler_OutputPin * This,
            /* [retval][out] */ CC_UINT *p);
        
        END_INTERFACE
    } ICC_Audio_Resampler_OutputPinVtbl;

    interface ICC_Audio_Resampler_OutputPin
    {
        CONST_VTBL struct ICC_Audio_Resampler_OutputPinVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_Audio_Resampler_OutputPin_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_Audio_Resampler_OutputPin_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_Audio_Resampler_OutputPin_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ICC_Audio_Resampler_OutputPin_get_OutputCallback(This,pp)	\
    ( (This)->lpVtbl -> get_OutputCallback(This,pp) ) 

#define ICC_Audio_Resampler_OutputPin_put_OutputCallback(This,p)	\
    ( (This)->lpVtbl -> put_OutputCallback(This,p) ) 

#define ICC_Audio_Resampler_OutputPin_get_SampleRate(This,p)	\
    ( (This)->lpVtbl -> get_SampleRate(This,p) ) 

#define ICC_Audio_Resampler_OutputPin_get_NumChannels(This,p)	\
    ( (This)->lpVtbl -> get_NumChannels(This,p) ) 

#define ICC_Audio_Resampler_OutputPin_get_BitsPerSample(This,p)	\
    ( (This)->lpVtbl -> get_BitsPerSample(This,p) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_Audio_Resampler_OutputPin_INTERFACE_DEFINED__ */


#ifndef __ICC_Audio_Resampler_OutputPin_Internal_INTERFACE_DEFINED__
#define __ICC_Audio_Resampler_OutputPin_Internal_INTERFACE_DEFINED__

/* interface ICC_Audio_Resampler_OutputPin_Internal */
/* [object][uuid] */ 


EXTERN_C const IID IID_ICC_Audio_Resampler_OutputPin_Internal;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("6A6F9C13-0445-422C-9D16-03B146448D5B")
    ICC_Audio_Resampler_OutputPin_Internal : public IUnknown
    {
    public:
        virtual HRESULT STDMETHODCALLTYPE PushData( 
            CC_PCBYTE pbData,
            DWORD cbSize,
            DWORD cbOffset,
            CC_TIME pts) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_Audio_Resampler_OutputPin_InternalVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_Audio_Resampler_OutputPin_Internal * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_Audio_Resampler_OutputPin_Internal * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_Audio_Resampler_OutputPin_Internal * This);
        
        HRESULT ( STDMETHODCALLTYPE *PushData )( 
            ICC_Audio_Resampler_OutputPin_Internal * This,
            CC_PCBYTE pbData,
            DWORD cbSize,
            DWORD cbOffset,
            CC_TIME pts);
        
        END_INTERFACE
    } ICC_Audio_Resampler_OutputPin_InternalVtbl;

    interface ICC_Audio_Resampler_OutputPin_Internal
    {
        CONST_VTBL struct ICC_Audio_Resampler_OutputPin_InternalVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_Audio_Resampler_OutputPin_Internal_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_Audio_Resampler_OutputPin_Internal_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_Audio_Resampler_OutputPin_Internal_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ICC_Audio_Resampler_OutputPin_Internal_PushData(This,pbData,cbSize,cbOffset,pts)	\
    ( (This)->lpVtbl -> PushData(This,pbData,cbSize,cbOffset,pts) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_Audio_Resampler_OutputPin_Internal_INTERFACE_DEFINED__ */


#ifndef __ICC_Audio_Resampler_OutputPin_Producer_INTERFACE_DEFINED__
#define __ICC_Audio_Resampler_OutputPin_Producer_INTERFACE_DEFINED__

/* interface ICC_Audio_Resampler_OutputPin_Producer */
/* [object][uuid] */ 


EXTERN_C const IID IID_ICC_Audio_Resampler_OutputPin_Producer;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("A1DCFCFA-F18F-4E72-B250-7A5FC367D03F")
    ICC_Audio_Resampler_OutputPin_Producer : public IUnknown
    {
    public:
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_DataSize( 
            /* [retval][out] */ CC_UINT *pDataSize) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_StartTime( 
            /* [retval][out] */ LONGLONG *pStartTime) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_Duration( 
            /* [retval][out] */ LONGLONG *pDuration) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE GetData( 
            /* [size_is][out] */ CC_PBYTE pbData,
            /* [in] */ CC_UINT cbBufSize,
            /* [retval][out] */ CC_UINT *pcbRetSize) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_Audio_Resampler_OutputPin_ProducerVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_Audio_Resampler_OutputPin_Producer * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_Audio_Resampler_OutputPin_Producer * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_Audio_Resampler_OutputPin_Producer * This);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_DataSize )( 
            ICC_Audio_Resampler_OutputPin_Producer * This,
            /* [retval][out] */ CC_UINT *pDataSize);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_StartTime )( 
            ICC_Audio_Resampler_OutputPin_Producer * This,
            /* [retval][out] */ LONGLONG *pStartTime);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Duration )( 
            ICC_Audio_Resampler_OutputPin_Producer * This,
            /* [retval][out] */ LONGLONG *pDuration);
        
        HRESULT ( STDMETHODCALLTYPE *GetData )( 
            ICC_Audio_Resampler_OutputPin_Producer * This,
            /* [size_is][out] */ CC_PBYTE pbData,
            /* [in] */ CC_UINT cbBufSize,
            /* [retval][out] */ CC_UINT *pcbRetSize);
        
        END_INTERFACE
    } ICC_Audio_Resampler_OutputPin_ProducerVtbl;

    interface ICC_Audio_Resampler_OutputPin_Producer
    {
        CONST_VTBL struct ICC_Audio_Resampler_OutputPin_ProducerVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_Audio_Resampler_OutputPin_Producer_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_Audio_Resampler_OutputPin_Producer_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_Audio_Resampler_OutputPin_Producer_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ICC_Audio_Resampler_OutputPin_Producer_get_DataSize(This,pDataSize)	\
    ( (This)->lpVtbl -> get_DataSize(This,pDataSize) ) 

#define ICC_Audio_Resampler_OutputPin_Producer_get_StartTime(This,pStartTime)	\
    ( (This)->lpVtbl -> get_StartTime(This,pStartTime) ) 

#define ICC_Audio_Resampler_OutputPin_Producer_get_Duration(This,pDuration)	\
    ( (This)->lpVtbl -> get_Duration(This,pDuration) ) 

#define ICC_Audio_Resampler_OutputPin_Producer_GetData(This,pbData,cbBufSize,pcbRetSize)	\
    ( (This)->lpVtbl -> GetData(This,pbData,cbBufSize,pcbRetSize) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_Audio_Resampler_OutputPin_Producer_INTERFACE_DEFINED__ */


#ifndef __ICC_Audio_Resampler_InputPin_Settings_INTERFACE_DEFINED__
#define __ICC_Audio_Resampler_InputPin_Settings_INTERFACE_DEFINED__

/* interface ICC_Audio_Resampler_InputPin_Settings */
/* [object][uuid] */ 


EXTERN_C const IID IID_ICC_Audio_Resampler_InputPin_Settings;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("63EACA46-C31A-43EC-8771-9296F0F3C10D")
    ICC_Audio_Resampler_InputPin_Settings : public ICC_Settings
    {
    public:
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_SampleRate( 
            /* [retval][out] */ CC_UINT *p) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_SampleRate( 
            /* [in] */ CC_UINT v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_NumChannels( 
            /* [retval][out] */ CC_UINT *p) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_NumChannels( 
            /* [in] */ CC_UINT v) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_BitsPerSample( 
            /* [retval][out] */ CC_UINT *p) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_BitsPerSample( 
            /* [in] */ CC_UINT v) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_Audio_Resampler_InputPin_SettingsVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_Audio_Resampler_InputPin_Settings * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_Audio_Resampler_InputPin_Settings * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_Audio_Resampler_InputPin_Settings * This);
        
        HRESULT ( STDMETHODCALLTYPE *Clear )( 
            ICC_Audio_Resampler_InputPin_Settings * This,
            /* [in] */ LPCSTR strVarName);
        
        HRESULT ( STDMETHODCALLTYPE *Assigned )( 
            ICC_Audio_Resampler_InputPin_Settings * This,
            /* [in] */ LPCSTR strVarName,
            /* [defaultvalue][retval][out] */ CC_BOOL *__MIDL__ICC_Settings0000);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_XML )( 
            ICC_Audio_Resampler_InputPin_Settings * This,
            /* [retval][out] */ CC_STRING *pstrXml);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_XML )( 
            ICC_Audio_Resampler_InputPin_Settings * This,
            /* [in] */ CC_STRING strXml);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_SampleRate )( 
            ICC_Audio_Resampler_InputPin_Settings * This,
            /* [retval][out] */ CC_UINT *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_SampleRate )( 
            ICC_Audio_Resampler_InputPin_Settings * This,
            /* [in] */ CC_UINT v);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_NumChannels )( 
            ICC_Audio_Resampler_InputPin_Settings * This,
            /* [retval][out] */ CC_UINT *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_NumChannels )( 
            ICC_Audio_Resampler_InputPin_Settings * This,
            /* [in] */ CC_UINT v);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_BitsPerSample )( 
            ICC_Audio_Resampler_InputPin_Settings * This,
            /* [retval][out] */ CC_UINT *p);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_BitsPerSample )( 
            ICC_Audio_Resampler_InputPin_Settings * This,
            /* [in] */ CC_UINT v);
        
        END_INTERFACE
    } ICC_Audio_Resampler_InputPin_SettingsVtbl;

    interface ICC_Audio_Resampler_InputPin_Settings
    {
        CONST_VTBL struct ICC_Audio_Resampler_InputPin_SettingsVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_Audio_Resampler_InputPin_Settings_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_Audio_Resampler_InputPin_Settings_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_Audio_Resampler_InputPin_Settings_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ICC_Audio_Resampler_InputPin_Settings_Clear(This,strVarName)	\
    ( (This)->lpVtbl -> Clear(This,strVarName) ) 

#define ICC_Audio_Resampler_InputPin_Settings_Assigned(This,strVarName,__MIDL__ICC_Settings0000)	\
    ( (This)->lpVtbl -> Assigned(This,strVarName,__MIDL__ICC_Settings0000) ) 

#define ICC_Audio_Resampler_InputPin_Settings_get_XML(This,pstrXml)	\
    ( (This)->lpVtbl -> get_XML(This,pstrXml) ) 

#define ICC_Audio_Resampler_InputPin_Settings_put_XML(This,strXml)	\
    ( (This)->lpVtbl -> put_XML(This,strXml) ) 


#define ICC_Audio_Resampler_InputPin_Settings_get_SampleRate(This,p)	\
    ( (This)->lpVtbl -> get_SampleRate(This,p) ) 

#define ICC_Audio_Resampler_InputPin_Settings_put_SampleRate(This,v)	\
    ( (This)->lpVtbl -> put_SampleRate(This,v) ) 

#define ICC_Audio_Resampler_InputPin_Settings_get_NumChannels(This,p)	\
    ( (This)->lpVtbl -> get_NumChannels(This,p) ) 

#define ICC_Audio_Resampler_InputPin_Settings_put_NumChannels(This,v)	\
    ( (This)->lpVtbl -> put_NumChannels(This,v) ) 

#define ICC_Audio_Resampler_InputPin_Settings_get_BitsPerSample(This,p)	\
    ( (This)->lpVtbl -> get_BitsPerSample(This,p) ) 

#define ICC_Audio_Resampler_InputPin_Settings_put_BitsPerSample(This,v)	\
    ( (This)->lpVtbl -> put_BitsPerSample(This,v) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_Audio_Resampler_InputPin_Settings_INTERFACE_DEFINED__ */


#ifndef __ICC_Audio_Resampler_InputPin_INTERFACE_DEFINED__
#define __ICC_Audio_Resampler_InputPin_INTERFACE_DEFINED__

/* interface ICC_Audio_Resampler_InputPin */
/* [object][uuid] */ 


EXTERN_C const IID IID_ICC_Audio_Resampler_InputPin;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("31D0F90D-0299-4CCA-81E0-0A94241B1893")
    ICC_Audio_Resampler_InputPin : public IUnknown
    {
    public:
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_SampleRate( 
            /* [retval][out] */ CC_UINT *p) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_NumChannels( 
            /* [retval][out] */ CC_UINT *p) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_BitsPerSample( 
            /* [retval][out] */ CC_UINT *p) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_Audio_Resampler_InputPinVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_Audio_Resampler_InputPin * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_Audio_Resampler_InputPin * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_Audio_Resampler_InputPin * This);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_SampleRate )( 
            ICC_Audio_Resampler_InputPin * This,
            /* [retval][out] */ CC_UINT *p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_NumChannels )( 
            ICC_Audio_Resampler_InputPin * This,
            /* [retval][out] */ CC_UINT *p);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_BitsPerSample )( 
            ICC_Audio_Resampler_InputPin * This,
            /* [retval][out] */ CC_UINT *p);
        
        END_INTERFACE
    } ICC_Audio_Resampler_InputPinVtbl;

    interface ICC_Audio_Resampler_InputPin
    {
        CONST_VTBL struct ICC_Audio_Resampler_InputPinVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_Audio_Resampler_InputPin_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_Audio_Resampler_InputPin_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_Audio_Resampler_InputPin_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ICC_Audio_Resampler_InputPin_get_SampleRate(This,p)	\
    ( (This)->lpVtbl -> get_SampleRate(This,p) ) 

#define ICC_Audio_Resampler_InputPin_get_NumChannels(This,p)	\
    ( (This)->lpVtbl -> get_NumChannels(This,p) ) 

#define ICC_Audio_Resampler_InputPin_get_BitsPerSample(This,p)	\
    ( (This)->lpVtbl -> get_BitsPerSample(This,p) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_Audio_Resampler_InputPin_INTERFACE_DEFINED__ */


#ifndef __ICC_Audio_Resampler_InputPin_Internal_INTERFACE_DEFINED__
#define __ICC_Audio_Resampler_InputPin_Internal_INTERFACE_DEFINED__

/* interface ICC_Audio_Resampler_InputPin_Internal */
/* [object][uuid] */ 


EXTERN_C const IID IID_ICC_Audio_Resampler_InputPin_Internal;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("8A0DB759-7FC9-41A3-85C9-62ADBFCB56BF")
    ICC_Audio_Resampler_InputPin_Internal : public IUnknown
    {
    public:
        virtual HRESULT STDMETHODCALLTYPE AddOutputPin( 
            ICC_Audio_Resampler_OutputPin *p) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ICC_Audio_Resampler_InputPin_InternalVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ICC_Audio_Resampler_InputPin_Internal * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ICC_Audio_Resampler_InputPin_Internal * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ICC_Audio_Resampler_InputPin_Internal * This);
        
        HRESULT ( STDMETHODCALLTYPE *AddOutputPin )( 
            ICC_Audio_Resampler_InputPin_Internal * This,
            ICC_Audio_Resampler_OutputPin *p);
        
        END_INTERFACE
    } ICC_Audio_Resampler_InputPin_InternalVtbl;

    interface ICC_Audio_Resampler_InputPin_Internal
    {
        CONST_VTBL struct ICC_Audio_Resampler_InputPin_InternalVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ICC_Audio_Resampler_InputPin_Internal_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ICC_Audio_Resampler_InputPin_Internal_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ICC_Audio_Resampler_InputPin_Internal_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ICC_Audio_Resampler_InputPin_Internal_AddOutputPin(This,p)	\
    ( (This)->lpVtbl -> AddOutputPin(This,p) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ICC_Audio_Resampler_InputPin_Internal_INTERFACE_DEFINED__ */



#ifndef __CinecoderPluginCodecs_LIBRARY_DEFINED__
#define __CinecoderPluginCodecs_LIBRARY_DEFINED__

/* library CinecoderPluginCodecs */
/* [uuid] */ 









EXTERN_C const IID LIBID_CinecoderPluginCodecs;

EXTERN_C const CLSID CLSID_CC_DV_VideoEncoder;

#ifdef __cplusplus

class DECLSPEC_UUID("E1221A6B-EAF3-418C-8417-05D891FA474A")
CC_DV_VideoEncoder;
#endif

EXTERN_C const CLSID CLSID_CC_DV_VideoEncoderSettings;

#ifdef __cplusplus

class DECLSPEC_UUID("4D545BBB-247B-4F61-8654-2531EC9B769F")
CC_DV_VideoEncoderSettings;
#endif

EXTERN_C const CLSID CLSID_CC_DV_VideoDecoder;

#ifdef __cplusplus

class DECLSPEC_UUID("7BA735DD-913F-4EF1-B519-F51423AAA1BD")
CC_DV_VideoDecoder;
#endif

EXTERN_C const CLSID CLSID_CC_DV_VideoDecoderSettings;

#ifdef __cplusplus

class DECLSPEC_UUID("ED240DF1-EBF1-49AF-A1AC-B34B1216B01D")
CC_DV_VideoDecoderSettings;
#endif

EXTERN_C const CLSID CLSID_CC_ProRes_VideoDecoder;

#ifdef __cplusplus

class DECLSPEC_UUID("08F5ED66-EB13-4140-9871-34127BEADCF8")
CC_ProRes_VideoDecoder;
#endif

EXTERN_C const CLSID CLSID_CC_ProRes_VideoDecoderSettings;

#ifdef __cplusplus

class DECLSPEC_UUID("14866CE8-E838-48C8-A038-39422062C9DE")
CC_ProRes_VideoDecoderSettings;
#endif

EXTERN_C const CLSID CLSID_CC_ProRes_VideoEncoder;

#ifdef __cplusplus

class DECLSPEC_UUID("5B9F1E3D-A55E-463C-8124-AFEAE2CD8669")
CC_ProRes_VideoEncoder;
#endif

EXTERN_C const CLSID CLSID_CC_ProRes_VideoEncoderSettings;

#ifdef __cplusplus

class DECLSPEC_UUID("3C7C4E7B-521F-4087-9670-AE0411DF4B45")
CC_ProRes_VideoEncoderSettings;
#endif

EXTERN_C const CLSID CLSID_CC_PCM_AudioMapperInputPinSettings;

#ifdef __cplusplus

class DECLSPEC_UUID("AA37D0EE-720A-4A16-A92A-24B634F4F1A7")
CC_PCM_AudioMapperInputPinSettings;
#endif

EXTERN_C const CLSID CLSID_CC_PCM_AudioMapperSettings;

#ifdef __cplusplus

class DECLSPEC_UUID("3EEB082C-9E61-4D59-8372-D28FFF7A6AD7")
CC_PCM_AudioMapperSettings;
#endif

EXTERN_C const CLSID CLSID_CC_PCM_AudioMapperLinkSettings;

#ifdef __cplusplus

class DECLSPEC_UUID("DD2AA956-3F3B-420E-AD8D-22B6B2E9BF9C")
CC_PCM_AudioMapperLinkSettings;
#endif

EXTERN_C const CLSID CLSID_CC_PCM_AudioMapperOutputStreamSettings;

#ifdef __cplusplus

class DECLSPEC_UUID("260C4DF4-26D1-47D4-BB10-BDD4D61B9775")
CC_PCM_AudioMapperOutputStreamSettings;
#endif

EXTERN_C const CLSID CLSID_CC_PCM_AudioMapper;

#ifdef __cplusplus

class DECLSPEC_UUID("9C5A7A5E-8C77-4633-8F1F-319DF978CACE")
CC_PCM_AudioMapper;
#endif

EXTERN_C const CLSID CLSID_CC_Audio_Resampler;

#ifdef __cplusplus

class DECLSPEC_UUID("1CA2D209-8FD1-472C-9643-0685AF61D835")
CC_Audio_Resampler;
#endif

EXTERN_C const CLSID CLSID_CC_Audio_Resampler_Settings;

#ifdef __cplusplus

class DECLSPEC_UUID("DBB2B4B4-3F7B-43CE-8BBD-05000E567C7C")
CC_Audio_Resampler_Settings;
#endif

EXTERN_C const CLSID CLSID_CC_Audio_Resampler_InputPin_Settings;

#ifdef __cplusplus

class DECLSPEC_UUID("3F2F9807-8238-4C52-9878-B1DD97BE4AE6")
CC_Audio_Resampler_InputPin_Settings;
#endif
#endif /* __CinecoderPluginCodecs_LIBRARY_DEFINED__ */

/* Additional Prototypes for ALL interfaces */

/* end of Additional Prototypes */

#ifdef __cplusplus
}
#endif

#endif


