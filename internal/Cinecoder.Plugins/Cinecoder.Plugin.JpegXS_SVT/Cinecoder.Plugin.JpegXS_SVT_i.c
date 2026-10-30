

/* this ALWAYS GENERATED file contains the IIDs and CLSIDs */

/* link this file in with the server and any clients */


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



#ifdef __cplusplus
extern "C"{
#endif 


#include <rpc.h>
#include <rpcndr.h>

#ifdef _MIDL_USE_GUIDDEF_

#ifndef INITGUID
#define INITGUID
#include <guiddef.h>
#undef INITGUID
#else
#include <guiddef.h>
#endif

#define MIDL_DEFINE_GUID(type,name,l,w1,w2,b1,b2,b3,b4,b5,b6,b7,b8) \
        DEFINE_GUID(name,l,w1,w2,b1,b2,b3,b4,b5,b6,b7,b8)

#else // !_MIDL_USE_GUIDDEF_

#ifndef __IID_DEFINED__
#define __IID_DEFINED__

typedef struct _IID
{
    unsigned long x;
    unsigned short s1;
    unsigned short s2;
    unsigned char  c[8];
} IID;

#endif // __IID_DEFINED__

#ifndef CLSID_DEFINED
#define CLSID_DEFINED
typedef IID CLSID;
#endif // CLSID_DEFINED

#define MIDL_DEFINE_GUID(type,name,l,w1,w2,b1,b2,b3,b4,b5,b6,b7,b8) \
        EXTERN_C __declspec(selectany) const type name = {l,w1,w2,{b1,b2,b3,b4,b5,b6,b7,b8}}

#endif // !_MIDL_USE_GUIDDEF_

MIDL_DEFINE_GUID(IID, IID_ICC_JpegXS_VideoDecoder_SVT,0xbd87be1d,0xf8ef,0x49b1,0xb0,0x92,0xb2,0x97,0xc3,0x79,0xb8,0xef);


MIDL_DEFINE_GUID(IID, IID_ICC_JpegXS_VideoEncoder_SVT,0xe0f876ef,0xacc3,0x4791,0x94,0xdb,0xf9,0x68,0x37,0x10,0xad,0x51);


MIDL_DEFINE_GUID(IID, IID_ICC_JpegXS_VideoEncoderSettings_SVT,0x507da1c5,0x8a4e,0x402f,0x9a,0xdc,0x7f,0x89,0xfe,0x2a,0x64,0xb1);


MIDL_DEFINE_GUID(IID, IID_ICC_JpegXS_VideoStreamInfo_SVT,0x2a1debae,0xb8f9,0x40b2,0xa1,0xec,0xd3,0x45,0x4d,0x89,0xc2,0x62);


MIDL_DEFINE_GUID(IID, IID_ICC_JpegXS_VideoFrameInfo_SVT,0x36286a62,0x6c51,0x4c02,0x81,0x49,0x9b,0x8c,0xf5,0x55,0x79,0x91);


MIDL_DEFINE_GUID(IID, LIBID_Cinecoder_Plugin_JpegXS_SVT,0x7c2d8cbd,0x066f,0x450a,0x94,0x10,0x49,0x87,0x61,0x30,0xa4,0xe4);


MIDL_DEFINE_GUID(CLSID, CLSID_CC_JpegXS_VideoDecoder_SVT,0x55d08907,0xc686,0x4f34,0xa4,0x7a,0x85,0xad,0x28,0x5c,0xf0,0xe7);


MIDL_DEFINE_GUID(CLSID, CLSID_CC_JpegXS_VideoEncoder_SVT,0x601e23e1,0x6ede,0x434d,0xa4,0x3d,0xb6,0x51,0xa6,0xa7,0x5b,0xf1);


MIDL_DEFINE_GUID(CLSID, CLSID_CC_JpegXS_VideoEncoderSettings_SVT,0x051f439c,0x04a8,0x48eb,0xa1,0xc1,0x8f,0xbe,0xf3,0x10,0x5f,0x66);

#undef MIDL_DEFINE_GUID

#ifdef __cplusplus
}
#endif



