

/* this ALWAYS GENERATED file contains the IIDs and CLSIDs */

/* link this file in with the server and any clients */


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
        const type name = {l,w1,w2,{b1,b2,b3,b4,b5,b6,b7,b8}}

#endif !_MIDL_USE_GUIDDEF_

MIDL_DEFINE_GUID(IID, IID_ICC_DNxHD_VideoEncoder,0x495AE5F1,0xC244,0x42F5,0xB4,0x1C,0x9C,0x2F,0xB8,0x6F,0x97,0xDE);


MIDL_DEFINE_GUID(IID, IID_ICC_DNxHD_VideoStreamInfo,0x449E6E5A,0xBE8C,0x4900,0x89,0x10,0x00,0x83,0xC6,0xE2,0xDD,0x42);


MIDL_DEFINE_GUID(IID, IID_ICC_DNxHD_VideoEncoderSettings,0xA2846808,0x3782,0x4E20,0x87,0x27,0x32,0x85,0x44,0x7C,0x2D,0x73);


MIDL_DEFINE_GUID(IID, IID_ICC_DNX_VideoEncoderSettings,0x19FBA4DA,0x50DD,0x4321,0x88,0x03,0x91,0x50,0xDB,0x15,0x58,0xF1);


MIDL_DEFINE_GUID(IID, IID_ICC_DNX_VideoStreamInfo,0x34B25A9C,0x5F78,0x43CF,0x84,0x4A,0x03,0xEB,0xC3,0x5F,0xB0,0x99);


MIDL_DEFINE_GUID(IID, IID_ICC_DNX_VideoEncoder,0x83A4BEC2,0x19C4,0x493E,0xA8,0xDB,0x07,0xBF,0x62,0xCA,0x62,0xF8);


MIDL_DEFINE_GUID(IID, LIBID_Cinecoder_Plugin_Codecs_DNxHD,0x2e4df2d6,0xa698,0x47fe,0xb5,0x9e,0x05,0x20,0xb1,0x90,0x70,0x2b);


MIDL_DEFINE_GUID(CLSID, CLSID_CC_DNxHD_VideoEncoder,0x31D1E4DA,0x4130,0x497D,0x83,0x36,0x50,0x01,0x6F,0x8E,0xB6,0x52);


MIDL_DEFINE_GUID(CLSID, CLSID_CC_DNxHD_VideoEncoderSettings,0xA9141A3B,0x42BD,0x4B96,0xBD,0xF7,0xF5,0xB4,0x78,0x1E,0x0F,0xA2);


MIDL_DEFINE_GUID(CLSID, CLSID_CC_DNX_VideoEncoderSettings,0x5FFC519A,0xD567,0x4180,0xAC,0x27,0x4C,0x3A,0x8A,0x9B,0x16,0x7D);


MIDL_DEFINE_GUID(CLSID, CLSID_CC_DNX_VideoEncoder,0x0B53BDBD,0x5F4D,0x4E14,0x8B,0x64,0x32,0xCC,0x1A,0xC0,0x18,0x61);

#undef MIDL_DEFINE_GUID

#ifdef __cplusplus
}
#endif



