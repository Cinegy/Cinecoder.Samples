// Daniel2.Encoder.FileReadMT.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "CEncoderTest.h"

#include <cinecoder_i.c>

#define COMPANYNAME "TEST"
#define LICENSEKEY "6JFJYAZJA3KNULY5F8T0A4WFPXBNBR5TU7792XPPC9PAPKTS0JGRDND19EJ1PZYE"

int print_error(int err, const char *str = 0)
{
	if(str) fprintf(stderr, "%s, ", str);

	if(SUCCEEDED(err))
	{
		fprintf(stderr, "code=%08xh\n", err);
	}
	else if(LPCSTR errstr = Cinecoder_GetErrorString(err))
	{
		fprintf(stderr, "code=%08xh (%s)", err, errstr);
	}
	else
	{
		char buf[1024] = {0};
		FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM, 0, err, 0, buf, sizeof(buf), 0);
		fprintf(stderr, "code=%08xh (%s)", err, buf);
	}

	return err;
}

int _tmain(int argc, TCHAR *argv[])
{
	CC_VERSION_INFO ver = Cinecoder_GetVersion();
	printf("Cinecoder version %d.%02d.%02d\n", ver.VersionHi, ver.VersionLo, ver.EditionNo);

	HRESULT hr;

	CComPtr<ICC_ClassFactory> pFactory;
	if (FAILED(hr = Cinecoder_CreateClassFactory(&pFactory)))
		return print_error(hr, "Cinecoder factory creation error");

	if (FAILED(hr = pFactory->AssignLicense(COMPANYNAME, LICENSEKEY)))
		return print_error(hr, "AssignLicense error");

	CComPtr<ICC_VideoEncoder> pEncoder;
	if (FAILED(hr = pFactory->CreateInstance(CLSID_CC_DanielVideoEncoder_CUDA, IID_ICC_VideoEncoder, (IUnknown**)&pEncoder)))
		return print_error(hr, "Encoder creation error");

	CComPtr<ICC_DanielVideoEncoderSettings_CUDA> pSettings;
	if (FAILED(hr = pFactory->CreateInstance(CLSID_CC_DanielVideoEncoderSettings_CUDA, IID_ICC_DanielVideoEncoderSettings_CUDA, (IUnknown**)&pSettings)))
		return print_error(hr, "Encoder settings creation error");

	pSettings->put_FrameSize(MK_SIZE(7680, 4320));
	pSettings->put_FrameRate(MK_RATIONAL(60000, 1001));
	pSettings->put_InputColorFormat(CCF_V210);

	pSettings->put_ChromaFormat(CC_CHROMA_422);
	pSettings->put_BitDepth(10);

	pSettings->put_RateMode(CC_CBR);
	pSettings->put_BitRate(1000000000);
	pSettings->put_CodingMethod(CC_D2_METHOD_CUDA);

	pSettings->put_DeviceID(0);
	pSettings->put_NumSingleEncoders(4);

	if(FAILED(hr = pEncoder->Init(pSettings)))
		return print_error(hr, "Encoder initialization error");

	CEncoderTest Test;

    return 0;
}

