#include "stdafx.h"
#include "CppUnitTest.h"
#include "CEncoderTest.h"

#include <cinecoder_i.c>

#define COMPANYNAME "cinegy"
#define LICENSEKEY "R5H6F6YDRHG51CEM1SC79SN1U4ZC6T3NYB4KWS54GBFTC7KPM1TJCY4HUF5CC4NG"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace Daniel2NativeTests
{		
	TEST_CLASS(CDaniel2Encoding)
	{
	public:
		
		TEST_METHOD(CreateEncoder)
		{
			CC_VERSION_INFO ver = Cinecoder_GetVersion();
			printf("Cinecoder version %d.%02d.%02d\n", ver.VersionHi, ver.VersionLo, ver.EditionNo);

			HRESULT hr;

			CComPtr<ICC_ClassFactory> pFactory;
			if (FAILED(hr = Cinecoder_CreateClassFactory(&pFactory)))
				Assert::Fail(); // "Cinecoder factory creation error"));

			if (FAILED(hr = pFactory->AssignLicense(COMPANYNAME, LICENSEKEY)))
				Assert::Fail(); //  "AssignLicense error");

			CComPtr<ICC_VideoEncoder> pEncoder;
			if (FAILED(hr = pFactory->CreateInstance(CLSID_CC_DanielVideoEncoder_CUDA, IID_ICC_VideoEncoder, (IUnknown**)&pEncoder)))
				Assert::Fail(); // "Encoder creation error");

			CComPtr<ICC_DanielVideoEncoderSettings_CUDA> pSettings;
			if (FAILED(hr = pFactory->CreateInstance(CLSID_CC_DanielVideoEncoderSettings_CUDA, IID_ICC_DanielVideoEncoderSettings_CUDA, (IUnknown**)&pSettings)))
				Assert::Fail(); //  "Encoder settings creation error");

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

			if (FAILED(hr = pEncoder->Init(pSettings)))
				Assert::Fail(); // "Encoder initialization error");

			ENCODER_PARAMS par = {};
			par.pEncoder = pEncoder;
			//par.InputFileName = argv[1];
			//par.OutputFileName = argv[2];
			par.ColorFormat = CCF_V210;
			par.NumReadThreads = 4;
			par.QueueSize = 16;

			CEncoderTest Test;
			if (FAILED(hr = Test.AssignParameters(par)))
				Assert::Fail(); // "EncoderTest.AssignParameters error");

			LARGE_INTEGER t0, freq;
			QueryPerformanceFrequency(&freq);
			QueryPerformanceCounter(&t0);

			//ENCODER_STATS s0 = {};

			//Test.Run();
		}

	};


}