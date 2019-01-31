#include "pch.h"
#include "CEncoderTest.h"
#include <Cinecoder_i.c>
#include "../common/cinecoder_license_string.h"

CComPtr<ICC_ClassFactory> p_factory;
CComPtr<ICC_VideoEncoder> p_encoder;
CComPtr<ICC_DanielVideoEncoderSettings> p_settings;

TEST(Daniel2Native,CheckCinecoderVersion)
{
	const auto ver = Cinecoder_GetVersion();
	printf("Cinecoder version %d.%02d.%02d\n", ver.VersionHi, ver.VersionLo, ver.EditionNo);

}

TEST(Daniel2Native,CreateCinecoderFactory)
{
	const auto hr = Cinecoder_CreateClassFactory(&p_factory);
	ASSERT_HRESULT_SUCCEEDED(hr) << "Cinecoder factory creation error";
}


TEST(Daniel2Native,AssignCinecoderLicense)
{
	const auto hr = p_factory->AssignLicense(COMPANYNAME, LICENSEKEY);
	ASSERT_HRESULT_SUCCEEDED(hr) << "AssignLicense error";
}

TEST(Daniel2Native,CreateEncoder)
{
	const auto hr = p_factory->CreateInstance(CLSID_CC_DanielVideoEncoder, IID_ICC_VideoEncoder, reinterpret_cast<IUnknown**>(&p_encoder));
	ASSERT_HRESULT_SUCCEEDED(hr) << "Encoder creation error";
}

TEST(Daniel2Native,CreateSettings)
{
	const auto hr = p_factory->CreateInstance(CLSID_CC_DanielVideoEncoderSettings, IID_ICC_DanielVideoEncoderSettings, reinterpret_cast<IUnknown**>(&p_settings));
	ASSERT_HRESULT_SUCCEEDED(hr) << "Encoder settings creation error";
}

TEST(Daniel2Native, InitEncoder)
{	
	p_settings->put_FrameSize(MK_SIZE(7680, 4320));
#ifdef _WIN32 //32-bit can't handle more than 4K - literally not enough room...
	p_settings->put_FrameSize(MK_SIZE(4096, 2160));
#endif

	p_settings->put_FrameRate(MK_RATIONAL(60000, 1001));
	p_settings->put_InputColorFormat(CCF_V210);
	p_settings->put_ChromaFormat(CC_CHROMA_422);
	p_settings->put_BitDepth(10);
	p_settings->put_RateMode(CC_CBR);
	p_settings->put_BitRate(1000000000);
	p_settings->put_CodingMethod(CC_D2_METHOD_CUDA);
	//p_settings->put_DeviceID(0); //CUDA settings only
	p_settings->put_NumSingleEncoders(4);
	
	const auto hr = p_encoder->Init(p_settings);
	ASSERT_HRESULT_SUCCEEDED(hr) << "Encoder initialization error";
}

TEST(Daniel2Native, AssignEncoderParameters)
{		
	ENCODER_PARAMS par = {};
	par.pEncoder = p_encoder;
	par.ColorFormat = CCF_V210;
	par.NumReadThreads = 4;
	par.QueueSize = 16;

	CEncoderTest test;
	const auto hr = test.AssignParameters(par);
	ASSERT_HRESULT_SUCCEEDED(hr) << "EncoderTest.AssignParameters error";
}
