#include "stdafx.h"
#include "DecodeDaniel2.h"

// Cinecoder
#include <Cinecoder_i.c>
#if defined(__WIN32__)
#if (CINECODER_VERSION < 40000)
#include <Cinecoder.Plugin.GpuCodecs_i.c>
#endif
#pragma comment(lib, "windowscodecs.lib") // for IID_IDXGIFactory
#endif
#include "CinecoderErrorHandler.h"
C_CinecoderErrorHandler g_ErrorHandler;

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

DecodeDaniel2::DecodeDaniel2() :
	m_width(3840),
	m_height(2160),
	m_outputImageFormat(IMAGE_FORMAT_RGBA8BIT),
	m_outputBufferFormat(BUFFER_FORMAT_RGBA32),
	m_bProcess(false),
	m_bPause(false),
	m_bDecode(true),
	m_bInitDecoder(false),
	m_bUseCuda(false),
	m_bUseOpenCL(false),
	m_bUseCudaHost(false),
	m_bPutColorFormat(false),
	m_pVideoDec(nullptr),
	m_pMediaReader(nullptr),
	m_strStreamType("Unknown"),
	bIntraFormat(false),
	bCalculatePTS(false),
	m_llDuration(1),
	m_llTimeBase(1),
	m_iNegativePTS(0),
	m_iGpuDevice(0)
{
	m_FrameRate.num = 0;
	m_FrameRate.denom = 1;

	m_AspectRatio.num = 1;
	m_AspectRatio.denom = 1;

#if defined(__WIN32__)
	m_pVideoDecD3D11 = nullptr;
	m_pRender = nullptr;
	m_pCapableAdapter = nullptr;
#endif
}

DecodeDaniel2::~DecodeDaniel2()
{
	StopDecode(); // stop main thread of decode

	DestroyDecoder(); // destroy decoder

	DestroyValues(); // destroy values

	m_file.CloseFile(); // close reading DN2 file
}

int DecodeDaniel2::OpenFile(const char* const filename, ST_VIDEO_DECODER_PARAMS dec_params)
{
	m_bInitDecoder = false;

	m_dec_params = dec_params;

	bool useCuda = m_dec_params.type == VD_TYPE_CUDA ? true : false;
	bool useOpenCL = m_dec_params.type == VD_TYPE_OpenCL ? true : false;

	m_bUseCuda = useCuda;
	m_bUseOpenCL = useOpenCL;
	
	m_filename = filename;

	int res = m_file.OpenFile(filename); // open input DN2 file

	size_t iMaxCountDecoders = std::max((size_t)1, std::min(m_dec_params.max_count_decoders, (size_t)4)); // 1..4

	if (res == 0)
		res = CreateDecoder(); // create decoders

	if (res == 0)
	{
		unsigned char* coded_frame = nullptr;
		size_t coded_frame_size = 0;
		size_t frame_number = 0;
		
		CodedFrame buffer;

		HRESULT hr = S_OK;

		m_eventInitDecoder.Reset();

		for (size_t i = 0; i < 1; i++)
		{
			coded_frame_size = 0;
			res = m_file.ReadFrame(i, buffer.coded_frame, coded_frame_size, frame_number); // get i-coded frame for add decoder and init values after decode first frames

			if (res != 0) continue;

			coded_frame = buffer.coded_frame.GetPtr(); // poiter to i-coded frame
			
			CC_TIME pts = 0; // (frame_number * m_llTimeBase * m_FrameRate.denom) / m_FrameRate.num;

			if (SUCCEEDED(hr)) hr = m_pVideoDec->ProcessData(coded_frame, static_cast<CC_UINT>(coded_frame_size), 0, pts); __check_hr

			if (FAILED(hr)) // add coded frame to decoder
			{
				_assert(0);

				printf("ProcessData failed hr=%d coded_frame_size=%zu coded_frame=%p\n", hr, coded_frame_size, coded_frame);

				hr = m_pVideoDec->Break(CC_FALSE); // break decoder 

				__check_hr

				DestroyDecoder(); // destroy decoder

				return hr;
			}
		}

		if (SUCCEEDED(hr))
			hr = m_pVideoDec->Break(CC_TRUE); // break decoder and flush data from decoder (call DataReady)

		if (!m_eventInitDecoder.Wait(2000) || !m_bInitDecoder) // wait 2 seconds for init decode
		{
			printf("Init decode - failed!\n");
			return -1; // if could not decode the 0-frame until 10 second, return an error
		}

		if (useCuda) // reinit GPU decoder for set TargetColorFormat
		{
			m_bPutColorFormat = true;

			res = DestroyDecoder();

			if (res == 0)
				res = CreateDecoder(); // create decoders
		}

		if (res == 0)
			res = InitValues(); // init values
	}

	if (res == 0)
		m_file.StartPipe(); // if initialize was successful, starting pipeline for reading DN2 file

	if (res == 0) // printing parameters such as: filename, size of frame, format image
	{
		printf("-------------------------------------\n");
		printf("filename      : %s\n", filename);
		printf("stream type   : %s\n", m_strStreamType);
		printf("width x height: %zu x %zu\n", m_width, m_height);
		if (m_dec_params.scale_factor != CC_VDEC_NO_SCALE)
		{
			size_t factor = (2 << (m_dec_params.scale_factor - 1));
			printf("scale factor  : 1/%zd (original size %zu x %zu)\n", factor, m_width * factor, m_height * factor);
		}
		if (strcmp(m_strStreamType, "Daniel") == 0) 
		{
			switch (m_ChromaFormat)
			{
			case CC_CHROMA_422:		printf("format        : CHROMA_422 / %d bits\n", m_BitDepth); break;
			case CC_CHROMA_RGBA:	printf("format        : CHROMA_RGBA / %d bits\n", m_BitDepth); break;
			case CC_CHROMA_RGB:		printf("format        : CHROMA_RGB / %d bits\n", m_BitDepth); break;
			case CC_CHROMA_4444:	printf("format        : CHROMA_4444 / %d bits\n", m_BitDepth); break;
			case CC_CHROMA_444:		printf("format        : CHROMA_444 / %d bits\n", m_BitDepth); break;
			}
		}
		printf("frame rate    : %g\n", (double)m_FrameRate.num / m_FrameRate.denom);
		printf("output format : ");
		switch (m_outputImageFormat)
		{
		case IMAGE_FORMAT_RGBA8BIT:		printf("RGBA 8bit\n"); break;
		case IMAGE_FORMAT_BGRA8BIT:		printf("BGRA 8bit\n"); break;
		case IMAGE_FORMAT_RGBA16BIT:	printf("RGBA 16bit\n"); break;
		case IMAGE_FORMAT_BGRA16BIT:	printf("BGRA 16bit\n"); break;
		case IMAGE_FORMAT_RGB30:		printf("RGB 30bit\n"); break;
		default: printf("-----\n"); break;
		}
		printf("buffer format : ");
		switch(m_outputBufferFormat)
		{
		case BUFFER_FORMAT_RGBA32:	printf("RGBA 8bit\n"); break;
		case BUFFER_FORMAT_RGBA64:	printf("RGBA 16bit\n"); break;
		case BUFFER_FORMAT_YUY2:	printf("YUY2\n"); break;
		case BUFFER_FORMAT_Y216:	printf("Y216\n"); break;
		case BUFFER_FORMAT_NV12:	printf("NV12\n"); break;
		case BUFFER_FORMAT_P016:	printf("P016\n"); break;
		default: printf("-----\n"); break;
		}
		if (m_bUseCuda)
		{
			if (m_bUseCudaHost)
				printf("pipeline: cuda (host to device)\n");
			else
				printf("pipeline: cuda (device to device)\n");
		}
#if defined(__WIN32__)
		if (m_pVideoDecD3D11)
		{
			printf("Cinecoder + DirectX11: yes\n");
		}
#endif
		printf("-------------------------------------\n");
	}

	return res;
}

int DecodeDaniel2::StartDecode()
{
	// start main thread of decode (call ThreadProc)

	if (!m_bInitDecoder)
		return -1;

	Create(); // creating thread <ThreadProc>

	return 0;
}

int DecodeDaniel2::StopDecode()
{
	// stop main thread of decode

	m_bProcess = false;

#ifdef USE_SIMPL_QUEUE
	m_queueFrames.Complete();
	m_queueFrames_free.Complete();
#endif

	Close(); // closing thread <ThreadProc>

	return 0;
}

C_Block* DecodeDaniel2::MapFrame()
{
	C_Block *pBlock = nullptr;
#ifdef USE_SIMPL_QUEUE
	m_queueFrames.Get(&pBlock); // receiving a block (C_Block) from a queue of finished decoded frames
#else
	m_queueFrames.Get(&pBlock, m_evExit); // receiving a block (C_Block) from a queue of finished decoded frames
#endif
	return pBlock;
}

void DecodeDaniel2::UnmapFrame(C_Block* pBlock)
{
	if (pBlock)
	{
		m_queueFrames_free.Queue(pBlock); // adding a block (C_Block) to the queue for processing (free frame queue)
	}
}

#if defined(__WIN32__)
BOOL GetFileVersion(LPCTSTR szPath, LARGE_INTEGER &lgVersion)
{
	if (szPath == NULL)
		return FALSE;

	DWORD dwHandle;
	UINT  cb;
	cb = GetFileVersionInfoSize(szPath, &dwHandle);
	if (cb > 0)
	{
		BYTE* pFileVersionBuffer = new BYTE[cb];
		if (pFileVersionBuffer == NULL)
			return FALSE;

		if (GetFileVersionInfo(szPath, 0, cb, pFileVersionBuffer))
		{
			VS_FIXEDFILEINFO* pVersion = NULL;
			if (VerQueryValue(pFileVersionBuffer, TEXT("\\"), (VOID**)&pVersion, &cb) &&
				pVersion != NULL)
			{
				lgVersion.HighPart = pVersion->dwFileVersionMS;
				lgVersion.LowPart = pVersion->dwFileVersionLS;
				delete[] pFileVersionBuffer;
				return TRUE;
			}
		}

		delete[] pFileVersionBuffer;
	}
	return FALSE;
}

HRESULT DecodeDaniel2::LoadPlugin(const char* pluginDLL)
{
	char strCinecoder[] = "Cinecoder.dll";
	char strPath[MAX_PATH] = { 0 };
	GetModuleFileNameA(GetModuleHandleA(strCinecoder), strPath, MAX_PATH);
	std::string strPluginDLL(strPath);
	std::size_t found = strPluginDLL.find(strCinecoder);
	if (found != std::string::npos)
	{
		strPluginDLL.erase(found);
		strPluginDLL += pluginDLL;
	}
	else return E_FAIL;

	if (std::ifstream(strPluginDLL.c_str()).good())
	{
		LARGE_INTEGER lgVersion = { 0 };
		std::wstring wstr(strPluginDLL.begin(), strPluginDLL.end());
		GetFileVersion(wstr.c_str(), lgVersion);
		printf("%s # File Version: %d.%d.%d.%d\n", pluginDLL, HIWORD(lgVersion.HighPart), LOWORD(lgVersion.HighPart), HIWORD(lgVersion.LowPart), LOWORD(lgVersion.LowPart));
	}
	else return E_FAIL;

	//std::wstring strWPluginDLL(strPluginDLL.begin(), strPluginDLL.end());
	//LPCTSTR szVersionFile = strWPluginDLL.c_str();

	//DWORD verHandle = 0;
	//DWORD verSize = GetFileVersionInfoSize(szVersionFile, &verHandle);

	//if (verSize != NULL)
	//{
	//	std::vector<char> verData(verSize);
	//	UINT size = 0;
	//	LPBYTE lpBuffer = NULL;

	//	if (GetFileVersionInfo(szVersionFile, verHandle, verSize, verData.data()))
	//	{
	//		if (VerQueryValue(verData.data(), TEXT("\\"), (VOID FAR* FAR*)&lpBuffer, &size))
	//		{
	//			if (size)
	//			{
	//				VS_FIXEDFILEINFO *verInfo = (VS_FIXEDFILEINFO *)lpBuffer;
	//				if (verInfo->dwSignature == 0xfeef04bd)
	//				{
	//					// Doesn't matter if you are on 32 bit or 64 bit,
	//					// DWORD is always 32 bits, so first two revision numbers
	//					// come from dwFileVersionMS, last two come from dwFileVersionLS

	//					//wchar_t szBuffer[2048];
	//					//swprintf_s(szBuffer, L"%d.%d.%d.%d", HIWORD(verInfo->dwProductVersionMS), LOWORD(verInfo->dwProductVersionMS),
	//					//	HIWORD(verInfo->dwProductVersionLS), LOWORD(verInfo->dwProductVersionLS));

	//					printf("%s # File Version: %d.%d.%d.%d\n",
	//					pluginDLL, HIWORD(verInfo->dwFileVersionMS), LOWORD(verInfo->dwFileVersionMS), HIWORD(verInfo->dwFileVersionLS), LOWORD(verInfo->dwFileVersionLS));
	//				}
	//			}
	//		}
	//	}
	//}

	CC_STRING plugin_filename_str = _com_util::ConvertStringToBSTR(strPluginDLL.c_str());
	return m_piFactory->LoadPlugin(plugin_filename_str); // no error here
}
#endif

int DecodeDaniel2::CreateDecoder()
{
	HRESULT hr = S_OK;

	m_piFactory = nullptr;

	size_t iMaxCountDecoders = std::max((size_t)1, std::min(m_dec_params.max_count_decoders, (size_t)4)); // 1..4

	bool useCuda = m_dec_params.type == VD_TYPE_CUDA ? true : false;
	bool useOpenCL = m_dec_params.type == VD_TYPE_OpenCL ? true : false;
	bool useMetal = m_dec_params.type == VD_TYPE_Metal ? true : false;
	bool useQuickSync = m_dec_params.type == VD_TYPE_QuickSync ? true : false;
	bool useIVPL = m_dec_params.type == VD_TYPE_IVPL ? true : false;
	bool useAMF = m_dec_params.type == VD_TYPE_AMF ? true : false;
	bool useNVDEC = m_dec_params.type == VD_TYPE_NVDEC ? true : false;

	Cinecoder_CreateClassFactory((ICC_ClassFactory**)&m_piFactory); // get Factory
	if (FAILED(hr)) 
		return printf("DecodeDaniel2: Cinecoder_CreateClassFactory failed!\n"), hr;

	hr = m_piFactory->AssignLicense(COMPANYNAME, LICENSEKEY); // set license
	if (FAILED(hr)) 
		return printf("DecodeDaniel2: AssignLicense failed!\n"), hr;

	CC_VERSION_INFO version = Cinecoder_GetVersion(); // get version of Cinecoder

	std::string strCinecoderVersion;

	strCinecoderVersion = "Cinecoder # ";
	strCinecoderVersion += std::to_string((long long)version.VersionHi);
	strCinecoderVersion += ".";
	strCinecoderVersion += std::to_string((long long)version.VersionLo);
	strCinecoderVersion += ".";
	strCinecoderVersion += std::to_string((long long)version.EditionNo);
	strCinecoderVersion += ".";
	strCinecoderVersion += std::to_string((long long)version.RevisionNo);

	printf("%s\n", strCinecoderVersion.c_str()); // print version of Cinecoder

#if defined(__WIN32__) && (CINECODER_VERSION < 40000)
	LoadPlugin("Cinecoder.Plugin.GpuCodecs.dll");
#endif

//#ifdef _DEBUG
	if (FAILED(hr = Cinecoder_SetErrorHandler(&g_ErrorHandler))) // set error handler
		printf("Error: call Cinecoder_SetErrorHandler() return 0x%x\n", hr);
//#endif

	CLSID clsidDecoder;
	switch(m_file.GetStreamType())
	{
		case CC_ES_TYPE_VIDEO_AVC_INTRA:
			clsidDecoder = CLSID_CC_AVCIntraDecoder2;
			useCuda = false;
			m_strStreamType = "AVC-Intra";
			bIntraFormat = true;
			break;

		case CC_ES_TYPE_VIDEO_J2K:
			clsidDecoder = CLSID_CC_J2K_VideoDecoder;
			useCuda = false;
			m_strStreamType = "JPEG-2000";
			bIntraFormat = true;
			break;

		case CC_ES_TYPE_VIDEO_MPEG2:
			clsidDecoder = CLSID_CC_MpegVideoDecoder;
			useCuda = false;
			m_strStreamType = "MPEG";
			bIntraFormat = false;
			break;

#if !defined(__WIN32__)
		case CC_ES_TYPE_VIDEO_H264:
		case CC_ES_TYPE_VIDEO_AVC1:
			clsidDecoder = CLSID_CC_H264VideoDecoder;
			useCuda = false;
			m_strStreamType = "H264";
			bIntraFormat = false;
			break;
#else
		case CC_ES_TYPE_VIDEO_H264:
		case CC_ES_TYPE_VIDEO_AVC1:
			//clsidDecoder = CLSID_CC_AVC1VideoDecoder_NV; // work without UnwrapFrame()
			//m_strStreamType = "AVC1";

#if 0		// For H264/AVC1/HEVC/HVC1 - support only CPU pipeline or GPU pipeline with D3DX11 (use: -cuda -d3d11 -cinecoderD3D11) / GetFrame failed for only GPU
			useCuda = m_pRender && useCuda ? true : false;
#endif
			clsidDecoder = useCuda ? CLSID_CC_H264VideoDecoder_NV : CLSID_CC_H264VideoDecoder;
			if (useQuickSync) clsidDecoder = CLSID_CC_H264VideoDecoder_IMDK;
#if	(CINECODER_VERSION >= 41201)
			if (useIVPL) clsidDecoder = CLSID_CC_H264VideoDecoder_IVPL;
#endif
			if (useAMF) clsidDecoder = CLSID_CC_H264VideoDecoder_AMF;
			if (useNVDEC) clsidDecoder = CLSID_CC_H264VideoDecoder_NV;
			m_strStreamType = "H264";
			bIntraFormat = false;
			break;

		case CC_ES_TYPE_VIDEO_HEVC:
		case CC_ES_TYPE_VIDEO_HVC1:
			//clsidDecoder = CLSID_CC_HVC1VideoDecoder_NV; // work without UnwrapFrame()
			//m_strStreamType = "HVC1";

#if 0		// For H264/AVC1/HEVC/HVC1 - support only CPU pipeline or GPU pipeline with D3DX11 (use: -cuda -d3d11 -cinecoderD3D11) / GetFrame failed for only GPU
			useCuda = m_pRender && useCuda ? true : false;
#endif
			//clsidDecoder = useCuda ? CLSID_CC_HEVCVideoDecoder_NV : CLSID_CC_HEVCVideoDecoder;
			clsidDecoder = CLSID_CC_HEVCVideoDecoder_NV; // as we do not have software HEVC try always NV
			if (useQuickSync) clsidDecoder = CLSID_CC_HEVCVideoDecoder_IMDK;
#if	(CINECODER_VERSION >= 41201)
			if (useIVPL) clsidDecoder = CLSID_CC_HEVCVideoDecoder_IVPL;
#endif
			if (useAMF) clsidDecoder = CLSID_CC_HEVCVideoDecoder_AMF;
			if (useNVDEC) clsidDecoder = CLSID_CC_HEVCVideoDecoder_NV;
			m_strStreamType = "HEVC";
			bIntraFormat = false;
			break;
#endif

#if	(CINECODER_VERSION >= 42403)
		case CC_ES_TYPE_VIDEO_AV1:
			//clsidDecoder = useCuda ? CLSID_CC_AV1_VideoDecoder_NV : CLSID_CC_AV1_VideoDecoder; // AV1 Decoder has no CPU implementation, use substitutions.
			clsidDecoder = CLSID_CC_AV1_VideoDecoder_NV;

			if (useIVPL) clsidDecoder = CLSID_CC_AV1_VideoDecoder_IVPL;
			if (useAMF) clsidDecoder = CLSID_CC_AV1_VideoDecoder_AMF;
			if (useNVDEC) clsidDecoder = CLSID_CC_AV1_VideoDecoder_NV;

			m_strStreamType = "AV1";
			bIntraFormat = false;
			break;
#endif

		case CC_ES_TYPE_VIDEO_DANIEL:
			clsidDecoder = useCuda ? CLSID_CC_DanielVideoDecoder_CUDA : CLSID_CC_DanielVideoDecoder;
#if (CINECODER_VERSION >= 42003)
			if (useOpenCL) clsidDecoder = CLSID_CC_DanielVideoDecoder_OCL;
#endif
#if (CINECODER_VERSION >= 42211)
			if (useMetal) clsidDecoder = CLSID_CC_DanielVideoDecoder_MTL;
#endif
			m_strStreamType = "Daniel";
			bIntraFormat = true;
			break;

		default:
			printf("DecodeDaniel2: Not support current format!\n");
			return -1;
			break;
	}

	if (m_bUseCuda && !useCuda)
	{
		//m_bUseCudaHost = true; // use CUDA-pipeline with host memory
		printf("Error: cannot support GPU-decoding for this format(%s) or pipe(use: -cuda -d3d11 -cinecoderD3D11)!\n", m_strStreamType);
		return -1; // if not GPU-decoder -> exit
	}

	if (FAILED(hr = m_piFactory->CreateInstance(clsidDecoder, IID_ICC_VideoDecoder, (IUnknown**)&m_pVideoDec)))
		return printf("DecodeDaniel2: CreateInstance failed!\n"), hr;

	if (useCuda && clsidDecoder == CLSID_CC_DanielVideoDecoder_CUDA && m_bPutColorFormat) //CUDA decoder needs a little extra help getting the color format correct
	{
		com_ptr<ICC_DanielVideoDecoder_CUDA> pCuda;

		if (FAILED(hr = m_pVideoDec->QueryInterface(IID_ICC_DanielVideoDecoder_CUDA, (void**)&pCuda)) || !pCuda)
			return printf("DecodeDaniel2: Failed to get ICC_DanielVideoDecoder_CUDA interface!\n"), hr;

		//if (FAILED(hr = pCuda->put_TargetColorFormat(static_cast<CC_COLOR_FMT>(CCF_B8G8R8A8)))) // need call put_TargetColorFormat for using GetFrame when using GPU-pipeline
		if (FAILED(hr = pCuda->put_TargetColorFormat(static_cast<CC_COLOR_FMT>(m_fmt)))) // need call put_TargetColorFormat for using GetFrame when using GPU-pipeline
			return printf("DecodeDaniel2: put_TargetColorFormat failed!\n"), hr;
	}

	com_ptr<ICC_ProcessDataPolicyProp> pPolicy = nullptr;
	if (SUCCEEDED(hr = m_pVideoDec->QueryInterface(IID_ICC_ProcessDataPolicyProp, (void**)&pPolicy)) && pPolicy)
	{
		if (FAILED(hr = pPolicy->put_ProcessDataPolicy(CC_PDP_PARSED_DATA)))
			return printf("DecodeDaniel2: put_ProcessDataPolicy failed!\n"), hr;
	}

	com_ptr<ICC_ConcurrencyLevelProp> pConcur = nullptr;
	if (SUCCEEDED(hr = m_pVideoDec->QueryInterface(IID_ICC_ConcurrencyLevelProp, (void**)&pConcur)) && pConcur)
	{
		// set count of decoders in carousel of decoders
		if (FAILED(hr = pConcur->put_ConcurrencyLevel(static_cast<CC_AMOUNT>(iMaxCountDecoders))))
			return printf("DecodeDaniel2: put_ConcurrencyLevel failed!\n"), hr;
	}

	// set output callback
	if (FAILED(hr = m_pVideoDec->put_OutputCallback((ICC_DataReadyCallback *)this)))
		return printf("DecodeDaniel2: put_OutputCallback failed!\n"), hr;

#if defined(__WIN32__)
	m_pVideoDec->QueryInterface(IID_ICC_D3D11VideoProducer, (void**)&m_pVideoDecD3D11);
	if (m_bUseCuda && m_pVideoDecD3D11 && m_pRender)
	{
		IDXGIAdapter1* pCapableAdapter = m_pCapableAdapter;

		if (FAILED(hr = m_pVideoDecD3D11->AssignAdapter(pCapableAdapter)))
			return printf("DecodeDaniel2: AssignAdapter failed!\n"), hr;
	}
	else m_pVideoDecD3D11 = nullptr;
#endif

	if (m_dec_params.scale_factor > 0)
	{
		com_ptr<ICC_VDecFixedScaleFactorProp> pVDecFixedScaleFactorProp = nullptr;
		hr = m_pVideoDec->QueryInterface(IID_ICC_VDecFixedScaleFactorProp, (void**)&pVDecFixedScaleFactorProp);
		if (SUCCEEDED(hr) && pVDecFixedScaleFactorProp)
		{
			CC_VDEC_SCALE_FACTOR v = (CC_VDEC_SCALE_FACTOR)(m_dec_params.scale_factor);
			hr = pVDecFixedScaleFactorProp->put_FixedScaleFactor(v);
		}
	}

#ifdef USE_CUDA_SDK
	if (m_bUseCuda)
	{
		cudaError_t err;

		err = cudaGetDevice(&m_iGpuDevice); __vrcu

		//CUresult cuRes = CUDA_SUCCESS;

		//cuRes = cuInit(0);

		//CUdevice cuDevice;
		//CUcontext cuContext;

		//cuRes = cuDeviceGet(&cuDevice, i_cuda_device);
		//cuRes = cuDevicePrimaryCtxRetain(&cuContext, cuDevice);

		//com_ptr<ICC_CudaContextProp> pCudaCtxProp = nullptr;
		//if (SUCCEEDED(hr = m_pVideoDec->QueryInterface(IID_ICC_CudaContextProp, (void**)&pCudaCtxProp)))
		//{
		//	hr = pCudaCtxProp->put_CudaContext(cuContext);
		//}
	}
#endif

	// set device id
	com_ptr<ICC_DeviceIDProp> pDeviceIDProp = nullptr;
	if (SUCCEEDED(hr = m_pVideoDec->QueryInterface(IID_ICC_CudaContextProp, (void**)&pDeviceIDProp)))
	{
		if (FAILED(hr = pDeviceIDProp->put_DeviceID(m_iGpuDevice)))
			return printf("DecodeDaniel2: put_DeviceID failed!\n"), hr;
	}

	// init decoder
	if (FAILED(hr = m_pVideoDec->Init()))
		return printf("DecodeDaniel2: Init failed!\n"), hr;

	com_ptr<ICC_FrameRateProp> pFrameRateProp = nullptr;
	hr = m_pVideoDec->QueryInterface(IID_ICC_FrameRateProp, (void**)&pFrameRateProp);
	if (SUCCEEDED(hr) && pFrameRateProp)
	{
		CC_FRAME_RATE frame_rate;
		if (SUCCEEDED(hr)) hr = m_file.GetFrameRate(frame_rate);
		if (frame_rate.num != 0)
		{
			if (SUCCEEDED(hr)) hr = pFrameRateProp->put_FrameRate(frame_rate);
			if (SUCCEEDED(hr)) m_FrameRate = frame_rate;
		}
	}

	return 0;
}

int DecodeDaniel2::DestroyDecoder()
{
	HRESULT hr = S_OK;

	if (m_pVideoDec != nullptr)
	{
		hr = m_pVideoDec->Done(CC_FALSE); // call done for decoder with param CC_FALSE (without flush data to DataReady)
	}

	m_pVideoDec = nullptr;

	//CC_BOOL bOpened; 
	//hr = m_pMediaReader->get_IsOpened(&bOpened);
	//if (bOpened)
	//{
	//	hr = m_pMediaReader->Close();
	//}
	m_pMediaReader = nullptr;

	m_piFactory = nullptr;

	return 0;
}

int DecodeDaniel2::InitValues()
{
	size_t iCountBlocks = 5; // set count of blocks in queue

	int res = 0;

	for (size_t i = 0; i < iCountBlocks; i++) // allocating memory of list of blocks (C_Block)
	{
		m_listBlocks.emplace_back(C_Block());

		size_t size = 0;

		//if (strcmp(m_strStreamType, "HEVC") == 0 ||
		//	strcmp(m_strStreamType, "H264") == 0 ||
		//	strcmp(m_strStreamType, "HVC1") == 0 ||
		//	strcmp(m_strStreamType, "AVC1") == 0)
		//	size = m_stride * m_height * 3 / 2;

		if (m_outputBufferFormat == BUFFER_FORMAT_NV12 || m_outputBufferFormat == BUFFER_FORMAT_P016)
		{
			size = (m_stride * m_height) + (m_stride * (m_height / 2));
		}
		else if (m_outputBufferFormat == BUFFER_FORMAT_P216)
		{
			size = (m_stride * m_height) * 2;
		}

		res = m_listBlocks.back().Init(m_width, m_height, m_stride, size, m_bUseCuda);

		if (res != 0)
		{
			printf("InitBlocks: Init() return error - %d\n", res);
			return res;
		}

		//m_queueFrames_free.Queue(&m_listBlocks.back()); // add free pointers to queue
	}

	return 0;
}

int DecodeDaniel2::DestroyValues()
{
	// clear all queues and lists

	m_queueFrames.Free();
	m_queueFrames_free.Free();

#if defined(__WIN32__)
	m_pVideoDecD3D11 = nullptr;
	m_pRender = nullptr;
	m_pCapableAdapter = nullptr;
#endif

	m_listBlocks.clear();

	return 0;
}

HRESULT STDMETHODCALLTYPE DecodeDaniel2::DataReady(IUnknown *pDataProducer)
{
	HRESULT hr = S_OK;

	com_ptr<ICC_VideoProducer> pVideoProducer;

	if (FAILED(hr = m_pVideoDec->QueryInterface(IID_ICC_VideoProducer, (void**)&pVideoProducer))) // get Producer
		return printf("DecodeDaniel2: DataReady get VideoProducer failed!\n"), hr;

	///////////////////////////////////////////////
	// getting information about the decoded frame
	///////////////////////////////////////////////

	DWORD						BitDepth = 0;
	CC_CHROMA_FORMAT			ChromaFormat = CC_CHROMA_FORMAT_UNKNOWN;
	CC_DANIEL2_CODING_METHOD	CodingMethod = CC_D2_METHOD_DEFAULT;
	CC_COLOUR_DESCRIPTION		ColorCoefs = { CC_CPRIMS_UNKNOWN, CC_TXCHRS_UNKNOWN, CC_MCOEFS_UNKNOWN };
	CC_FRAME_RATE				FrameRate = { 0 };
	CC_SIZE						FrameSize = { 0 };
	CC_RATIONAL					AspectRatio = { 0 };
	CC_PICTURE_ORIENTATION		PictureOrientation = CC_PO_DEFAULT;
	CC_FLOAT					QuantScale = 0;
	CC_UINT						CodingNumber = 0;
	CC_TIME						PTS = 0;
	CC_TIME						DTS = 0;

	com_ptr<ICC_VideoStreamInfo> pVideoStreamInfo = nullptr;
	com_ptr<ICC_VideoFrameInfo> pVideoFrameInfo = nullptr;

	if (FAILED(hr = pVideoProducer->GetVideoStreamInfo((ICC_VideoStreamInfo**)&pVideoStreamInfo)) || !pVideoStreamInfo)
		return printf("GetVideoStreamInfo() fails\n"), hr;

	if (FAILED(hr = pVideoProducer->GetVideoFrameInfo((ICC_VideoFrameInfo**)&pVideoFrameInfo)) || !pVideoFrameInfo)
		return printf("GetVideoFrameInfo() fails\n"), hr;

	hr = pVideoStreamInfo->get_FrameRate(&FrameRate); __check_hr
	hr = pVideoStreamInfo->get_FrameSize(&FrameSize); __check_hr
	hr = pVideoStreamInfo->get_AspectRatio(&AspectRatio); __check_hr
	
	com_ptr<ICC_VideoStreamInfoExt>	pVideoStreamInfoExt = nullptr;
	if(FAILED(hr = pVideoStreamInfo->QueryInterface(IID_ICC_VideoStreamInfoExt, (void**)&pVideoStreamInfoExt)) || !pVideoStreamInfoExt)
		return printf("Failed to get ICC_VideoStreamInfoExt interface\n"), hr;

	hr = pVideoStreamInfoExt->get_ChromaFormat(&ChromaFormat); __check_hr
	hr = pVideoStreamInfoExt->get_ColorCoefs(&ColorCoefs); __check_hr
	hr = pVideoStreamInfoExt->get_BitDepthLuma(&BitDepth); __check_hr

	com_ptr<ICC_DanielVideoStreamInfo>	pDanielVideoStreamInfo;
	if(SUCCEEDED(pVideoStreamInfo->QueryInterface(IID_ICC_DanielVideoStreamInfo, (void**)&pDanielVideoStreamInfo)) && pDanielVideoStreamInfo)
	{
		hr = pDanielVideoStreamInfo->get_PictureOrientation(&PictureOrientation); __check_hr
	}

	hr = pVideoFrameInfo->get_PTS(&PTS); __check_hr
	hr = pVideoFrameInfo->get_CodingNumber(&CodingNumber); __check_hr

	if (PTS < m_iNegativePTS)
	{
		m_iNegativePTS = PTS;
	}

	PTS -= m_iNegativePTS;

	//printf("DataReady: coding_num = %d PTS = %d\n", CodingNumber, PTS);

	///////////////////////////////////////////////

	if (m_bProcess)
	{
		C_Block *pBlock = nullptr;
	#ifdef USE_SIMPL_QUEUE
		m_queueFrames_free.Get(&pBlock); // get free pointer to object of C_Block form queue
	#else
		m_queueFrames_free.Get(&pBlock, m_evExit); // get free pointer to object of C_Block form queue
	#endif
		if (pBlock)
		{
			pBlock->SetRotate((PictureOrientation == CC_PO_FLIP_VERTICAL || PictureOrientation == CC_PO_ROTATED_180DEG) ? true : false);  // rotate frame

			DWORD cb = 0;

#ifdef USE_CUDA_SDK
			if (m_bUseCuda)
			{
				cudaSetDevice(m_iGpuDevice); // set context or need use cuCtxPushCurrent(cuContext)/cuCtxPopCurrent(nullptr)

				if (m_bUseCudaHost) // use CUDA-pipeline with host memory
				{
					hr = pVideoProducer->GetFrame(m_fmt, pBlock->DataPtr(), (DWORD)pBlock->Size(), (INT)pBlock->Pitch(), &cb); // get decoded frame from Cinecoder
					__check_hr
					pBlock->CopyToGPU(); // copy frame from host to device memory
				}
				else
				{
#if defined(__CUDAConvertLib__)
					pBlock->iMatrixCoeff_YUYtoRGBA = ConvertMatrixCoeff_Default;
#endif
					if (ChromaFormat == CC_CHROMA_422)
						pBlock->iMatrixCoeff_YUYtoRGBA = (size_t)(ColorCoefs.MC); // need for CC_CHROMA_422

#if defined(__WIN32__)
					if (m_pVideoDecD3D11)
					{
						CC_VA_STATUS vaStatus = CC_VA_STATUS_OFF;
						CC_VIDEO_FRAME_DESCR frameDesc = {};

						m_pVideoDecD3D11->get_VA_Status(&vaStatus);
						if (vaStatus == CC_VA_STATUS_ON) m_pVideoDecD3D11->GetVideoFrameDescr(&frameDesc);

						ID3D11Buffer* pResourceDXD11 = (ID3D11Buffer*)pBlock->GetD3DX11ResourcePtr();
						//ID3D11Texture2D* pResourceDXD11 = (ID3D11Texture2D*)pBlock->GetD3DX11ResourcePtr();
						if (pResourceDXD11)
						{
							m_pRender->MultithreadSyncBegin();
							RegisterResourceD3DX11(pResourceDXD11); // Register the resources of D3DX11 in Cinecoder
							hr = m_pVideoDecD3D11->GetFrame(pResourceDXD11, &frameDesc); // Copy frame to resources of D3DX11
							UnregisterResourceD3DX11(pResourceDXD11); // Unregister the resources of D3DX11 in Cinecoder
							m_pRender->MultithreadSyncEnd();
							__check_hr
						}
					} // if (m_pVideoDecD3D11)
					else
#endif
					{
						hr = pVideoProducer->GetFrame(m_fmt, pBlock->DataGPUPtr(), (DWORD)pBlock->Size(), (INT)pBlock->Pitch(), &cb); // get decoded frame from Cinecoder
						__check_hr
					}
				}
			}
			else // use CPU-pipeline
			{
				hr = pVideoProducer->GetFrame(m_fmt, pBlock->DataPtr(), (DWORD)pBlock->Size(), (INT)pBlock->Pitch(), &cb); // get decoded frame from Cinecoder
				__check_hr
			}
#else // #ifdef USE_CUDA_SDK
			hr = pVideoProducer->GetFrame(m_fmt, pBlock->DataPtr(), (DWORD)pBlock->Size(), (INT)pBlock->Pitch(), &cb); // get decoded frame from Cinecoder
			__check_hr
#endif // #ifdef USE_CUDA_SDK

			if (m_llDuration > 0)
				pBlock->iFrameNumber = static_cast<size_t>(PTS) / static_cast<size_t>(m_llDuration); // save PTS (in our case this is the frame number)
			else
				pBlock->iFrameNumber = static_cast<size_t>(PTS * m_FrameRate.num) / static_cast<size_t>(m_llTimeBase * m_FrameRate.denom);

			m_queueFrames.Queue(pBlock); // add pointer to object of C_Block with final picture to queue
		} // if (pBlock)
	} // if (m_bProcess)
	else if (!m_bInitDecoder) // init values after first decoding frame
	{
		if (FrameRate.num == 0)
		{
			if (m_FrameRate.num == 0)
			{
				//printf("Error: video frame rate == 0\n");

				hr = m_piFactory->CreateInstance(CLSID_CC_MediaReader, IID_ICC_MediaReader, (IUnknown**)&m_pMediaReader);
				if (FAILED(hr)) return hr;

				const char* filename = m_filename.c_str();

#if defined(__WIN32__)
				CC_STRING file_name_str = _com_util::ConvertStringToBSTR(filename);
#elif defined(__APPLE__) || defined(__LINUX__)
				CC_STRING file_name_str = const_cast<CC_STRING>(filename);
#endif
				hr = m_pMediaReader->Open(file_name_str, CC_ROF_DISABLE_VIDEO_DECODER);
				if (FAILED(hr)) return hr;

				CC_FRAME_RATE FrameRateMR;
				hr = m_pMediaReader->get_FrameRate(&FrameRateMR);
				if (FAILED(hr)) return hr;

				//CC_BOOL bOpened = FALSE;
				//hr = m_pMediaReader->get_IsOpened(&bOpened);
				//if (bOpened)
				//{
				//	hr = m_pMediaReader->Close();
				//	if (FAILED(hr)) return hr;
				//}
				m_pMediaReader = nullptr;

				if (FrameRateMR.num == 0)
					return E_FAIL;

				printf("Set frame rate (MediaReader): %.2f\n", ((float)FrameRateMR.num / (float)FrameRateMR.denom));

				bCalculatePTS = true;
				FrameRate = FrameRateMR;
			}
			else
			{
				FrameRate = m_FrameRate;
			}
		}

		m_FrameRate = FrameRate;
		
		m_ChromaFormat = ChromaFormat;
		m_BitDepth = BitDepth;

		m_width = FrameSize.cx; // get width
		m_height = FrameSize.cy; // get height
		m_AspectRatio = AspectRatio;

		m_llTimeBase = 1; m_llDuration = 0;

		CC_TIME duration = 0;
		CC_TIMEBASE timeBase = 0;

		if (SUCCEEDED(pVideoFrameInfo->get_Duration(&duration)) && duration != 0)
			m_llDuration = duration;

		if (SUCCEEDED(m_pVideoDec->get_TimeBase(&timeBase)) && timeBase != 0)
			m_llTimeBase = timeBase;

		//com_ptr<ICC_VDecFixedScaleFactorProp> pVDecFixedScaleFactorProp = nullptr;
		//hr = m_pVideoDec->QueryInterface((ICC_VDecFixedScaleFactorProp**)&pVDecFixedScaleFactorProp);
		//if (SUCCEEDED(hr) && pVDecFixedScaleFactorProp)
		//{
		//	hr = pVDecFixedScaleFactorProp->get_FixedScaleFactor(&m_dec_params.scale_factor);
		//}

		com_ptr<ICC_VideoAccelerationInfo> pVideoAccelerationInfo;
		if (SUCCEEDED(m_pVideoDec->QueryInterface(IID_ICC_VideoAccelerationInfo, (void**)&pVideoAccelerationInfo)))
		{
			CC_VA_STATUS vaStatus;
			if (pVideoAccelerationInfo && SUCCEEDED(pVideoAccelerationInfo->get_VA_Status(&vaStatus)))
			{
				if (!(vaStatus == CC_VA_STATUS_ON || vaStatus == CC_VA_STATUS_PARTIAL))
				{
					// check for GPU pipeline initialization, if we expected GPU and switched to the CPU -> exit
					if (m_dec_params.type == VD_TYPE_QuickSync ||
						m_dec_params.type == VD_TYPE_AMF ||
						m_dec_params.type == VD_TYPE_NVDEC ||
						m_dec_params.type == VD_TYPE_IVPL)
						return 0;

					if (m_bUseCuda)
					{
#if defined(__WIN32__)
						if (m_pRender || m_pCapableAdapter)
						{
							printf("Error: cannot switch to CPU, please disable option <cinecoderD3D11>!\n");
							return 0;
						}
#endif
						m_bUseCudaHost = true; // use CUDA-pipeline with host memory
					}
					if (m_bUseCuda && !m_bUseCudaHost) return 0; // Init GPU-decoder failed -> exit
				}
			}
		}

#if defined(__WIN32__)
		if (strcmp(m_strStreamType, "HEVC") == 0 ||
			strcmp(m_strStreamType, "H264") == 0 ||
			strcmp(m_strStreamType, "HVC1") == 0 ||
			strcmp(m_strStreamType, "AVC1") == 0 ||
			strcmp(m_strStreamType, "AV1") == 0)
		{
			com_ptr<ICC_D3D11VideoObject> d3d11VideoObject;
			if (SUCCEEDED(m_pVideoDec->QueryInterface((ICC_D3D11VideoObject**)&d3d11VideoObject)))
			{
				CC_VA_STATUS vaStatus;
				if (d3d11VideoObject && SUCCEEDED(d3d11VideoObject->get_VA_Status(&vaStatus)))
				{
					if (!(vaStatus == CC_VA_STATUS_ON || vaStatus == CC_VA_STATUS_PARTIAL))
					{
						//if (m_bUseCuda) m_bUseCudaHost = true; // use CUDA-pipeline with host memory
						if (m_bUseCuda && !m_bUseCudaHost) return 0; // Init GPU-decoder failed -> exit
					}
				}

				if (m_bUseCuda)
				{
					CC_VIDEO_FRAME_DESCR vid_frame_desc;
					if (SUCCEEDED(d3d11VideoObject->GetVideoFrameDescr(&vid_frame_desc)))
					{
						m_fmt = vid_frame_desc.cFormat; // get native format
						m_stride = vid_frame_desc.iStride;
						m_outputImageFormat = IMAGE_FORMAT_RGBA8BIT;
						m_outputBufferFormat = BUFFER_FORMAT_NV12;

						if (m_fmt == CCF_NV12)
						{
							m_outputImageFormat = IMAGE_FORMAT_RGBA8BIT;
							m_outputBufferFormat = BUFFER_FORMAT_NV12;
						}
						else if (m_fmt == CCF_P016) // CCF_NV12_16BIT
						{
							m_outputImageFormat = IMAGE_FORMAT_RGBA16BIT;
							m_outputBufferFormat = BUFFER_FORMAT_P016;
						}
						else if (m_fmt == CCF_P216) // CCF_NV16_16BIT
						{
							m_outputImageFormat = IMAGE_FORMAT_RGBA16BIT;
							m_outputBufferFormat = BUFFER_FORMAT_P216;
						}
						else
							return E_FAIL;

						m_bInitDecoder = true; // set init decoder value
						m_eventInitDecoder.Set(); // set event about decoder was initialized

						return S_OK;
					}
				}
			}
		}
#endif

		CC_COLOR_FMT fmt = CCF_B8G8R8A8; // set output format

		// set user settings for output format
		if (m_dec_params.outputFormat == IMAGE_FORMAT_RGBA8BIT)
			BitDepth = 8;
		else if (m_dec_params.outputFormat == IMAGE_FORMAT_RGBA16BIT)
			BitDepth = 16;

		if (BitDepth > 8) fmt = fmt == CCF_B8G8R8A8 ? CCF_B16G16R16A16 : CCF_R16G16B16A16;

//#if defined(__WIN32__)
		if (m_bUseCuda && ChromaFormat == CC_CHROMA_422) fmt = BitDepth == 8 ? CCF_YUY2 : CCF_Y216;
//#endif

		CC_BOOL bRes = CC_FALSE;
		hr = pVideoProducer->IsFormatSupported(fmt, &bRes);
		if (!bRes || hr != S_OK)
		{
			fmt = CCF_B8G8R8A8; // last chance - try RGBA format
			hr = pVideoProducer->IsFormatSupported(fmt, &bRes);
			if (!bRes || hr != S_OK)
			{
				static bool err = true;
				if (err)
				{
					err = false;
					return printf("IsFormatSupported failed! (error = 0x%x)\n", hr), hr;
				}
			}
		}

		if (bRes) // fix problem when IsFormatSupported return OK, but GetFrame return error MPG_E_FORMAT_NOT_SUPPORTED
		{
			std::vector<CC_COLOR_FMT> list_fmt;
			list_fmt.push_back(fmt);
			list_fmt.push_back(CCF_B8G8R8A8); // default format for copy to texture - RGBA, for YUY need add convertor

			for (size_t i = 0; i < list_fmt.size(); i++)
			{
				DWORD iStride = 0;
				pVideoProducer->GetStride(list_fmt[i], &iStride); // get stride

				bool bUseCuda = m_bUseCuda && !m_bUseCudaHost;

				C_Block block;
				long lres = block.Init(m_width, m_height, iStride, 0, bUseCuda);
				if (lres == 0)
				{
					DWORD cb = 0;
					hr = pVideoProducer->GetFrame(list_fmt[i], bUseCuda ? block.DataGPUPtr() : block.DataPtr(), (DWORD)block.Size(), (INT)block.Pitch(), &cb);
					if (SUCCEEDED(hr))
					{
						fmt = list_fmt[i]; break;
					}
					else printf("InitDecoder: GetFrame(fmt = %d) failed! (error = 0x%x)\n", list_fmt[i], hr);
				}
			}

			if (FAILED(hr))
				return hr;
		}

		if (bRes)
		{
			DWORD iStride = 0;
			pVideoProducer->GetStride(fmt, &iStride); // get stride
			m_stride = (size_t)iStride;

			m_fmt = fmt;

			if (m_fmt == CCF_R8G8B8A8)
				m_outputImageFormat = IMAGE_FORMAT_RGBA8BIT;
			else if (m_fmt == CCF_B8G8R8A8)
				m_outputImageFormat = IMAGE_FORMAT_BGRA8BIT;
			else if (m_fmt == CCF_R16G16B16A16)
				m_outputImageFormat = IMAGE_FORMAT_RGBA16BIT;
			else if (m_fmt == CCF_B16G16R16A16)
				m_outputImageFormat = IMAGE_FORMAT_BGRA16BIT;
			else if (m_fmt == CCF_RGB30)
				m_outputImageFormat = IMAGE_FORMAT_RGB30;
			else if (m_fmt == CCF_YUY2)
				m_outputImageFormat = IMAGE_FORMAT_RGBA8BIT;
			else if (m_fmt == CCF_Y216)
				m_outputImageFormat = IMAGE_FORMAT_RGBA16BIT;
			else if (m_fmt == CCF_NV12)
			{
				m_outputImageFormat = IMAGE_FORMAT_RGBA8BIT;
				m_outputBufferFormat = BUFFER_FORMAT_NV12;
			}
			else if (m_fmt == CCF_P016) // CCF_NV12_16BIT
			{
				m_outputImageFormat = IMAGE_FORMAT_RGBA16BIT;
				m_outputBufferFormat = BUFFER_FORMAT_P016;
			}
			else if (m_fmt == CCF_P216) // CCF_NV16_16BIT
			{
				m_outputImageFormat = IMAGE_FORMAT_RGBA16BIT;
				m_outputBufferFormat = BUFFER_FORMAT_P216;
			}

			m_outputBufferFormat = BitDepth == 8 ? BUFFER_FORMAT_RGBA32 : BUFFER_FORMAT_RGBA64;

			if (m_bUseCuda && ChromaFormat == CC_CHROMA_422)
			{
				m_outputBufferFormat = BitDepth == 8 ? BUFFER_FORMAT_YUY2 : BUFFER_FORMAT_Y216; // need for convert to D3DX11/OpenGL texture
			}

#if defined(__WIN32__)
			if (m_pVideoDecD3D11)
			{
				// check actual decoder HW state
				CC_VA_STATUS vaStatus = CC_VA_STATUS_OFF;
				CC_VIDEO_FRAME_DESCR frameDesc = {};

				if (m_pVideoDecD3D11)
				{
					m_pVideoDecD3D11->get_VA_Status(&vaStatus);
					if (vaStatus == CC_VA_STATUS_ON) 
						hr = m_pVideoDecD3D11->GetVideoFrameDescr(&frameDesc);
					else m_pVideoDecD3D11 = nullptr;
				}
			}
#endif
			m_bInitDecoder = true; // set init decoder value
			m_eventInitDecoder.Set(); // set event about decoder was initialized
		}
	}

	__check_hr

	return hr;
}

long DecodeDaniel2::ThreadProc()
{
	m_bProcess = true;

	unsigned char* coded_frame = nullptr;
	size_t coded_frame_size = 0;
	size_t frame_number = 0;
	size_t coding_number = 0;

	HRESULT hr = S_OK;

	for (auto it = m_listBlocks.begin(); it != m_listBlocks.end(); ++it)
	{
#if defined(__WIN32__)
		if (m_pVideoDecD3D11)
		{
			ID3D11Buffer* pResourceDXD11 = nullptr;
			//ID3D11Texture2D* pResourceDXD11 = nullptr;

			size_t size = m_stride * m_height;

			if (m_outputBufferFormat == BUFFER_FORMAT_NV12 || m_outputBufferFormat == BUFFER_FORMAT_P016)
			{
				size = (m_stride * m_height) + (m_stride * (m_height / 2));
			}
			else if (m_outputBufferFormat == BUFFER_FORMAT_P216)
			{
				size = (m_stride * m_height);
			}

			if (true)
			{
				hr = m_pRender->CreateD3DXBuffer(&pResourceDXD11, size); __check_hr
			}
			//else
			//{
			//	D3D11_USAGE Usage = m_bUseCuda ? D3D11_USAGE_DEFAULT : D3D11_USAGE_DYNAMIC;
			//	DXGI_FORMAT format = DXGI_FORMAT_R16G16B16A16_UNORM;
			//	switch (m_outputImageFormat) {
			//	case IMAGE_FORMAT_RGBA8BIT: { format = DXGI_FORMAT_R8G8B8A8_UNORM; break; }
			//	case IMAGE_FORMAT_BGRA8BIT: { format = DXGI_FORMAT_B8G8R8A8_UNORM; break; }
			//	case IMAGE_FORMAT_RGBA16BIT:
			//	case IMAGE_FORMAT_BGRA16BIT: { format = DXGI_FORMAT_R16G16B16A16_UNORM; break; }
			//	default: { format = DXGI_FORMAT_R8G8B8A8_UNORM; }
			//	}

			//	ID3D11ShaderResourceView* pTexture_Srv = nullptr;
			//	hr = m_pRender->CreateD3DXTexture(format, Usage, m_width, m_height, &pResourceDXD11, &pTexture_Srv); __check_hr
			//}

			size_t buffer_size = size;
			it->InitD3DResource(pResourceDXD11, m_width, m_height, m_stride, buffer_size);
		}
#endif
		m_queueFrames_free.Queue(&(*it)); // add free pointers to queue
	}

	while (m_bProcess)
	{
		if (m_bPause)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			continue;
		}

		CodedFrame* frame = nullptr;
		frame = m_file.MapFrame(); // get pointer to next coded frame form DN2 file

		if (frame)
		{
			coded_frame = frame->coded_frame.GetPtr(); // poiter to coded frame
			coded_frame_size = frame->coded_frame_size; // size of coded frame
			frame_number = frame->frame_number; // number of display frame
			coding_number = frame->coding_number; // number of coding frame

			if (m_bDecode)
			{
				CC_TIME pts = (frame_number * m_llTimeBase * m_FrameRate.denom) / m_FrameRate.num;
				//printf("ProcessData: coding_num = %d frame_num = %d pts = %d\n", coding_number, frame_number, pts);
				if (coding_number == 0 || frame->flags == 1) // seek
				{
					hr = m_pVideoDec->Break(CC_TRUE); __check_hr
					if (SUCCEEDED(hr)) hr = m_pVideoDec->ProcessData(coded_frame, static_cast<CC_UINT>(coded_frame_size), 0, pts); __check_hr
				}
				else
				{
					if (bIntraFormat || bCalculatePTS)
						hr = m_pVideoDec->ProcessData(coded_frame, static_cast<CC_UINT>(coded_frame_size), 0, pts);
					else
						hr = m_pVideoDec->ProcessData(coded_frame, static_cast<CC_UINT>(coded_frame_size)); __check_hr
				}

				if (FAILED(hr)) // add coded frame to decoder
				{
					_assert(0);

					printf("ProcessData failed hr=%d coded_frame_size=%zu coded_frame=%p frame_number = %zd\n", hr, coded_frame_size, coded_frame, frame_number);

					hr = m_pVideoDec->Break(CC_FALSE); // break decoder with param CC_FALSE (without flush data to DataReady)

					__check_hr
				}
			}
			else
			{
				C_Block *pBlock = nullptr;
			#ifdef USE_SIMPL_QUEUE
				m_queueFrames_free.Get(&pBlock); // get free pointer to object of C_Block form queue
			#else
				m_queueFrames_free.Get(&pBlock, m_evExit); // get free pointer to object of C_Block form queue
			#endif

				if (pBlock)
				{
					pBlock->iFrameNumber = frame_number; // save frame number
					m_queueFrames.Queue(pBlock); // add pointer to object of C_Block with final picture to queue
				}
			}

			m_file.UnmapFrame(frame); // add to queue free pointer for reading coded frame
		}
	}

	m_pVideoDec->Break(CC_FALSE); // break decoder with param CC_FALSE (without flush data to DataReady)

	m_file.StopPipe(); // stop pipeline for reading DN2 file

	return 0;
}
