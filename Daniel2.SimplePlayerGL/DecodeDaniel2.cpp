#include "stdafx.h"
#include "DecodeDaniel2.h"

// Cinecoder
#include <Cinecoder_i.c>
#if defined(__WIN32__)
#include <Cinecoder.Plugin.GpuCodecs_i.c>
#pragma comment(lib, "windowscodecs.lib") // for IID_IDXGIFactory
#endif
#include "CinecoderErrorHandler.h"

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
	m_bUseCudaHost(false),
	m_pVideoDec(nullptr),
	m_pMediaReader(nullptr),
	m_strStreamType("Unknown"),
	bIntraFormat(false),
	m_llDuration(1),
	m_llTimeBase(1)
{
	m_FrameRate.num = 60;
	m_FrameRate.denom = 1;

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

int DecodeDaniel2::OpenFile(const char* const filename, size_t iMaxCountDecoders, bool useCuda, IMAGE_FORMAT outputFormat)
{
	m_bInitDecoder = false;

	m_bUseCuda = useCuda;

	m_setOutputFormat = outputFormat;

	int res = m_file.OpenFile(filename); // open input DN2 file

	iMaxCountDecoders = std::max((size_t)1, std::min(iMaxCountDecoders, (size_t)4)); // 1..4

	if (res == 0)
		res = CreateDecoder(iMaxCountDecoders, useCuda); // create decoders

	unsigned char* coded_frame = nullptr;
	size_t coded_frame_size = 0;

	CodedFrame buffer;

	if (res == 0)
	{
		coded_frame_size = 0;
		res = m_file.ReadFrame(0, buffer.coded_frame, coded_frame_size); // get 0-coded frame for add decoder and init values after decode first frame
	}

	if (res == 0)
	{
		coded_frame = buffer.coded_frame.GetPtr(); // poiter to 0-coded frame

		HRESULT hr = S_OK;

		m_eventInitDecoder.Reset();

		hr = m_pVideoDec->ProcessData(coded_frame, static_cast<CC_UINT>(coded_frame_size), 0, 0); __check_hr

		for (size_t i = 0; i < 2; i++)
		{
			if (SUCCEEDED(hr)) hr = m_pVideoDec->ProcessData(coded_frame, static_cast<CC_UINT>(coded_frame_size)); __check_hr

			if (FAILED(hr)) // add coded frame to decoder
			{
				assert(0);

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

	Close(); // closing thread <ThreadProc>

	return 0;
}

C_Block* DecodeDaniel2::MapFrame()
{
	C_Block *pBlock = nullptr;

	m_queueFrames.Get(&pBlock, m_evExit); // receiving a block (C_Block) from a queue of finished decoded frames

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

	CC_STRING plugin_filename_str = _com_util::ConvertStringToBSTR(strPluginDLL.c_str());
	return m_piFactory->LoadPlugin(plugin_filename_str); // no error here
}
#endif

int DecodeDaniel2::CreateDecoder(size_t iMaxCountDecoders, bool useCuda)
{
	HRESULT hr = S_OK;

	m_piFactory = nullptr;

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

#if defined(__WIN32__)
	LoadPlugin("Cinecoder.Plugin.GpuCodecs.dll");
#endif

//#ifdef _DEBUG
	Cinecoder_SetErrorHandler(&g_ErrorHandler); // set error handler
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

		case CC_ES_TYPE_VIDEO_H264:
			clsidDecoder = CLSID_CC_H264VideoDecoder;
			useCuda = false;
			m_strStreamType = "H.264";
			bIntraFormat = false;
			break;
			
#if defined(__WIN32__)
		case CC_ES_TYPE_VIDEO_HEVC:
			//clsidDecoder = useCuda ? CLSID_CC_HEVCVideoDecoder_NV : CLSID_CC_HEVCVideoDecoder;
			clsidDecoder = CLSID_CC_HEVCVideoDecoder_NV;
			m_strStreamType = "HEVC";
			bIntraFormat = false;
			break;
#endif
		case CC_ES_TYPE_VIDEO_DANIEL:
			clsidDecoder = useCuda ? CLSID_CC_DanielVideoDecoder_CUDA : CLSID_CC_DanielVideoDecoder;
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
		printf("Error: cannot support GPU-decoding for this format!\n");
		return -1; // if not GPU-decoder -> exit
	}

	if (FAILED(hr = m_piFactory->CreateInstance(clsidDecoder, IID_ICC_VideoDecoder, (IUnknown**)&m_pVideoDec)))
		return printf("DecodeDaniel2: CreateInstance failed!\n"), hr;

#if !defined(__WIN32__)
	if (useCuda && clsidDecoder == CLSID_CC_DanielVideoDecoder_CUDA) //CUDA decoder needs a little extra help getting the color format correct
	{
		com_ptr<ICC_DanielVideoDecoder_CUDA> pCuda;

		if(FAILED(hr = m_pVideoDec->QueryInterface(IID_ICC_DanielVideoDecoder_CUDA, (void**)&pCuda)) || !pCuda)
			return printf("DecodeDaniel2: Failed to get ICC_DanielVideoDecoder_CUDA interface!\n"), hr;

		if (FAILED(hr = pCuda->put_TargetColorFormat(static_cast<CC_COLOR_FMT>(CCF_BGRA)))) // need call put_TargetColorFormat for using GetFrame when using GPU-pipeline
			return printf("DecodeDaniel2: put_TargetColorFormat failed!\n"), hr;
	}
#endif

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

	// init decoder
	if (FAILED(hr = m_pVideoDec->Init()))
		return printf("DecodeDaniel2: Init failed!\n"), hr;

	return 0;
}

int DecodeDaniel2::DestroyDecoder()
{
	if (m_pVideoDec != nullptr)
		m_pVideoDec->Done(CC_FALSE); // call done for decoder with param CC_FALSE (without flush data to DataReady)

	m_pVideoDec = nullptr;

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

		if (strcmp(m_strStreamType, "HEVC") == 0)
			size = m_stride * m_height * 3 / 2;

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

	///////////////////////////////////////////////

	if (m_bProcess)
	{
		C_Block *pBlock = nullptr;

		m_queueFrames_free.Get(&pBlock, m_evExit); // get free pointer to object of C_Block form queue

		if (pBlock)
		{
			pBlock->SetRotate((PictureOrientation == CC_PO_FLIP_VERTICAL || PictureOrientation == CC_PO_ROTATED_180DEG) ? true : false);  // rotate frame

			DWORD cb = 0;

#ifdef USE_CUDA_SDK
#if defined(__WIN32__)// || defined(__LINUX__)
			if (m_bUseCuda)
			{
				if (m_bUseCudaHost) // use CUDA-pipeline with host memory
				{
					hr = pVideoProducer->GetFrame(m_fmt, pBlock->DataPtr(), (DWORD)pBlock->Size(), (INT)pBlock->Pitch(), &cb); // get decoded frame from Cinecoder
					__check_hr
					pBlock->CopyToGPU(); // copy frame from host to device memory
				}
				else
				{
					pBlock->iMatrixCoeff_YUYtoRGBA = ConvertMatrixCoeff_Default;

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
					}
					else
#endif
					{
						hr = pVideoProducer->GetFrame(m_fmt, pBlock->DataGPUPtr(), (DWORD)pBlock->Size(), (INT)pBlock->Pitch(), &cb); // get decoded frame from Cinecoder
						__check_hr
					}
				}
			}
			else
#endif
			{
				hr = pVideoProducer->GetFrame(m_fmt, pBlock->DataPtr(), (DWORD)pBlock->Size(), (INT)pBlock->Pitch(), &cb); // get decoded frame from Cinecoder
				__check_hr
			}
#else
			hr = pVideoProducer->GetFrame(m_fmt, pBlock->DataPtr(), (DWORD)pBlock->Size(), (INT)pBlock->Pitch(), &cb); // get decoded frame from Cinecoder
			__check_hr
#endif
			if (m_llDuration > 0)
				pBlock->iFrameNumber = static_cast<size_t>(PTS) / m_llDuration; // save PTS (in our case this is the frame number)
			else
				pBlock->iFrameNumber = (PTS * m_FrameRate.num) / (m_llTimeBase * m_FrameRate.denom);

			m_queueFrames.Queue(pBlock); // add pointer to object of C_Block with final picture to queue
		}
	}
	else if (!m_bInitDecoder) // init values after first decoding frame
	{
		m_FrameRate = FrameRate;

		m_width = FrameSize.cx; // get width
		m_height = FrameSize.cy; // get height

		m_llTimeBase = 1; m_llDuration = 0;

		CC_TIME duration = 0;
		CC_TIMEBASE timeBase = 0;

		if (SUCCEEDED(pVideoFrameInfo->get_Duration(&duration)) && duration != 0)
			m_llDuration = duration;

		if (SUCCEEDED(m_pVideoDec->get_TimeBase(&timeBase)) && timeBase != 0)
			m_llTimeBase = timeBase;

		com_ptr<ICC_VideoAccelerationInfo> pVideoAccelerationInfo;
		if (SUCCEEDED(m_pVideoDec->QueryInterface(IID_ICC_VideoAccelerationInfo, (void**)&pVideoAccelerationInfo)))
		{
			CC_VA_STATUS vaStatus;
			if (pVideoAccelerationInfo && SUCCEEDED(pVideoAccelerationInfo->get_VA_Status(&vaStatus)))
			{
				if (!(vaStatus == CC_VA_STATUS_ON || vaStatus == CC_VA_STATUS_PARTIAL))
				{
					//if (m_bUseCuda) m_bUseCudaHost = true; // use CUDA-pipeline with host memory
					if (m_bUseCuda && !m_bUseCudaHost) return 0; // Init GPU-decoder failed -> exit
				}
			}
		}

#if defined(__WIN32__)
		if (strcmp(m_strStreamType, "HEVC") == 0)
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
						m_fmt = vid_frame_desc.cFormat;
						m_stride = vid_frame_desc.iStride;
						m_outputImageFormat = IMAGE_FORMAT_RGBA8BIT;
						m_outputBufferFormat = BUFFER_FORMAT_NV12;

						if (m_fmt != CCF_NV12)
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
		if (m_setOutputFormat == IMAGE_FORMAT_RGBA8BIT)
			BitDepth = 8;
		else if (m_setOutputFormat == IMAGE_FORMAT_RGBA16BIT)
			BitDepth = 16;

		if (BitDepth > 8) fmt = fmt == CCF_B8G8R8A8 ? CCF_B16G16R16A16 : CCF_R16G16B16A16;

#if defined(__WIN32__)
		if (m_bUseCuda && ChromaFormat == CC_CHROMA_422) fmt = BitDepth == 8 ? CCF_YUY2 : CCF_Y216;
#endif

		CC_BOOL bRes = CC_FALSE;
		hr = pVideoProducer->IsFormatSupported(fmt, &bRes);
		if (!bRes || hr != S_OK)
		{
			static bool err = true;
			if (err)
			{
				err = false;
				return printf("IsFormatSupported failed!\n"), hr;
			}
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

			m_ChromaFormat = ChromaFormat;
			m_BitDepth = BitDepth;
			
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

	HRESULT hr = S_OK;

	for (auto it = m_listBlocks.begin(); it != m_listBlocks.end(); ++it)
	{
#if defined(__WIN32__)
		if (m_pVideoDecD3D11)
		{
			ID3D11Buffer* pResourceDXD11 = nullptr;
			//ID3D11Texture2D* pResourceDXD11 = nullptr;

			if (true)
			{
				hr = m_pRender->CreateD3DXBuffer(&pResourceDXD11, m_stride * m_width); __check_hr
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

			size_t buffer_size = m_stride * m_width;
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
			frame_number = frame->frame_number; // number of coded frame

			if (m_bDecode)
			{
				CC_TIME pts = (frame_number * m_llTimeBase * m_FrameRate.denom) / m_FrameRate.num;

				if (frame_number == 0 || frame->flags == 1) // seek
				{
					hr = m_pVideoDec->Break(CC_TRUE); __check_hr
					if (SUCCEEDED(hr)) hr = m_pVideoDec->ProcessData(coded_frame, static_cast<CC_UINT>(coded_frame_size), 0, pts); __check_hr
				}
				else
				{
					if (bIntraFormat)
						hr = m_pVideoDec->ProcessData(coded_frame, static_cast<CC_UINT>(coded_frame_size), 0, pts);
					else
						hr = m_pVideoDec->ProcessData(coded_frame, static_cast<CC_UINT>(coded_frame_size)); __check_hr
				}

				if (FAILED(hr)) // add coded frame to decoder
				{
					assert(0);

					printf("ProcessData failed hr=%d coded_frame_size=%zu coded_frame=%p frame_number = %zd\n", hr, coded_frame_size, coded_frame, frame_number);

					hr = m_pVideoDec->Break(CC_FALSE); // break decoder with param CC_FALSE (without flush data to DataReady)

					__check_hr
				}
			}
			else
			{
				C_Block *pBlock = nullptr;

				m_queueFrames_free.Get(&pBlock, m_evExit); // get free pointer to object of C_Block form queue

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
