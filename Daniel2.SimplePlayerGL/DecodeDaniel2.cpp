#include "stdafx.h"
#include "DecodeDaniel2.h"

#include <algorithm>    // std::max / std::min
#include <string>
#include <cassert>

// Cinecoder
#include <Cinecoder_i.c>

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
	m_bProcess(false),
	m_bPause(false),
	m_bInitDecoder(false),
	m_pVideoDec(nullptr)
{
}

DecodeDaniel2::~DecodeDaniel2()
{
	StopDecode(); // stop main thread of decode

	DestroyDecoder(); // destroy decoder

	DestroyValues(); // destroy values

	m_file.CloseFile(); // close reading DN2 file
}

int DecodeDaniel2::OpenFile(const char* const filename, size_t iMaxCountDecoders)
{ 
	m_bInitDecoder = false;

	int res = m_file.OpenFile(filename); // open input DN2 file

	iMaxCountDecoders = std::max((size_t)1, std::min(iMaxCountDecoders, (size_t)4)); // 1..4
	
	if (res == 0)
		res = CreateDecoder(iMaxCountDecoders); // create decoders

	unsigned char* coded_frame = nullptr;
	size_t coded_frame_size = 0;

	if (res == 0)
	{
		m_buffer.resize(m_width * m_height); // set maximum of size for coded frame

		coded_frame_size = 0;
		res = m_file.ReadFrame(0, m_buffer, coded_frame_size); // get 0-coded frame for add decoder and init values after decode first frame
	}

	if (res == 0) 
	{
		coded_frame = m_buffer.data(); // poiter to 0-coded frame

		HRESULT hr = S_OK;

		m_eventInitDecoder.Reset();

		for (size_t i = 0; i < 3; i++)
		{
			if (FAILED(hr = m_pVideoDec->ProcessData(coded_frame, static_cast<CC_UINT>(coded_frame_size), 0, 0))) // add coded frame to decoder
			{
				assert(0);

				printf("ProcessData failed hr=%d coded_frame_size=%zu coded_frame=%p", hr, coded_frame_size, coded_frame);

				m_pVideoDec->Break(CC_FALSE); // break decoder

				DestroyDecoder(); // destroy decoder

				return hr;
			}
		}

		if (SUCCEEDED(hr)) 
			hr = m_pVideoDec->Break(CC_TRUE); // break decoder and flush data from decoder (call DataReady)

		if (!m_eventInitDecoder.Wait(10000) || !m_bInitDecoder) // wait 10 second for init decode
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
		printf("filename: %s\n", filename);
		printf("width x height: %zu x %zu\n", m_width, m_height);
		switch (m_outputImageFormat)
		{
		case IMAGE_FORMAT_RGBA8BIT: printf("format image: RGBA 8bit\n"); break;
		case IMAGE_FORMAT_RGBA16BIT: printf("format image: RGBA 16bit\n"); break;
		case IMAGE_FORMAT_RGB30: printf("format image: RGB 30bit\n"); break;
		}
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

	m_hExitEvent.Set();

	Close(); // closing thread <ThreadProc>

	return 0;
}

C_Block* DecodeDaniel2::MapFrame()
{
	C_Block *pBlock = nullptr;

	m_queueFrames.Get(&pBlock, m_hExitEvent); // receiving a block (C_Block) from a queue of finished decoded frames

	return pBlock;
}

void DecodeDaniel2::UnmapFrame(C_Block* pBlock)
{
	if (pBlock)
	{
		m_queueFrames_free.Queue(pBlock); // adding a block (C_Block) to the queue for processing (free frame queue)
	}
}

int DecodeDaniel2::CreateDecoder(size_t iMaxCountDecoders)
{
	HRESULT hr = S_OK;

	com_ptr<ICC_ClassFactory> piFactory;

	Cinecoder_CreateClassFactory((ICC_ClassFactory**)&piFactory); // get Factory
	if (FAILED(hr)) return hr;

	hr = piFactory->AssignLicense(COMPANYNAME, LICENSEKEY); // set license
	if (FAILED(hr)) return hr;

	CC_VERSION_INFO version = Cinecoder_GetVersion(); // get version of Cinecoder

	std::string strCinecoderVersion;

	strCinecoderVersion = "Cinecoder version: ";
	strCinecoderVersion += std::to_string((long long)version.VersionHi);
	strCinecoderVersion += ".";
	strCinecoderVersion += std::to_string((long long)version.VersionLo);
	strCinecoderVersion += ".";
	strCinecoderVersion += std::to_string((long long)version.EditionNo);
	strCinecoderVersion += ".";
	strCinecoderVersion += std::to_string((long long)version.RevisionNo);

	printf("%s\n", strCinecoderVersion.c_str()); // print version of Cinecoder

	hr = piFactory->CreateInstance(CLSID_CC_DanielVideoDecoder, IID_ICC_DanielVideoDecoder, (IUnknown**)&m_pVideoDec); // create CPU decoder
	
	if (FAILED(hr))
	{
		printf("DecodeDaniel2: CreateInstance failed!");
		return hr;
	}

	com_ptr<ICC_ProcessDataPolicyProp> pPolicy;
	m_pVideoDec->QueryInterface(IID_ICC_ProcessDataPolicyProp, (void**)&pPolicy);

	if (FAILED(hr = pPolicy->put_ProcessDataPolicy(CC_PDP_PARSED_DATA)))
	{
		printf("DecodeDaniel2: put_ProcessDataPolicy failed!");
		return hr;
	}

	com_ptr<ICC_ConcurrencyLevelProp> pConcur;
	m_pVideoDec->QueryInterface(IID_ICC_ConcurrencyLevelProp, (void**)&pConcur);

	if (FAILED(hr = pConcur->put_ConcurrencyLevel(static_cast<CC_AMOUNT>(iMaxCountDecoders)))) // set count of decoders in carousel of decoders
	{
		printf("DecodeDaniel2: put_ConcurrencyLevel failed!");
		return hr;
	}

	hr = m_pVideoDec->put_OutputCallback((ICC_DataReadyCallback *)this); // set output callback

	if (FAILED(hr))
	{
		printf("DecodeDaniel2: put_OutputCallback failed!");
		return hr;
	}

	hr = m_pVideoDec->Init(); // init decoder

	if (FAILED(hr))
	{
		printf("DecodeDaniel2: Init failed!");
		return hr;
	}

	return 0;
}

int DecodeDaniel2::DestroyDecoder()
{
	if (m_pVideoDec != nullptr)
		m_pVideoDec->Done(CC_FALSE); // call done for decoder with param CC_FALSE (without flush data to DataReady)

	m_pVideoDec = nullptr;

	return 0;
}

int DecodeDaniel2::InitValues()
{
	size_t iCountBlocks = 4; // set count of blocks in queue

	int res = 0;

	for (size_t i = 0; i < iCountBlocks; i++) // allocating memory of list of blocks (C_Block)
	{
		m_listBlocks.push_back(C_Block());

		m_listBlocks.back().Init(m_width, m_height, m_stride);

		if (res != 0)
		{
			printf("InitBlocks: Init() return error - %d", res);
			return res;
		}

		m_queueFrames_free.Queue(&m_listBlocks.back()); // add free pointers to queue
	}

	return 0;
}

int DecodeDaniel2::DestroyValues()
{
	// clear all queues and lists

	m_queueFrames.Free();
	m_queueFrames_free.Free();

	m_listBlocks.clear();

	return 0;
}

HRESULT STDMETHODCALLTYPE DecodeDaniel2::DataReady(IUnknown *pDataProducer)
{
	HRESULT hr = S_OK;
	
	com_ptr<ICC_VideoProducer> pVideoProducer;

	if (FAILED(hr = m_pVideoDec->QueryInterface(IID_ICC_VideoProducer, (void**)&pVideoProducer))) // get Producer
		return hr;

	///////////////////////////////////////////////
	// getting information about the decoded frame
	///////////////////////////////////////////////

	com_ptr<ICC_VideoFrameInfo> pCCFrameInfo;
	com_ptr<ICC_VideoStreamInfo> pCCStreamInfo;

	hr = pVideoProducer->GetVideoFrameInfo((ICC_VideoFrameInfo**)&pCCFrameInfo);
	hr = pVideoProducer->GetVideoStreamInfo((ICC_VideoStreamInfo**)&pCCStreamInfo);

	com_ptr<ICC_DanielVideoStreamInfo> pDanielVideoStreamInfo;
	hr = pCCStreamInfo->QueryInterface(IID_ICC_DanielVideoStreamInfo, (void**)&pDanielVideoStreamInfo);

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

	if (SUCCEEDED(hr)) hr = pDanielVideoStreamInfo->get_BitDepth(&BitDepth);
	if (SUCCEEDED(hr)) hr = pDanielVideoStreamInfo->get_ChromaFormat(&ChromaFormat);
	if (SUCCEEDED(hr)) hr = pDanielVideoStreamInfo->get_ColorCoefs(&ColorCoefs);
	if (SUCCEEDED(hr)) hr = pDanielVideoStreamInfo->get_FrameRate(&FrameRate);
	if (SUCCEEDED(hr)) hr = pDanielVideoStreamInfo->get_FrameSize(&FrameSize);
	if (SUCCEEDED(hr)) hr = pDanielVideoStreamInfo->get_PictureOrientation(&PictureOrientation);

	com_ptr<ICC_DanielVideoFrameInfo> pDanielVideoFrameInfo;
	hr = pCCFrameInfo->QueryInterface(IID_ICC_DanielVideoFrameInfo, (void**)&pDanielVideoFrameInfo);

	if (SUCCEEDED(hr)) hr = pDanielVideoFrameInfo->get_CodingMethod(&CodingMethod);
	if (SUCCEEDED(hr)) hr = pDanielVideoFrameInfo->get_QuantScale(&QuantScale);
	if (SUCCEEDED(hr)) hr = pDanielVideoFrameInfo->get_CodingNumber(&CodingNumber);
	if (SUCCEEDED(hr)) hr = pDanielVideoFrameInfo->get_PTS(&PTS);

	///////////////////////////////////////////////

	if (m_bProcess)
	{
		C_Block *pBlock = nullptr;

		m_queueFrames_free.Get(&pBlock, m_hExitEvent); // get free pointer to object of C_Block form queue

		if (pBlock)
		{
			pBlock->SetRotate((PictureOrientation == CC_PO_FLIP_VERTICAL || PictureOrientation == CC_PO_ROTATED_180DEG) ? true : false);  // rotate frame

			DWORD cb = 0;

			hr = pVideoProducer->GetFrame(m_fmt, pBlock->DataPtr(), (DWORD)pBlock->Size(), (INT)pBlock->Pitch(), &cb); // get decoded frame from Cinecoder

			pBlock->iFrameNumber = static_cast<size_t>(PTS); // save PTS (in our case this is the frame number)

			m_queueFrames.Queue(pBlock); // add pointer to object of C_Block with final picture to queue
		}
	}
	else if (!m_bInitDecoder) // init values after first decoding frame
	{
		m_width = FrameSize.cx; // get width
		m_height = FrameSize.cy; // get height

		//CC_COLOR_FMT fmt = BitDepth == 8 ? CCF_RGB32 : CCF_RGB64;

		//if (BitDepth == 10)
		//	fmt = CCF_RGB30;

		CC_COLOR_FMT fmt = CCF_BGR32;//CCF_RGB30; // set output format

		CC_BOOL bRes = CC_FALSE;
		pVideoProducer->IsFormatSupported(fmt, &bRes);
		if (bRes)
		{
			DWORD iStride = 0;
			pVideoProducer->GetStride(fmt, &iStride); // get stride
			m_stride = (size_t)iStride;

			m_fmt = fmt;

			if (m_fmt == CCF_RGB32 || m_fmt == CCF_BGR32)
				m_outputImageFormat = IMAGE_FORMAT_RGBA8BIT;
			else if (m_fmt == CCF_RGB64)
				m_outputImageFormat = IMAGE_FORMAT_RGBA16BIT;
			else if (m_fmt == CCF_RGB30)
				m_outputImageFormat = IMAGE_FORMAT_RGB30;

			m_bInitDecoder = true; // set init decoder value
		}

		m_eventInitDecoder.Set(); // set event about decoder was initialized
	}

	return hr;
}

long DecodeDaniel2::ThreadProc()
{
	m_bProcess = true;

	unsigned char* coded_frame = nullptr;
	size_t coded_frame_size = 0;
	size_t frame_number = 0;
	
	HRESULT hr = S_OK;

	while (m_bProcess)
	{
		if (!m_bPause)
		{
			CodedFrame* frame = nullptr;
			frame = m_file.MapFrame(); // get pointer to next coded frame form DN2 file

			if (frame)
			{
				coded_frame = frame->coded_frame.data(); // poiter to coded frame
				coded_frame_size = frame->coded_frame_size; // size of coded frame
				frame_number = frame->frame_number; // number of coded frame

				if (FAILED(hr = m_pVideoDec->ProcessData(coded_frame, static_cast<CC_UINT>(coded_frame_size), 0, frame_number))) // add coded frame to decoder
				{
					assert(0);

					printf("ProcessData failed hr=%d coded_frame_size=%zu coded_frame=%p", hr, coded_frame_size, coded_frame);

					m_pVideoDec->Break(CC_FALSE); // break decoder with param CC_FALSE (without flush data to DataReady)
				}

				m_file.UnmapFrame(frame); // add to queue free pointer for reading coded frame
			}
		}
		else
		{
			C_Block *pBlock = nullptr;

			m_queueFrames_free.Get(&pBlock, m_hExitEvent); // get free pointer to object of C_Block form queue

			if (pBlock)
			{
				m_queueFrames.Queue(pBlock); // add pointer to object of C_Block with final picture to queue
			}
		}
	}

	m_pVideoDec->Break(CC_FALSE); // break decoder with param CC_FALSE (without flush data to DataReady)

	m_file.StopPipe(); // stop pipeline for reading DN2 file

	return 0;
}
