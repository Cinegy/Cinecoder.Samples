#include "stdafx.h"

#include <iostream>
#include "CEncoderTest.h"

//---------------------------------------------------------------
CEncoderTest::CEncoderTest()
//---------------------------------------------------------------
{
	m_bRunning = FALSE;
	m_NumActiveThreads = 0;
}

//---------------------------------------------------------------
CEncoderTest::~CEncoderTest()
//---------------------------------------------------------------
{
	Close();
}

//---------------------------------------------------------------
int CEncoderTest::CheckParameters(const TEST_PARAMS &par)
//---------------------------------------------------------------
{
	return 0;
}

//---------------------------------------------------------------
int CEncoderTest::CreateEncoder(const TEST_PARAMS &par)
//---------------------------------------------------------------
{
	int hr;

	if (FAILED(hr = Cinecoder_CreateClassFactory(&m_pFactory)))
		return print_error(hr, "Cinecoder factory creation error");

	if (FAILED(hr = m_pFactory->AssignLicense(par.CompanyName, par.LicenseKey)))
		return print_error(hr, "AssignLicense error");

	com_ptr<ICC_VideoEncoder> pEncoder;
	CLSID clsidEnc = par.DeviceId >= 0 ? CLSID_CC_DanielVideoEncoder_CUDA : CLSID_CC_DanielVideoEncoder;
	if (FAILED(hr = m_pFactory->CreateInstance(clsidEnc, IID_ICC_VideoEncoder, (IUnknown**)&pEncoder)))
		return print_error(hr, "Encoder creation error");

	// We use ICC_DanielVideoEncoderSettings_CUDA settings for both encoders because _CUDA settings class is an inheritor of ICC_DanielVideoEncoderSettings.
	com_ptr<ICC_DanielVideoEncoderSettings_CUDA> pSettings;
	if (FAILED(hr = m_pFactory->CreateInstance(CLSID_CC_DanielVideoEncoderSettings_CUDA, IID_ICC_DanielVideoEncoderSettings_CUDA, (IUnknown**)&pSettings)))
		return print_error(hr, "Encoder settings creation error");

	pSettings->put_FrameSize(MK_SIZE(par.Width, par.Height));
	pSettings->put_FrameRate(MK_RATIONAL(par.FrameRateN, par.FrameRateD));
	pSettings->put_InputColorFormat(par.InputColorFormat);

	auto ChromaFormat = par.ChromaFormat;
	if(!ChromaFormat)
	{
	  if(par.InputColorFormat == CCF_YUY2 || par.InputColorFormat == CCF_UYVY || par.InputColorFormat == CCF_V210)
	    ChromaFormat = CC_CHROMA_422;
	  else
	    ChromaFormat = CC_CHROMA_RGBA;
	}
	pSettings->put_ChromaFormat(ChromaFormat);

	pSettings->put_BitDepth(par.BitDepth);
	pSettings->put_PictureOrientation(par.PictureOrientation);

	pSettings->put_RateMode(par.BitrateMode);
	pSettings->put_BitRate(par.Bitrate);
	pSettings->put_QuantScale(par.QuantScale);
	pSettings->put_CodingMethod(par.CodingMethod);

	pSettings->put_DeviceID(par.DeviceId);
	pSettings->put_NumSingleEncoders(par.NumSingleEncoders);

	if (FAILED(hr = pEncoder->Init(pSettings)))
		return print_error(hr, "Encoder initialization error");

	com_ptr<ICC_OutputFile> pFileWriter;
	if (FAILED(hr = m_pFactory->CreateInstance(CLSID_CC_OutputFile, IID_ICC_OutputFile, (IUnknown**)&pFileWriter)))
		return print_error(hr, "File writer creation error");

	const TCHAR *pext = _tcsrchr(par.OutputFileName, '.');
	if(!pext || (_tcsicmp(pext, _T(".DN2")) != 0 && _tcsicmp(pext, _T(".MXF")) != 0))
		return print_error(E_FAIL, "Unrecognized output file type");

	if (FAILED(hr = pFileWriter->Create(const_cast<TCHAR*>(par.OutputFileName))))
		return print_error(hr, "Output file creation error");

	if (_tcsicmp(pext, _T(".MXF")) == 0)
	{
		if(FAILED(hr = m_pFactory->LoadPlugin(LPTSTR(_T("Cinecoder.Plugin.Multiplexers.dll")))))
			return print_error(hr, "Error loading the MXF plugin");

		if (FAILED(hr = m_pFactory->CreateInstance(CLSID_CC_MXF_OP1A_Multiplexer, IID_ICC_Multiplexer, (IUnknown**)&m_pMuxer)))
			return print_error(hr, "Failed to create MXF OP1A multipexer");

		if(FAILED(m_pMuxer->Init(NULL)))
			return print_error(hr, "Failed to init the MXF multipexer");

		com_ptr<ICC_MXF_MultiplexerPinSettings> pMuxPinSettings;
		if (FAILED(hr = m_pFactory->CreateInstance(CLSID_CC_MXF_MultiplexerPinSettings, IID_ICC_MXF_MultiplexerPinSettings, (IUnknown**)&pMuxPinSettings)))
			return print_error(hr, "Failed to create MXF multipexer pin settings");

		//if (par.BitrateMode != CC_CBR)
		//	return print_error(E_INVALIDARG, "MXF can only work in CBR mode");

		pMuxPinSettings->put_StreamType(CC_ES_TYPE_VIDEO_DANIEL);
		pMuxPinSettings->put_BitRate(par.Bitrate);
		pMuxPinSettings->put_FrameRate(MK_RATIONAL(par.FrameRateN, par.FrameRateD));

		com_ptr<ICC_ByteStreamConsumer> pMuxerPin;
		if(FAILED(hr = m_pMuxer->CreatePin(pMuxPinSettings, &pMuxerPin)))
			return print_error(hr, "Failed to create MXF Multiplexer Daniel2 PIN");

		if (FAILED(hr = pEncoder->put_OutputCallback(pMuxerPin)))
			return print_error(hr, "Encoder cb assignment error");

		if(FAILED(hr = m_pMuxer->put_OutputCallback(pFileWriter)))
			return print_error(hr, "Muxer cb assignment error");
	}
	else if (false) // easy write mode - no index file
	{
		if (FAILED(hr = pEncoder->put_OutputCallback(pFileWriter)))
			return print_error(hr, "Encoder cb assignment error");
	}
	else // write with index file
	{
		com_ptr<ICC_IndexWriter> pIndexWriter;
		if (FAILED(hr = m_pFactory->CreateInstance(CLSID_CC_MvxWriter, IID_ICC_IndexWriter, (IUnknown**)&pIndexWriter)))
			return print_error(hr, "Index writer creation error");

		if (FAILED(hr = pEncoder->put_OutputCallback(pIndexWriter)))
			return print_error(hr, "Encoder cb assignment error");

		if (FAILED(hr = pIndexWriter->put_OutputCallback(pFileWriter)))
			return print_error(hr, "Index writer cb assignment error");

		com_ptr<ICC_OutputFile> pIndexFileWriter;
		if (FAILED(hr = m_pFactory->CreateInstance(CLSID_CC_OutputFile, IID_ICC_OutputFile, (IUnknown**)&pIndexFileWriter)))
			return print_error(hr, "Index file writer creation error");

		com_ptr<ICC_DataWriterEx> pDataWriterEx;
		if(FAILED(hr = pIndexFileWriter->QueryInterface(IID_ICC_DataWriterEx, (void**)&pDataWriterEx)))
			return print_error(hr, "ICC_File has no ICC_DataWriterEx interface");

		if (FAILED(hr = pIndexWriter->put_IndexCallback(pDataWriterEx)))
			return print_error(hr, "Index writer cb 2 assignment error");

		// making a proper filename for the video index
		TCHAR mvx_name[MAX_PATH];
		_tcscpy(mvx_name, par.OutputFileName);
		_tcscpy(_tcsrchr(mvx_name, '.'), _T(".mvx"));

		if (FAILED(hr = pIndexFileWriter->Create(mvx_name)))
			return print_error(hr, "Output file creation error");
	}

	m_pEncoder = pEncoder;

	return S_OK;
}

//---------------------------------------------------------------
int	CEncoderTest::AssignParameters(const TEST_PARAMS &par)
//---------------------------------------------------------------
{
	int	hr;

	if(FAILED(hr = CheckParameters(par)))
    	return hr;
	
	if(FAILED(hr = CreateEncoder(par)))
    	return hr;

	m_FrameSizeInBytes = 0;// par.ReadBlockSize;

	if(par.SetOfFiles)
	{
		m_FrameSizeInBytes = par.FileSize;
	}
	else switch (par.InputColorFormat)
	{
	case CCF_UYVY:
	case CCF_YUY2:
		m_FrameSizeInBytes = par.Height * par.Width * 2;
		break;
	
	case CCF_V210:
		m_FrameSizeInBytes = par.Height * ((par.Width + 47) / 48) * 128;
		break;

	case CCF_RGB30:
		m_FrameSizeInBytes = par.Height * par.Width * 4;
		break;

	case CCF_RGB48:
		m_FrameSizeInBytes = par.Height * par.Width * 6;
		break;

	default:
		return print_error(E_INVALIDARG, "The color format is not supported by test app"), E_INVALIDARG;
	}

	for(int i = 0; i < par.QueueSize; i++)
	{ 
		BufferDescr descr = {};
		
		extern void *cuda_alloc_pinned(size_t size);

		int page_aligned_size = (m_FrameSizeInBytes + 4095) & ~4095;

		if(par.DeviceId >= 0)
			descr.pBuffer = (LPBYTE)cuda_alloc_pinned(page_aligned_size);
		else
#ifdef _WIN32
			descr.pBuffer = (LPBYTE)VirtualAlloc(NULL, page_aligned_size, MEM_COMMIT, PAGE_READWRITE);
#elif defined(__APPLE__)
			descr.pBuffer = (LPBYTE)malloc(page_aligned_size);
#else
			descr.pBuffer = (LPBYTE)aligned_alloc(4096, page_aligned_size);
#endif		
		if (!descr.pBuffer)
			return print_error(E_OUTOFMEMORY, "Can't allocate an input frame buffer for the queue");
		
		descr.evVacant = new waiter();
		descr.evFilled = new waiter();
		descr.bFilled  = false;
		
		m_Queue.push_back(descr);
	}

	m_EncPar = par;

	return S_OK;
}

//---------------------------------------------------------------
int	CEncoderTest::Close()
//---------------------------------------------------------------
{
	if (!m_pEncoder)
		return S_FALSE;

	if (m_bRunning)
		Cancel();

	while(!m_Queue.empty())
	{
		BufferDescr descr = m_Queue.back();

		delete descr.evVacant;
		delete descr.evFilled;

		extern void cuda_free_pinned(void *ptr);

		if(m_EncPar.DeviceId >= 0)
			cuda_free_pinned(descr.pBuffer);
		else
#ifdef _WIN32
			VirtualFree(descr.pBuffer, 0, MEM_RELEASE);
#else
			free(descr.pBuffer);
#endif
		m_Queue.pop_back();
	}

	m_pEncoder->Done(CC_TRUE);

	if (m_pMuxer)
		m_pMuxer->Done(CC_TRUE);

	m_pEncoder = NULL;
	m_pMuxer = NULL;
	m_pFactory = NULL;

	return S_OK;
}

//---------------------------------------------------------------
int		CEncoderTest::Run()
//---------------------------------------------------------------
{
	m_NumActiveThreads = m_EncPar.NumReadThreads + 1;
	m_ReadFrameCounter = 0;
	m_hrResult         = S_OK;

	m_EncodingThread = std::thread(encoding_thread_proc, this);

	for (int i = 0; i < m_EncPar.NumReadThreads; i++)
		m_ReadingThreads.push_back(std::thread(reading_thread_proc, this, i));

	m_bRunning = TRUE;

	return S_OK;
}

//---------------------------------------------------------------
bool	CEncoderTest::IsActive() const
//---------------------------------------------------------------
{
	return m_NumActiveThreads != 0;
}
//---------------------------------------------------------------
HRESULT CEncoderTest::GetResult() const
//---------------------------------------------------------------
{
	return m_hrResult;
}

//---------------------------------------------------------------
int		CEncoderTest::Cancel()
//---------------------------------------------------------------
{
	if (!m_bRunning)
		return S_FALSE;

	m_bCancel = true;

	for(size_t i = 0; i < m_ReadingThreads.size(); i++)
		m_Queue[i].evVacant->cond_var.notify_one();

	for(size_t i = 0; i < m_ReadingThreads.size(); i++)
		m_ReadingThreads[i].join();

	m_ReadingThreads.clear();

	m_EncodingThread.join();

	m_bRunning = false;

	return S_OK;
}

//---------------------------------------------------------------
int		CEncoderTest::GetCurrentEncodingStats(ENCODER_STATS *pStats)
//---------------------------------------------------------------
{
	if (!pStats) return E_POINTER;

	m_StatsLock.lock();
	*pStats = m_Stats;
	m_StatsLock.unlock();

	return S_OK;
}

//---------------------------------------------------------------
DWORD	CEncoderTest::reading_thread_proc(void *p, int thread_idx)
//---------------------------------------------------------------
{
	return reinterpret_cast<CEncoderTest*>(p)->ReadingThreadProc(thread_idx);
}

#include "filework.h"

//---------------------------------------------------------------
DWORD 	CEncoderTest::ReadingThreadProc(int thread_idx)
//---------------------------------------------------------------
{
    fprintf(stderr, "Reading thread %d is started\n", thread_idx);

    file_handle_t hFile = INVALID_FILE_HANDLE;

    if(!m_EncPar.SetOfFiles)
	    hFile = open_file(m_EncPar.InputFileName, !m_EncPar.UseCache);

	BufferDescr *pbufdescr = nullptr;
    
	bool forward_read = m_EncPar.StopFrameNum < 0 || m_EncPar.StopFrameNum >= m_EncPar.StartFrameNum;

    for(;;)
    {
		int frame_idx = m_ReadFrameCounter++;
    	int buffer_id = frame_idx % m_EncPar.QueueSize;

		pbufdescr = &m_Queue[buffer_id];

		{
			std::unique_lock<std::mutex> lck(pbufdescr->evVacant->mutex);

			while(!m_bCancel && pbufdescr->bOccupied)
				pbufdescr->evVacant->cond_var.wait(lck);

			pbufdescr->bOccupied = true;
		}

		if(m_bCancel)
		{
			pbufdescr->hrReadStatus = S_FALSE;
			break;
		}

		int frame_no = -1;

		if (forward_read)
		{
			frame_no = m_EncPar.StartFrameNum + frame_idx;

			if (m_EncPar.StopFrameNum >= 0)
			{
				if (frame_no > m_EncPar.StopFrameNum && !m_EncPar.Looped)
				{
					pbufdescr->hrReadStatus = S_FALSE;
					break;
				}

				frame_no = m_EncPar.StartFrameNum + frame_idx % (m_EncPar.StopFrameNum - m_EncPar.StartFrameNum + 1);
			}
		}
		else
		{
			frame_no = m_EncPar.StartFrameNum - frame_idx;

			if (frame_no < m_EncPar.StopFrameNum && !m_EncPar.Looped)
			{
				pbufdescr->hrReadStatus = S_FALSE;
				break;
			}

			frame_no = m_EncPar.StartFrameNum - frame_idx % (m_EncPar.StartFrameNum - m_EncPar.StopFrameNum + 1);
		}

	    if(m_EncPar.SetOfFiles)
	    {
	    	TCHAR filename[MAX_PATH];
	    	_stprintf(filename, m_EncPar.InputFileName, frame_no);
	        hFile = open_file(filename, !m_EncPar.UseCache);
//	        _ftprintf(stderr, _T("openfile(%s)=%d\n"), filename, hFile);
	    }
	    else
	    {
			LONGLONG offset = frame_no * LONGLONG(m_FrameSizeInBytes);
			set_file_pos(hFile, offset);
		}

		if (hFile == INVALID_FILE_HANDLE)
		{
			pbufdescr->hrReadStatus = S_FALSE;
			break;
		}

		DWORD r = 0;
		if (!read_file(hFile, pbufdescr->pBuffer, (m_FrameSizeInBytes + 4095) & ~4095, &r))
		{
#ifdef _WIN32
			pbufdescr->hrReadStatus = HRESULT_FROM_WIN32(GetLastError());
#else
			pbufdescr->hrReadStatus = HRESULT(errno | 0x80000000u);
//	        _ftprintf(stderr, _T("readfile(%d,%p,%d)=%d\n"), hFile, pbufdescr->pBuffer, (m_FrameSizeInBytes + 4095) & ~4095, r);
#endif
			break;
		}

		if (r != m_FrameSizeInBytes)
		{
			pbufdescr->hrReadStatus = S_FALSE;
			break;
		}

	    if(m_EncPar.SetOfFiles)
	    {
	    	close_file(hFile);
	    }

		m_StatsLock.lock();
		m_Stats.NumFramesRead++;
		m_Stats.NumBytesRead += m_FrameSizeInBytes;
		m_StatsLock.unlock();

		std::unique_lock<std::mutex> lck(pbufdescr->evFilled->mutex);
		pbufdescr->bFilled = true;
		pbufdescr->evFilled->cond_var.notify_one();
	}

	// we have to notify the waiter
	std::unique_lock<std::mutex> lck(pbufdescr->evFilled->mutex);
	pbufdescr->bFilled = true;
	pbufdescr->evFilled->cond_var.notify_one();

	HRESULT hr = pbufdescr->hrReadStatus;

    if(FAILED(hr))
    {
      fprintf(stderr, "Reading thread %d: error %08x\n", thread_idx, hr);
      m_bCancel = true;
    }

    close_file(hFile);
    m_NumActiveThreads--;

    fprintf(stderr, "Reading thread %d is done\n", thread_idx);

    return hr;
}

//---------------------------------------------------------------
DWORD	CEncoderTest::encoding_thread_proc(void *p)
//---------------------------------------------------------------
{
	return reinterpret_cast<CEncoderTest*>(p)->EncodingThreadProc();
}

//---------------------------------------------------------------
DWORD	CEncoderTest::EncodingThreadProc()
//---------------------------------------------------------------
{
	fprintf(stderr, "Encoding thread is started\n");

	for(int frame_no = 0; ; frame_no++)
	{
		const int buffer_id = frame_no % m_EncPar.QueueSize;

		BufferDescr &bufdescr = m_Queue[buffer_id];

		auto t0 = system_clock::now();

		{
			std::unique_lock<std::mutex> lck(bufdescr.evFilled->mutex);

			while(!m_bCancel && !bufdescr.bFilled)
				bufdescr.evFilled->cond_var.wait(lck);
		}

		if (m_bCancel)
		{
			m_hrResult = S_FALSE;// E_ABORT;
			break;
		}

		if (bufdescr.hrReadStatus != S_OK)
		{
			m_hrResult = bufdescr.hrReadStatus;
			break;
		}

		CC_VIDEO_FRAME_DESCR frame_descr = {};
		frame_descr.cFormat = m_EncPar.InputColorFormat;
		frame_descr.iStride = m_EncPar.InputPitch;
#if 0
		com_ptr<ICC_VideoConsumerExtAsync> pEncAsync;
		if (SUCCEEDED(m_pEncoder->QueryInterface(IID_ICC_VideoConsumerExtAsync, (void**)&pEncAsync)))
		{
			m_hrResult = pEncAsync->AddScaleFrameAsync(
				bufdescr.pBuffer + m_EncPar.DataOffset,
				m_FrameSizeInBytes - m_EncPar.DataOffset,
				&frame_descr,
				com_ptr<IUnknown>(),
				nullptr);
		}
		else
#endif
		{
			m_hrResult = m_pEncoder->AddScaleFrame(
				bufdescr.pBuffer + m_EncPar.DataOffset,
				m_FrameSizeInBytes - m_EncPar.DataOffset,
				&frame_descr,
				NULL);
		}

		auto t1 = system_clock::now();
		auto dT = duration_cast<milliseconds>(t1 - t0).count();

//		int wait_time = 42 - dT;
//		if (wait_time > 1)
//			Sleep(wait_time);

		m_StatsLock.lock();
		m_Stats.NumFramesWritten++;
		m_Stats.NumBytesWritten += 0;
		m_StatsLock.unlock();

		std::unique_lock<std::mutex> lck(bufdescr.evVacant->mutex);
		bufdescr.bFilled = false;
		bufdescr.bOccupied = false;
		bufdescr.evVacant->cond_var.notify_one();

	    if(FAILED(m_hrResult))
	    {
	      fprintf(stderr, "Encoding thread error %08x\n", m_hrResult);
	      m_bCancel = true;
	      break;
	    }
	}

	m_NumActiveThreads --;
 	fprintf(stderr, "Encoding thread is done\n");

	return m_hrResult;
}
