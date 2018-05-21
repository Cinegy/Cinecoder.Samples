#include "stdafx.h"

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

	CComPtr<ICC_ClassFactory> pFactory;
	if (FAILED(hr = Cinecoder_CreateClassFactory(&pFactory)))
		return print_error(hr, "Cinecoder factory creation error");

	if (FAILED(hr = pFactory->AssignLicense(par.CompanyName, par.LicenseKey)))
		return print_error(hr, "AssignLicense error");

	CComPtr<ICC_VideoEncoder> pEncoder;
	CLSID clsidEnc = par.DeviceId >= 0 ? CLSID_CC_DanielVideoEncoder_CUDA : CLSID_CC_DanielVideoEncoder;
	if (FAILED(hr = pFactory->CreateInstance(clsidEnc, IID_ICC_VideoEncoder, (IUnknown**)&pEncoder)))
		return print_error(hr, "Encoder creation error");

	// We use ICC_DanielVideoEncoderSettings_CUDA settings for both encoders because _CUDA settings class is an inheritor of ICC_DanielVideoEncoderSettings.
	CComPtr<ICC_DanielVideoEncoderSettings_CUDA> pSettings;
	if (FAILED(hr = pFactory->CreateInstance(CLSID_CC_DanielVideoEncoderSettings_CUDA, IID_ICC_DanielVideoEncoderSettings_CUDA, (IUnknown**)&pSettings)))
		return print_error(hr, "Encoder settings creation error");

	pSettings->put_FrameSize(MK_SIZE(par.Width, par.Height));
	pSettings->put_FrameRate(MK_RATIONAL(par.FrameRateN, par.FrameRateD));
	pSettings->put_InputColorFormat(par.InputColorFormat);

	pSettings->put_ChromaFormat(par.ChromaFormat);
	pSettings->put_BitDepth(par.BitDepth);

	pSettings->put_RateMode(par.BitrateMode);
	pSettings->put_BitRate(par.Bitrate);
	pSettings->put_QuantScale(par.QuantScale);
	pSettings->put_CodingMethod(par.CodingMethod);

	pSettings->put_DeviceID(par.DeviceId);
	pSettings->put_NumSingleEncoders(par.NumSingleEncoders);

	if (FAILED(hr = pEncoder->Init(pSettings)))
		return print_error(hr, "Encoder initialization error");

	CComPtr<ICC_OutputFile> pFileWriter;
	if (FAILED(hr = pFactory->CreateInstance(CLSID_CC_OutputFile, IID_ICC_OutputFile, (IUnknown**)&pFileWriter)))
		return print_error(hr, "File writer creation error");

	if (FAILED(hr = pFileWriter->Create(CComBSTR(par.OutputFileName))))
		return print_error(hr, "Output file creation error");
#if 0
	if (FAILED(hr = pEncoder->put_OutputCallback(pFileWriter)))
		return print_error(hr, "Encoder cb assignment error");
#else
	CComPtr<ICC_IndexWriter> pIndexWriter;
	if (FAILED(hr = pFactory->CreateInstance(CLSID_CC_MvxWriter, IID_ICC_IndexWriter, (IUnknown**)&pIndexWriter)))
		return print_error(hr, "Index writer creation error");

	if(FAILED(hr = pEncoder->put_OutputCallback(pIndexWriter)))
		return print_error(hr, "Encoder cb assignment error");

	if(FAILED(hr = pIndexWriter->put_OutputCallback(pFileWriter)))
		return print_error(hr, "Index writer cb assignment error");

	CComPtr<ICC_OutputFile> pIndexFileWriter;
	if (FAILED(hr = pFactory->CreateInstance(CLSID_CC_OutputFile, IID_ICC_OutputFile, (IUnknown**)&pIndexFileWriter)))
		return print_error(hr, "Index file writer creation error");

	CComQIPtr<ICC_DataWriterEx> pDataWriterEx = pIndexFileWriter;
	if(!pDataWriterEx)
		return print_error(hr, "ICC_File has no ICC_DataWriterEx interface");

	if (FAILED(hr = pIndexWriter->put_IndexCallback(pDataWriterEx)))
		return print_error(hr, "Index writer cb 2 assignment error");

	// making a proper filename for the video index
	TCHAR drive[MAX_PATH], dir[MAX_PATH], name[MAX_PATH], ext[MAX_PATH];
	_tsplitpath(par.OutputFileName, drive, dir, name, ext);
	TCHAR mvx_name[MAX_PATH];
	_tmakepath(mvx_name, drive, dir, name, _T(".mvx"));

	if (FAILED(hr = pIndexFileWriter->Create(CComBSTR(mvx_name))))
		return print_error(hr, "Output file creation error");
#endif
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

	switch (par.InputColorFormat)
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
	default:
		return print_error(E_INVALIDARG, "The color format is not supported by test app"), E_INVALIDARG;
	}

	for(int i = 0; i < par.QueueSize; i++)
	{ 
		BufferDescr descr = {};
		
		extern void *cuda_alloc_pinned(size_t size);

		if(par.DeviceId >= 0)
			descr.pBuffer = (LPBYTE)cuda_alloc_pinned(m_FrameSizeInBytes);
		else
			descr.pBuffer = (LPBYTE)VirtualAlloc(NULL, m_FrameSizeInBytes, MEM_COMMIT, PAGE_READWRITE);
		
		if (!descr.pBuffer)
			return print_error(E_OUTOFMEMORY, "Can't allocate an input frame buffer for the queue");
		
		descr.evVacant = CreateEvent(NULL, FALSE, TRUE, NULL);
		descr.evFilled = CreateEvent(NULL, FALSE, FALSE, NULL);
		
		m_Queue.push_back(descr);
	}

	m_evCancel = CreateEvent(NULL, TRUE, FALSE, NULL);

	m_EncPar = par;

	memset((void*)&m_Stats, 0, sizeof(m_Stats));

	return S_OK;
}

//---------------------------------------------------------------
int	CEncoderTest::Close()
//---------------------------------------------------------------
{
	if (!m_pEncoder)
		return S_FALSE;

	if(m_bRunning)
		Cancel();

	while(!m_Queue.empty())
	{
		BufferDescr descr = m_Queue.back();

		CloseHandle(descr.evVacant);
		CloseHandle(descr.evFilled);

		extern void cuda_free_pinned(void *ptr);

		if(m_EncPar.DeviceId >= 0)
			cuda_free_pinned(descr.pBuffer);
		else
			VirtualFree(descr.pBuffer, 0, MEM_RELEASE);

		m_Queue.pop_back();
	}

	CloseHandle(m_evCancel);

	m_pEncoder = NULL;

	return S_OK;
}

//---------------------------------------------------------------
int		CEncoderTest::Run()
//---------------------------------------------------------------
{
	m_hEncodingThread = CreateThread(NULL, 0, encoding_thread_proc, this, 0, NULL);

	for (int i = 0; i < m_EncPar.NumReadThreads; i++)
		m_hReadingThreads.push_back(CreateThread(NULL, 0, reading_thread_proc, this, 0, NULL));

	m_NumActiveThreads = m_EncPar.NumReadThreads + 1;
	m_ReadFrameCounter = 0;
	m_hrResult = S_OK;
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

	SetEvent(m_evCancel);

	WaitForSingleObject(m_hEncodingThread, INFINITE);
	WaitForMultipleObjects((DWORD)m_hReadingThreads.size(), &m_hReadingThreads[0], TRUE, INFINITE);

	for(size_t i = 0; i < m_hReadingThreads.size(); i++)
		CloseHandle(m_hReadingThreads[i]);

	m_hReadingThreads.clear();

	CloseHandle(m_hEncodingThread);

	m_bRunning = false;

	return S_OK;
}

//---------------------------------------------------------------
int		CEncoderTest::GetCurrentEncodingStats(ENCODER_STATS *pStats)
//---------------------------------------------------------------
{
	if (!pStats) return E_POINTER;
	
	pStats->NumFramesRead    = m_Stats.NumFramesRead;
	pStats->NumFramesWritten = m_Stats.NumFramesWritten;
	pStats->NumBytesRead     = InterlockedExchangeAdd64(&m_Stats.NumBytesRead, 0);
	pStats->NumBytesWritten  = InterlockedExchangeAdd64(&m_Stats.NumBytesWritten, 0);

	return S_OK;
}

//---------------------------------------------------------------
DWORD	WINAPI	CEncoderTest::reading_thread_proc(void *p)
//---------------------------------------------------------------
{
	return reinterpret_cast<CEncoderTest*>(p)->ReadingThreadProc();
}

//---------------------------------------------------------------
DWORD 	CEncoderTest::ReadingThreadProc()
//---------------------------------------------------------------
{
    fprintf(stderr, "Reading thread %lu is started\n", GetCurrentThreadId());

    HANDLE hFile = CreateFile(m_EncPar.InputFileName, GENERIC_READ, FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, m_EncPar.UseCache ? 0 : FILE_FLAG_NO_BUFFERING, NULL);

    //if(hFile == INVALID_HANDLE_VALUE)
    //	return fprintf(stderr, "Thread %d: error %08xh opening the file\n", GetCurrentThreadId(), HRESULT_FROM_WIN32(GetLastError())), HRESULT_FROM_WIN32(GetLastError());

    for(;;)
    {
    	int frame_no = InterlockedExchangeAdd(&m_ReadFrameCounter, 1);
    	int buffer_id = frame_no % m_EncPar.QueueSize;

		BufferDescr &bufdescr = m_Queue[buffer_id];

    	HANDLE hh[2] = { m_evCancel, bufdescr.evVacant };

		DWORD wait_result = WaitForMultipleObjects(2, hh, FALSE, INFINITE);
		if(wait_result == WAIT_OBJECT_0)
			break;

		if (m_EncPar.StopFrameNum >= 0)
		{
			if (frame_no > m_EncPar.StopFrameNum && !m_EncPar.Looped)
			{
				bufdescr.hrReadStatus = S_FALSE;
				SetEvent(bufdescr.evFilled);
				break;
			}

			int num_frames_in_range = m_EncPar.StopFrameNum - m_EncPar.StartFrameNum + 1;

			if (num_frames_in_range > 0)
				frame_no = frame_no % num_frames_in_range + m_EncPar.StartFrameNum;
			else
				frame_no = m_EncPar.StopFrameNum - frame_no % (1-num_frames_in_range);
		}

		LONGLONG offset = frame_no * LONGLONG(m_FrameSizeInBytes);
		SetFilePointer(hFile, (LONG)offset, ((LONG*)&offset)+1, FILE_BEGIN);

		DWORD r;
		if (hFile == INVALID_HANDLE_VALUE || !ReadFile(hFile, bufdescr.pBuffer, (m_FrameSizeInBytes + 4095) & ~4095, &r, NULL))
		{
			bufdescr.hrReadStatus = HRESULT_FROM_WIN32(GetLastError());
			SetEvent(bufdescr.evFilled);
			break;
		}

		if (r != m_FrameSizeInBytes)
		{
			bufdescr.hrReadStatus = S_FALSE;
			SetEvent(bufdescr.evFilled);
			break;
		}

		InterlockedIncrement(&m_Stats.NumFramesRead);
		InterlockedAdd64(&m_Stats.NumBytesRead, m_FrameSizeInBytes);

		SetEvent(bufdescr.evFilled);
	}

	HRESULT hr = HRESULT_FROM_WIN32(GetLastError());

    if(FAILED(hr))
    {
      fprintf(stderr, "Reading thread %lu: error %08lx\n", GetCurrentThreadId(), hr);
      SetEvent(m_evCancel);
    }

    CloseHandle(hFile);
    InterlockedDecrement(&m_NumActiveThreads);

    fprintf(stderr, "Reading thread %lu is done\n", GetCurrentThreadId());

    return hr;
}

//---------------------------------------------------------------
DWORD	WINAPI	CEncoderTest::encoding_thread_proc(void *p)
//---------------------------------------------------------------
{
	return reinterpret_cast<CEncoderTest*>(p)->EncodingThreadProc();
}

//---------------------------------------------------------------
DWORD	CEncoderTest::EncodingThreadProc()
//---------------------------------------------------------------
{
	HRESULT hr = S_OK;

	fprintf(stderr, "Encoding thread %lu is started\n", GetCurrentThreadId());

	for(int frame_no = 0; ; frame_no++)
	{
		const int buffer_id = frame_no % m_EncPar.QueueSize;

		const DWORD t0 = GetTickCount();

		HANDLE hh[2] = { m_evCancel, m_Queue[buffer_id].evFilled };

		const DWORD wait_result = WaitForMultipleObjects(2, hh, FALSE, INFINITE);
		if (wait_result == WAIT_OBJECT_0)
		{
			m_hrResult = S_FALSE;// E_ABORT;
			break;
		}

		if (m_Queue[buffer_id].hrReadStatus != S_OK)
		{
			m_hrResult = m_Queue[buffer_id].hrReadStatus;
			break;
		}

		CC_VIDEO_FRAME_DESCR frame_descr = { m_EncPar.InputColorFormat };

		if (CComQIPtr<ICC_VideoConsumerExtAsync> pEncAsync = m_pEncoder)
		{
			hr = pEncAsync->AddScaleFrameAsync(
				m_Queue[buffer_id].pBuffer,
				m_FrameSizeInBytes,
				&frame_descr,
				CComPtr<IUnknown>(),
				nullptr);
		}
		else
		{
			hr = m_pEncoder->AddScaleFrame(
				m_Queue[buffer_id].pBuffer,
				m_FrameSizeInBytes,
				&frame_descr,
				NULL);
		}

		if(FAILED(hr))
		  break;

		DWORD t1 = GetTickCount();
	
		int wait_time = 42 - (t1 - t0);
//		if (wait_time > 1)
//			Sleep(wait_time);

		InterlockedIncrement(&m_Stats.NumFramesWritten);
		InterlockedAdd64(&m_Stats.NumBytesWritten, 0);
		SetEvent(m_Queue[buffer_id].evVacant);
	}

	InterlockedDecrement(&m_NumActiveThreads);
	fprintf(stderr, "Encoding thread %lu is done\n", GetCurrentThreadId());

	return hr;
}
