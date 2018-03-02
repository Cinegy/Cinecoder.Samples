#include "stdafx.h"

#include "CEncoderTest.h"

//---------------------------------------------------------------
CEncoderTest::CEncoderTest()
//---------------------------------------------------------------
{
	m_bRunning = FALSE;
}

//---------------------------------------------------------------
CEncoderTest::~CEncoderTest()
//---------------------------------------------------------------
{
	Close();
}

//---------------------------------------------------------------
int		CEncoderTest::AssignParameters(const ENCODER_PARAMS &par)
//---------------------------------------------------------------
{
	if (!par.pEncoder)
		return E_INVALIDARG;

	m_pEncoder = par.pEncoder;
	m_evCancel = CreateEvent(NULL, TRUE, FALSE, NULL);

	for(int i = 0; i < par.QueueSize; i++)
	{ 
		m_BufferVacantEvents.push_back(CreateEvent(NULL, FALSE, TRUE , NULL));
		m_BufferFilledEvents.push_back(CreateEvent(NULL, FALSE, FALSE, NULL));
	}

	m_EncPar = par;
	memset((void*)&m_Stats, 0, sizeof(m_Stats));

	return S_OK;
}

//---------------------------------------------------------------
int		CEncoderTest::Close()
//---------------------------------------------------------------
{
	if (!m_pEncoder)
		return S_FALSE;

	if(m_bRunning)
		Cancel();

	for (size_t i = 0; i < m_BufferVacantEvents.size(); i++)
		CloseHandle(m_BufferVacantEvents[i]);
	m_BufferVacantEvents.clear();

	for (size_t i = 0; i < m_BufferFilledEvents.size(); i++)
		CloseHandle(m_BufferFilledEvents[i]);
	m_BufferFilledEvents.clear();

	CloseHandle(m_evCancel);

	return S_OK;
}

//---------------------------------------------------------------
int		CEncoderTest::Run()
//---------------------------------------------------------------
{
	m_hEncodingThread = CreateThread(NULL, 0, encoding_thread_proc, this, 0, NULL);

	for (int i = 0; i < m_EncPar.NumReadThreads; i++)
		m_hReadingThreads.push_back(CreateThread(NULL, 0, reading_thread_proc, this, 0, NULL));

	m_bRunning = TRUE;

	return S_OK;
}

//---------------------------------------------------------------
int		CEncoderTest::Cancel()
//---------------------------------------------------------------
{
	if (!m_bRunning)
		return S_FALSE;

	SetEvent(m_evCancel);
	WaitForMultipleObjects((DWORD)m_hReadingThreads.size(), &m_hReadingThreads[0], TRUE, INFINITE);
	WaitForSingleObject(m_hEncodingThread, INFINITE);

	for(size_t i = 0; i < m_hReadingThreads.size(); i++)
		CloseHandle(m_hReadingThreads[i]);
	m_hReadingThreads.clear();

	CloseHandle(m_hEncodingThread);

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
  fprintf(stderr, "Reading thread %d is started\n", GetCurrentThreadId());

  HANDLE hFile = CreateFile(m_EncPar.InputFileName, GENERIC_READ, FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, NULL);

  if(hFile == INVALID_HANDLE_VALUE)
	;//return fprintf(stderr, "Thread %d: error %08xh opening the file\n", GetCurrentThreadId(), HRESULT_FROM_WIN32(GetLastError())), HRESULT_FROM_WIN32(GetLastError());

  else for(;;)
  {
  	int frame_no = InterlockedExchangeAdd(&m_Stats.NumFramesRead, 1);
  	int buffer_id = frame_no % m_EncPar.QueueSize;

  	HANDLE hh[2] = { m_evCancel, m_BufferVacantEvents[buffer_id] };

    DWORD wait_result = WaitForMultipleObjects(2, hh, FALSE, INFINITE);
    if(wait_result == WAIT_OBJECT_0)
      break;

  	LONGLONG offset = frame_no * LONGLONG(m_FrameSizeInBytes);
	SetFilePointer(hFile, (LONG)offset, ((LONG*)offset)+1, FILE_BEGIN);

	DWORD r;
	if(!ReadFile(hFile, m_Buffers[buffer_id], m_FrameSizeInBytes, &r, NULL))
      break;

    if(r != m_FrameSizeInBytes)
      break;

	InterlockedAdd64(&m_Stats.NumBytesRead, m_FrameSizeInBytes);

    SetEvent(m_BufferFilledEvents[buffer_id]);
  }

  HRESULT hr = HRESULT_FROM_WIN32(GetLastError());
  
  if(FAILED(hr))
  {
    fprintf(stderr, "Reading thread %d: error %08x\n", GetCurrentThreadId(), hr);
    SetEvent(m_evCancel);
  }

  CloseHandle(hFile);

  fprintf(stderr, "Reading thread %d is done\n", GetCurrentThreadId());

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

  fprintf(stderr, "Encoding thread %d is started\n", GetCurrentThreadId());

  for(int frame_no = 0; ; frame_no++)
  {
  	int buffer_id = frame_no % m_EncPar.QueueSize;

  	HANDLE hh[2] = { m_evCancel, m_BufferFilledEvents[buffer_id] };

    DWORD wait_result = WaitForMultipleObjects(2, hh, FALSE, INFINITE);
    if(wait_result == WAIT_OBJECT_0)
      break;

    CC_VIDEO_FRAME_DESCR frame_descr = { m_EncPar.ColorFormat };

	if (CComQIPtr<ICC_VideoConsumerExtAsync> pEncAsync = m_pEncoder)
	{
		hr = pEncAsync->AddScaleFrameAsync(
			m_Buffers[buffer_id],
			m_FrameSizeInBytes,
			&frame_descr,
			CComPtr<IUnknown>(),
			NULL);
	}
	else
	{
		hr = m_pEncoder->AddScaleFrame(
			m_Buffers[buffer_id],
			m_FrameSizeInBytes,
			&frame_descr,
			NULL);
	}

	if(FAILED(hr))
	  break;
  }

  fprintf(stderr, "Encoding thread %d is done\n", GetCurrentThreadId());

  return hr;
}
