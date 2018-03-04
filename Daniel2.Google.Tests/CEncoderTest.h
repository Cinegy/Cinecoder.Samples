#pragma once

#include <Windows.h>
#include <vector>

#include "cinecoder_h.h"

struct ENCODER_PARAMS
{
	ICC_VideoEncoder* pEncoder;
	LPCTSTR		InputFileName;
	CC_COLOR_FMT ColorFormat;
	int			NumReadThreads;
	int			QueueSize;
	LPCTSTR		OutputFileName;
};

struct	ENCODER_STATS
{
	LONG		NumFramesRead;
	LONG		NumFramesWritten;
	LONGLONG	NumBytesRead;
	LONGLONG	NumBytesWritten;
};

//---------------------------------------------------------------
class CEncoderTest
//---------------------------------------------------------------
{
public:
	CEncoderTest();
	~CEncoderTest();

	int		AssignParameters(const ENCODER_PARAMS&);
	int		Close();

	int		Run();
	int		Cancel();

	int		GetCurrentEncodingStats(ENCODER_STATS*);

private:
	BOOL	m_bRunning;

	ENCODER_PARAMS	m_EncPar;
	DWORD	m_FrameSizeInBytes;

	CComPtr<ICC_VideoEncoder> m_pEncoder;

	std::vector<HANDLE>	m_hReadingThreads;
	DWORD	ReadingThreadProc();
	static	DWORD	WINAPI	reading_thread_proc(void *p);

	HANDLE	m_hEncodingThread;
	DWORD	EncodingThreadProc();
	static	DWORD	WINAPI	encoding_thread_proc(void *p);

	HANDLE	m_evCancel;
	std::vector<HANDLE>	m_BufferVacantEvents;
	std::vector<HANDLE>	m_BufferFilledEvents;
	std::vector<LPBYTE> m_Buffers;

	volatile ENCODER_STATS	m_Stats;
};
