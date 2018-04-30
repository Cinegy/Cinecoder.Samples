#pragma once

#include <memory>
#include <vector>
#include <fstream>

///////////////////////////////////////////////////////////////////////////////

#if defined(__APPLE__)
#ifndef COM_NO_WINDOWS_H
#define COM_NO_WINDOWS_H
#endif
#endif

// Cinecoder
#include <Cinecoder_h.h>

// Cinegy utils
#include "utils/comptr.h"

///////////////////////////////////////////////////////////////////////////////


class CodedFrame
{
public:
	CodedFrame() { coded_frame_size = 25 * 1024 * 1024; coded_frame.resize(coded_frame_size); }
	~CodedFrame() {}

	std::vector<unsigned char> coded_frame;
	size_t coded_frame_size;
	size_t frame_number;

private:
	//CodedFrame(const CodedFrame&);
	CodedFrame& operator=(const CodedFrame&);

public:
	int AddRef() { return 2; }
	int Release() { return 1; }
};

class ReadFileDN2 : public C_SimpleThread<ReadFileDN2>
{
private:
	std::ifstream m_file;
	size_t m_frames;

	com_ptr<ICC_MvxFile> m_fileMvx;

	bool m_bProcess;

	std::list<CodedFrame> m_listFrames;

	C_CritSec m_critical_read;
	C_CritSec m_critical_queue;

	C_Event m_hExitEvent;
	C_QueueT<CodedFrame> m_queueFrames;
	C_QueueT<CodedFrame> m_queueFrames_free;
	
	bool m_bSeek;
	size_t m_iSeekFrame;
	int m_iSpeed;

public:
	ReadFileDN2();
	~ReadFileDN2();

public:
	int OpenFile(const char* filename);
	int CloseFile();

	int StartPipe();
	int StopPipe();

	int ReadFrame(size_t frame, std::vector<unsigned char> & buffer, size_t & size);
	size_t GetCountFrames()
	{ 
		CC_UINT frames = 0; 
		m_fileMvx->get_Length(&frames); 
		return (size_t)frames; 
	}
	void SeekFrame(size_t nFrame) 
	{ 
		if (nFrame < m_frames) 
		{ 
			m_bSeek = true; 
			m_iSeekFrame = nFrame; 
		} 
	}
	void SetSpeed(int iSpeed)
	{
		if (abs(iSpeed) >= 1 && abs(iSpeed) <= 4)
			m_iSpeed = iSpeed;
	}
	int GetSpeed()
	{
		return m_iSpeed;
	}

public:
	CodedFrame* MapFrame();
	void UnmapFrame(CodedFrame* pFrame);

public:
	long ThreadProc();
};

