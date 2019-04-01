#pragma once

///////////////////////////////////////////////////////////////////////////////

//#define __STD_READ__ 1
//#define __FILE_READ__ 1
#define __UNBUFF_READ__ 1

// Cinecoder
#include <Cinecoder_h.h>

// License
#include "../common/cinecoder_license_string.h"

// Read file
#include "../common/read_file.h"

///////////////////////////////////////////////////////////////////////////////

#if defined (__STD_READ__) || defined (__FILE_READ__)
class C_Buffer
{
private:
	std::vector<unsigned char> buffer;

public:
	size_t GetSize() { return buffer.size(); }
	unsigned char* GetPtr() { return buffer.data(); }
	void Resize(size_t size) { if (size > buffer.size()) buffer.resize(size); }

	void SetDiff(size_t diff) { }
	size_t GetDiff() { return 0; }
};
#elif __UNBUFF_READ__
class C_Buffer
{
private:
	LPBYTE m_pBuffer {nullptr};
	size_t m_page_aligned_size {};
	size_t m_data_size {};
	size_t m_diff {};

	void Destroy()
	{
		if (m_pBuffer)
#if defined(__WIN32__)
			VirtualFree(m_pBuffer, 0, MEM_RELEASE);
#else
			free(m_pBuffer);
#endif	
		m_pBuffer = nullptr;
		m_page_aligned_size = 0;
		m_data_size = 0;
		m_diff = 0;
	}
public:
	~C_Buffer() { Destroy(); }
	
	void SetDiff(size_t diff) { m_diff = diff; }
	size_t GetDiff() { return m_diff; }

	size_t GetSize() { return m_data_size; }
	LPBYTE GetPtr() { return m_pBuffer + m_diff; }
	LPBYTE GetPtr(size_t diff) { return m_pBuffer + diff; }
	void Resize(size_t size)
	{
		if (size > m_page_aligned_size)
		{
			Destroy();

			m_data_size = size;
			m_page_aligned_size = (size + 4095) & ~4095;

#if defined(__WIN32__)
			m_pBuffer = (LPBYTE)VirtualAlloc(NULL, m_page_aligned_size, MEM_COMMIT, PAGE_READWRITE);
#elif defined(__APPLE__)
			m_pBuffer = (LPBYTE)malloc(m_page_aligned_size);
#else
			m_pBuffer = (LPBYTE)aligned_alloc(4096, m_page_aligned_size);
#endif	
		}
	}
};
#endif

class CodedFrame
{
public:
	CodedFrame() {}
	~CodedFrame() {}

	C_Buffer coded_frame;
	size_t coded_frame_size;
	size_t frame_number;
	size_t flags;

	CodedFrame(const CodedFrame&) = default;
	CodedFrame(CodedFrame&&) = default;

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
#ifdef __STD_READ__
	std::ifstream m_file;
#elif __FILE_READ__
	FILE *m_file;
#elif __UNBUFF_READ__
	C_ReadFile m_file;
#endif
	size_t m_frames;

	com_ptr<ICC_MvxFile> m_fileMvx;

	bool m_bProcess;
	bool m_bReadFile;

	std::list<CodedFrame> m_listFrames;

	C_CritSec m_critical_read;
	C_CritSec m_critical_queue;

	C_QueueT<CodedFrame> m_queueFrames;
	C_QueueT<CodedFrame> m_queueFrames_free;
	
	bool m_bSeek;
	size_t m_iSeekFrame;
	int m_iSpeed;

	std::atomic<size_t> data_rate;

public:
	ReadFileDN2();
	~ReadFileDN2();

public:
	int OpenFile(const char* filename);
	int CloseFile();

	int StartPipe();
	int StopPipe();

	int ReadFrame(size_t frame, C_Buffer & buffer, size_t & size);

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

			CC_MVX_ENTRY Idx;
			if (SUCCEEDED(m_fileMvx->FindKeyEntry((CC_UINT)nFrame, &Idx)))
			{
				m_iSeekFrame = Idx.CodingOrderNum;
			}
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
	CC_ELEMENTARY_STREAM_TYPE GetStreamType()
	{
		CC_ELEMENTARY_STREAM_TYPE type;
		m_fileMvx->get_StreamType(&type);
		return type;
	}
	void SetReadFile(bool bReadFile) { m_bReadFile = bReadFile; }
	bool GetReadFile() { return m_bReadFile; }

	size_t GetDataRate(bool bClearData) { size_t ret = data_rate; if (bClearData) data_rate = 0; return ret; }

public:
	CodedFrame* MapFrame();
	void UnmapFrame(CodedFrame* pFrame);

private:
	friend class C_SimpleThread<ReadFileDN2>;
	long ThreadProc();
};

