#pragma once

#include "Block.h"
#include "ReadFileDN2.h"

enum IMAGE_FORMAT { IMAGE_FORMAT_RGBA8BIT, IMAGE_FORMAT_BGRA8BIT, IMAGE_FORMAT_RGBA16BIT, IMAGE_FORMAT_BGRA16BIT, IMAGE_FORMAT_RGB30 };

class DecodeDaniel2 : public C_SimpleThread<DecodeDaniel2>, public ICC_DataReadyCallback
{
private:
	std::wstring m_filename;

	size_t m_width;
	size_t m_height;
	size_t m_stride;
	IMAGE_FORMAT m_outputImageFormat;
	CC_COLOR_FMT m_fmt;
	const char* m_strStreamType;

	CC_FRAME_RATE m_FrameRate;

	bool m_bProcess;
	bool m_bPause;
	bool m_bDecode;
	bool m_bInitDecoder;
	bool m_bUseCuda;
	bool m_bUseCudaHost;

	ReadFileDN2 m_file;
	std::vector<unsigned char> m_buffer;

	C_QueueT<C_Block> m_queueFrames;
	C_QueueT<C_Block> m_queueFrames_free;

	C_Event	m_eventInitDecoder;

	std::list<C_Block> m_listBlocks;

	com_ptr<ICC_ClassFactory> m_piFactory;
	com_ptr<ICC_VideoDecoder> m_pVideoDec;

	com_ptr<ICC_MediaReader> m_pMediaReader;
	com_ptr<ICC_AudioStreamInfo> m_pAudioStreamInfo;

	ULONGLONG m_llDuration;
	ULONGLONG m_llTimeBase;
	bool bIntraFormat;

public:
	DecodeDaniel2();
	~DecodeDaniel2();

public:
	int OpenFile(const char* const filename, size_t iMaxCountDecoders = 2,bool useCuda = false);
	int StartDecode();
	int StopDecode();

	size_t GetImageWidth() { return m_width; }
	size_t GetImageHeight() { return m_height; }
	IMAGE_FORMAT GetImageFormat() { return m_outputImageFormat; }

	C_Block* MapFrame();
	void  UnmapFrame(C_Block* pBlock);

	bool isProcess() { return m_bProcess; }
	bool isPause() { return m_bPause; }
	bool isDecode() { return m_bDecode; }

	void SetPause(bool bPause) { m_bPause = bPause; }
	void SetDecode(bool bDecode) { m_bDecode = bDecode; }
	
	void SeekFrame(size_t nFrame) { m_file.SeekFrame(nFrame); }

	void SetSpeed(int iSpeed) { if (bIntraFormat) m_file.SetSpeed(iSpeed); }
	int GetSpeed() { return m_file.GetSpeed(); }

	void SetReadFile(bool bReadFile) { if (bIntraFormat) m_file.SetReadFile(bReadFile); }
	bool GetReadFile() { return m_file.GetReadFile(); }

	size_t GetCountFrames() { return m_file.GetCountFrames(); }

	double GetFrameRate() { return ((double)m_FrameRate.num / (double)m_FrameRate.denom); }
	CC_FRAME_RATE GetFrameRateValue() { return m_FrameRate; }

private:
	int CreateDecoder(size_t iMaxCountDecoders, bool useCuda = false);
	int DestroyDecoder();

	int InitValues();
	int DestroyValues();

private:
	virtual ULONG STDMETHODCALLTYPE AddRef(void) { return 2; }
	virtual ULONG STDMETHODCALLTYPE Release(void) { return 1; }
	virtual HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void **ppv)
	{
		if (riid == IID_IUnknown || riid == IID_ICC_DataReadyCallback) { *ppv = (ICC_DataReadyCallback *)this; return S_OK; }
		return E_NOINTERFACE;
	}
	virtual HRESULT STDMETHODCALLTYPE DataReady(IUnknown *pDataProducer);

private:
	friend class C_SimpleThread<DecodeDaniel2>;
	long ThreadProc();
};
