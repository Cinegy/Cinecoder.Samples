#pragma once

#include "Block.h"
#include "ReadFileDN2.h"

///////////////////////////////////////////////////////////////////////////////

#if defined(__APPLE__)
#ifndef COM_NO_WINDOWS_H
#define COM_NO_WINDOWS_H
#endif
#endif

// Cinecoder
#include <Cinecoder_h.h>

// Cinegy utils
#include <comptr.h>

// License
#define COMPANYNAME "cinegy"
#define LICENSEKEY "X9S1C220USC51ZST6HG05EN6U4ZC1MLJ6LSK5S9SGBCMC773M6MNF2EHUG9GFENC"

///////////////////////////////////////////////////////////////////////////////

enum IMAGE_FORMAT { IMAGE_FORMAT_RGBA8BIT, IMAGE_FORMAT_RGBA16BIT, IMAGE_FORMAT_RGB30 };

class DecodeDaniel2 : public C_SimpleThread<DecodeDaniel2>, public ICC_DataReadyCallback
{
private:
	std::wstring m_filename;

	size_t m_width;
	size_t m_height;
	size_t m_stride;
	IMAGE_FORMAT m_outputImageFormat;
	CC_COLOR_FMT m_fmt;

	bool m_bProcess;
	bool m_bPause;
	bool m_bInitDecoder;

	ReadFileDN2 m_file;
	std::vector<unsigned char> m_buffer;

	C_Event m_hExitEvent;
	C_QueueT<C_Block> m_queueFrames;
	C_QueueT<C_Block> m_queueFrames_free;

	C_Event	m_eventInitDecoder;

	std::list<C_Block> m_listBlocks;

	com_ptr<ICC_DanielVideoDecoder> m_pVideoDec;

public:
	DecodeDaniel2();
	~DecodeDaniel2();

public:
	int OpenFile(const char* const filename, size_t iMaxCountDecoders = 2);
	int StartDecode();
	int StopDecode();

	size_t GetImageWidth() { return m_width; }
	size_t GetImageHeight() { return m_height; }
	IMAGE_FORMAT GetImageFormat() { return m_outputImageFormat; }

	C_Block* MapFrame();
	void  UnmapFrame(C_Block* pBlock);

	bool isProcess() { return m_bProcess; }
	bool isPause() { return m_bPause; }

	void SetPause(bool bPause) { m_bPause = bPause; }

	ReadFileDN2* GetReaderPtr() { return &m_file; }

private:
	int CreateDecoder(size_t iMaxCountDecoders);
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

protected:
	friend class C_SimpleThread<DecodeDaniel2>;
	long ThreadProc();
};
