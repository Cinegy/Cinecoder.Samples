//------------------------------------------------------------------
class C_PPMWriter : public C_Unknown, public ICC_DataReadyCallback
//------------------------------------------------------------------
{
	BYTE *m_pBuffer;
	DWORD m_cbFrameBytes;
	const char *m_strFileMask;

public:
	C_PPMWriter(const char *filemask) : m_strFileMask(filemask), m_pBuffer(NULL), m_cbFrameBytes(0)
	{
	}

	virtual ~C_PPMWriter()
	{
		if(m_pBuffer)
			delete [] m_pBuffer;
	}

	_IMPLEMENT_IUNKNOWN_1(ICC_DataReadyCallback);

	STDMETHOD(DataReady)(IUnknown *pUnk)
	{
		HRESULT hr = S_OK;

		com_ptr<ICC_VideoProducer> spProducer;
		if(FAILED(hr = pUnk->QueryInterface(IID_ICC_VideoProducer, (void**)&spProducer)))
			return hr;

		com_ptr<ICC_VideoStreamInfo> spVideoInfo;
		if(FAILED(hr = spProducer->GetVideoStreamInfo(&spVideoInfo)))
			return hr;

		SIZE szFrame;
		if(spVideoInfo->get_FrameSize(&szFrame) != S_OK)
			return E_UNEXPECTED;

		CC_COLOR_FMT formatImg = CCF_BGR24;
		int stride = szFrame.cx * 3;

		if(m_pBuffer == NULL) // first call!
		{
			printf("Frame size = %d x %d, Frame rate = ", szFrame.cx, szFrame.cy);

			CC_FRAME_RATE rFrameRate;
			if(S_OK == spVideoInfo->get_FrameRate(&rFrameRate))
				printf("%g\n", double(rFrameRate.num) / rFrameRate.denom);
			else
				printf("<unknown>\n");

			hr = spProducer->GetFrame(formatImg, NULL, 0, stride, &m_cbFrameBytes);

			if (FAILED(hr = spProducer->GetFrame(formatImg, NULL, 0, stride, &m_cbFrameBytes)))
				return hr;

			if(NULL == (m_pBuffer = new BYTE[m_cbFrameBytes]))
				return E_OUTOFMEMORY;
		}

		if (!m_pBuffer)
			return E_FAIL;

		com_ptr<ICC_VideoFrameInfo> spFrame;
		if(FAILED(hr = spProducer->GetVideoFrameInfo(&spFrame)))
  			return hr;
		
		DWORD dwFrameNumber = 0;
		if(FAILED(hr = spFrame->get_Number(&dwFrameNumber)))
			return hr;

		char filename[128];
		sprintf(filename, m_strFileMask, dwFrameNumber);
		fprintf(stderr, "%s:", filename);

		DWORD dwBytesWritten = 0;
		if(FAILED(hr = spProducer->GetFrame(formatImg, m_pBuffer, m_cbFrameBytes, stride, &dwBytesWritten)))
			return hr;

		FILE *f = fopen(filename, "wb");
		if(f == NULL)
			return E_FAIL;

		/* PPM header */
		char hdr[128];
		sprintf(hdr,"P6\n%d %d\n255\n", szFrame.cx, szFrame.cy);

		if(fwrite(hdr, 1, strlen(hdr), f) != strlen(hdr))
			return E_FAIL;

		if(fwrite(m_pBuffer, 1, dwBytesWritten, f) != dwBytesWritten)
			return E_FAIL;

		fclose(f);

		fprintf(stderr, "Ok\n");

		return S_OK;
	}
};

