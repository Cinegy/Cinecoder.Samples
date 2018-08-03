//------------------------------------------------------------------
class C_PPMWriter : public C_Unknown, public ICC_DataReadyCallback
//------------------------------------------------------------------
{
	BYTE *m_pBuffer;
	DWORD m_cbFrameBytes;
	const char *m_strFileMask;

	CC_COLOR_FMT formatImg;
	INT stride;

public:
	C_PPMWriter(const char *filemask) : m_strFileMask(filemask), m_pBuffer(NULL), m_cbFrameBytes(0) , formatImg(CCF_BGR24), stride(0)
	{
	}

	virtual ~C_PPMWriter()
	{
		if(m_pBuffer)
			delete [] m_pBuffer;
	}

	_IMPLEMENT_IUNKNOWN_1(ICC_DataReadyCallback);                                 \

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

		if(m_pBuffer == NULL) // first call!
		{
			printf("Frame size = %d x %d, Frame rate = ", szFrame.cx, szFrame.cy);

			CC_FRAME_RATE rFrameRate;
			if(S_OK == spVideoInfo->get_FrameRate(&rFrameRate))
				printf("%g\n", double(rFrameRate.num) / rFrameRate.denom);
			else
				printf("<unknown>\n");

			stride = szFrame.cx * 3;
			hr = spProducer->GetFrame(formatImg, NULL, 0, stride, &m_cbFrameBytes);

			if (hr == MPG_E_FORMAT_NOT_SUPPORTED)
			{
				formatImg = CCF_BGR32;
				stride = szFrame.cx * 4;
			} 
			else if (FAILED(hr)) return hr;

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

		DWORD dwBytesWrote = 0;
		if(FAILED(hr = spProducer->GetFrame(formatImg, m_pBuffer, m_cbFrameBytes, stride, &dwBytesWrote)))
			return hr;

		FILE *f = fopen(filename, "wb");
		if(f == NULL)
			return E_FAIL;

		/* PPM header */
		char hdr[128];
		sprintf(hdr,"P6\n%d %d\n255\n", szFrame.cx, szFrame.cy);

		if(fwrite(hdr, 1, strlen(hdr), f) != strlen(hdr))
			return E_FAIL;

		if (formatImg == CCF_BGR24 || formatImg == CCF_RGB24)
		{
			if(fwrite(m_pBuffer, 1, dwBytesWrote, f) != dwBytesWrote)
				return E_FAIL;
		}
		else if (formatImg == CCF_BGR32 || formatImg == CCF_RGB32)
		{
			size_t pixels = szFrame.cx * szFrame.cy;
			BYTE* pData = m_pBuffer;
			for (size_t i = 0; i < pixels; ++i)
			{
				if (fwrite(pData, 1, 3, f) != 3) // save only 3 components: R G B
					break;
				pData += 4;
			}
		}

		fclose(f);

		fprintf(stderr, "Ok\n");

		return S_OK;
	}
};

