//------------------------------------------------------------------
class C_BMP32Writer : public C_Unknown, public ICC_DataReadyCallback
//------------------------------------------------------------------
{
	BYTE *m_pBuffer{};
	DWORD m_cbFrameBytes{};
	const char *m_strFileMask;

	CC_COLOR_FMT m_imgFormat{CCF_RGB32};
	SIZE m_szFrameSize{};
	int m_iFramePitch{};

public:
	C_BMP32Writer(const char *filemask) : m_strFileMask(filemask)
	{
	}

	virtual ~C_BMP32Writer()
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

		if(m_pBuffer == NULL) // first call!
		{
			com_ptr<ICC_VideoStreamInfo> spVideoInfo;
			if(FAILED(hr = spProducer->GetVideoStreamInfo(&spVideoInfo)))
				return hr;

			if(spVideoInfo->get_FrameSize(&m_szFrameSize) != S_OK)
				return E_UNEXPECTED;

			m_iFramePitch = m_szFrameSize.cx * 4;

			printf("Frame size = %d x %d, Frame rate = ", m_szFrameSize.cx, m_szFrameSize.cy);

			CC_FRAME_RATE rFrameRate;
			if(S_OK == spVideoInfo->get_FrameRate(&rFrameRate))
				printf("%g\n", double(rFrameRate.num) / rFrameRate.denom);
			else
				printf("<unknown>\n");

			if (FAILED(hr = spProducer->GetFrame(m_imgFormat, NULL, 0, m_iFramePitch, &m_cbFrameBytes)))
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
		if(FAILED(hr = spProducer->GetFrame(m_imgFormat, m_pBuffer, m_cbFrameBytes, m_iFramePitch, &dwBytesWritten)))
			return hr;

		if(FAILED(hr = SaveBitmap(filename, m_szFrameSize.cx, m_szFrameSize.cy, m_iFramePitch, 32, m_pBuffer)))
			return hr;

		fprintf(stderr, "Ok\n");

		return S_OK;
	}

    //----------------------------------------------------------------
    HRESULT SaveBitmap(const char *filename, int w, int h, int src_pitch, int bpp, const BYTE *pbData)
    //----------------------------------------------------------------
    {
    	FILE *F = NULL;

		if( (F = fopen(filename, "wb")) == NULL )
			return fprintf(stderr, " error creating the file"), E_FAIL;

#pragma pack(push,1)
		struct //BITMAPFILEHEADER + BITMAPINFOHEADER
		{
			WORD  bfType;
			DWORD bfSize;
			WORD  bfReserved1;
			WORD  bfReserved2;
			DWORD bfOffBits;
		
			DWORD biSize;
			LONG  biWidth;
			LONG  biHeight;
			WORD  biPlanes;
			WORD  biBitCount;
			DWORD biCompression;
			DWORD biSizeImage;
			LONG  biXPelsPerMeter;
			LONG  biYPelsPerMeter;
			DWORD biClrUsed;
			DWORD biClrImportant;
		}
#pragma pack(pop)
		hdr = {};

		int   image_pitch = ((w * (bpp/8) + 3) & ~3);
		DWORD image_size  = image_pitch * h;

		hdr.bfType        = 0x4D42;
		hdr.bfOffBits     = sizeof(hdr);
		hdr.bfSize        = hdr.bfOffBits + image_size;
		                 
		hdr.biSize        = 40;
		hdr.biWidth       = w;              
		hdr.biHeight      = h;                
		hdr.biPlanes      = 1;                            
		hdr.biBitCount    = bpp;                         
		hdr.biSizeImage   = image_size;

		if(fwrite(&hdr, 1, sizeof(hdr), F) != sizeof(hdr))
			return fprintf(stderr, " error writing the file. Disk may be full."), E_FAIL;

		for(int i = h-1; i >= 0; i--) // Writing BMP top-down
			if(fwrite(pbData + i*src_pitch, 1, image_pitch, F) != image_pitch)
				return fprintf(stderr, " error writing the file. Disk may be full."), E_FAIL;
		  
		fclose(F);

    	return S_OK;
    }

};

