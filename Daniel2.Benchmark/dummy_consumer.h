//------------------------------------------------------------------
class C_DummyWriter : public ICC_DataReadyCallback
//------------------------------------------------------------------
{
  _IMPLEMENT_IUNKNOWN_STATICALLY(ICC_DataReadyCallback)

  BYTE         *m_pBuffer;
  CC_FRAME_RATE m_rFrameRate;
  CC_SIZE       m_szFrame;
  DWORD         m_cbFrameBytes;
  CC_COLOR_FMT  m_Format;
  int		m_FrameNo;

  com_ptr<ICC_VideoQualityMeter> m_pPsnrCalc;
  BYTE         *m_pRefBuffer;

public:
  C_DummyWriter(CC_COLOR_FMT fmt, BYTE *buffer, int bufsize, ICC_VideoQualityMeter *pPsnrCalc, BYTE *pRefBuffer) 
  : m_Format(fmt)
  , m_pBuffer(buffer)
  , m_cbFrameBytes(bufsize)
  , m_FrameNo(0)
  , m_pPsnrCalc(pPsnrCalc)
  , m_pRefBuffer(pRefBuffer)
  {}

  virtual ~C_DummyWriter()
  {
  }

  HRESULT DetectStreamCaps(ICC_VideoProducer *pProducer)
  {
    HRESULT hr = S_OK;
    
    com_ptr<ICC_VideoStreamInfo> spVideoInfo;
    if(FAILED(hr = pProducer->GetVideoStreamInfo(&spVideoInfo)))
      return hr;
    
    spVideoInfo->get_FrameSize(&m_szFrame);
    spVideoInfo->get_FrameRate(&m_rFrameRate);

    return S_OK;
  }

  STDMETHOD(DataReady)(IUnknown *pUnk)
  {
    HRESULT hr = S_OK;

    com_ptr<ICC_VideoProducer> spProducer;
    if(FAILED(hr = pUnk->QueryInterface(IID_ICC_VideoProducer, (void**)&spProducer)))
      return hr;

    if(m_FrameNo == 0)
    {
      g_DecoderTimeFirstFrameOut = system_clock::now();
        
      if(FAILED(hr = DetectStreamCaps(spProducer)))
      {
        fprintf(stderr, "detecting stream capabilities failed (code %08x)\n", hr); 
        return hr;
      }

      fprintf(stderr, "Frame size = %d x %d, Frame rate = %g\n", m_szFrame.cx, m_szFrame.cy, double(m_rFrameRate.num) / m_rFrameRate.denom);
    }

    if(m_Format != CCF_UNKNOWN)
    {
      DWORD dwBytesWrote = 0;
      if(FAILED(hr = spProducer->GetFrame(m_Format, m_pBuffer, m_cbFrameBytes, 0, &dwBytesWrote)))
        return hr;
#if 0
      if(m_FrameNo == 0)
      {
        static int cnt = 0;
        char namebuf[128];
        sprintf(namebuf, "decoded_frame_%05d.bin", cnt++);
        FILE *F = fopen(namebuf, "wb");
        fwrite(m_pBuffer, 1, m_cbFrameBytes, F);
        fclose(F);
      }
#endif

      if(m_FrameNo == 0 && m_pPsnrCalc)
      {
        BYTE *pBuffer    = m_pBuffer;
        BYTE *pRefBuffer = m_pRefBuffer;

		if(g_mem_type == MEM_GPU)
		{
		  pBuffer    = (BYTE*)mem_alloc(MEM_SYSTEM, m_cbFrameBytes);
		  pRefBuffer = (BYTE*)mem_alloc(MEM_SYSTEM, m_cbFrameBytes);

		  if(!pBuffer || !pRefBuffer)
		  {
		    hr = E_OUTOFMEMORY;
		  }
		  else
		  {
		    auto err = cudaMemcpy(pBuffer   , m_pBuffer   , m_cbFrameBytes, cudaMemcpyDeviceToHost);
		    if(!err) 
		    	err = cudaMemcpy(pRefBuffer, m_pRefBuffer, m_cbFrameBytes, cudaMemcpyDeviceToHost);

		    if(err)
		    {
      			fprintf(stderr, "cudaMemcpy(cudaMemcpyDeviceToHost) error %d\n", err);
      			hr = E_UNEXPECTED;
			}
		  }
		}

        CC_VIDEO_QUALITY_MEASUREMENT psnr = {};

        if(SUCCEEDED(hr))
        {
          hr = m_pPsnrCalc->Measure(CC_VQM_PSNR, m_szFrame, m_Format, 
								    pBuffer   , m_cbFrameBytes, 0,
								    pRefBuffer, m_cbFrameBytes, 0,
								    &psnr);
		}

		if(g_mem_type == MEM_GPU)
		{
		  mem_release(MEM_SYSTEM, pBuffer);
		  mem_release(MEM_SYSTEM, pRefBuffer);
		}

		if(FAILED(hr))
		  fprintf(stderr, "PSNR calculation failed, error code %xh\n", hr);

		else for(int i = 0; i < psnr.NumVals; i++)
		  printf("PSNR(%c) = %.3f dB%s", psnr.ValNames[i], psnr.QVal[i], i+1 < psnr.NumVals ? ", " : "\n");
      }
    }

    m_FrameNo++; 

    return S_OK;
  }
};

