//------------------------------------------------------------------
class C_DummyWriter : public ICC_DataReadyCallback
//------------------------------------------------------------------
{
  _IMPLEMENT_IUNKNOWN_STATICALLY(ICC_DataReadyCallback)

  //memobj_t&     m_memBuffer;
  std::vector<memobj_t> &m_memBuffers;
  CC_FRAME_RATE m_rFrameRate;
  CC_SIZE       m_szFrame;
  DWORD         m_cbFrameBytes;
  CC_COLOR_FMT  m_Format;
  int           m_pitch;
  CC_MEMORY_TYPE m_memtype;

  int		    m_FrameNo;

  com_ptr<ICC_VideoQualityMeter> m_pPsnrCalc;
  memobj_t&     m_memRefBuffer;

public:
  C_DummyWriter(CC_COLOR_FMT fmt, std::vector<memobj_t> &buffers, int bufsize, int pitch, CC_MEMORY_TYPE mem_type, ICC_VideoQualityMeter *pPsnrCalc, memobj_t &refBuffer)
  : m_Format(fmt)
  , m_memBuffers(buffers)
  , m_cbFrameBytes(bufsize)
  , m_pitch(pitch)
  , m_memtype(mem_type)
  , m_FrameNo(0)
  , m_pPsnrCalc(pPsnrCalc)
  , m_memRefBuffer(refBuffer)
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

    com_ptr<ICC_VideoProducerExtAsync2> spProdExtAsync2;
    pUnk->QueryInterface(IID_ICC_VideoProducerExtAsync2, (void**)&spProdExtAsync2);

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
      auto target_buffer = m_memBuffers[m_FrameNo % m_memBuffers.size()];

      if(spProdExtAsync2)
      {
        void *pData; DWORD cbSize; CC_MEMORY_TYPE mem_type;

        if(FAILED(hr = spProdExtAsync2->GetDecodeFrameAsyncSurface2(&pData, &cbSize, &mem_type, NULL)))
          return hr;

        target_buffer.Ptr = pData;
        target_buffer.Size = cbSize;
      }
      else
      {
        if(FAILED(hr = spProducer->GetFrame(m_Format, (BYTE*)target_buffer, m_cbFrameBytes, m_pitch)))
          return hr;
      }
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
        memobj_t memTmpBuffer = {}, memTmpRefBuffer = {};
        
        BYTE* pBuffer    = (BYTE*)target_buffer;
        BYTE *pRefBuffer = (BYTE*)m_memRefBuffer;

        if (g_mem_type == MEM_GPU)
		{
          if(g_cudaContext)
            if(auto err = cuCtxPushCurrent(g_cudaContext))
              return fprintf(stderr, "cuCtxPushCurrent() error %d (%s)\n", err, GetCudaDrvApiErrorText(err)), E_FAIL;

          memTmpBuffer    = mem_alloc(MEM_PINNED, m_cbFrameBytes);
		  memTmpRefBuffer = mem_alloc(MEM_PINNED, m_cbFrameBytes);

          pBuffer    = (BYTE*)(memTmpBuffer);
          pRefBuffer = (BYTE*)(memTmpRefBuffer);

		  if(!memTmpBuffer || !memTmpRefBuffer)
		  {
		    hr = E_OUTOFMEMORY;
		  }
		  else
		  {
            mem_copy(pBuffer, target_buffer, m_cbFrameBytes);
            mem_copy(pRefBuffer, m_memRefBuffer, m_cbFrameBytes);
		  }
		}

        CC_VIDEO_QUALITY_MEASUREMENT psnr = {};

        if(SUCCEEDED(hr))
        {
          hr = m_pPsnrCalc->Measure(CC_VQM_PSNR, m_szFrame, m_Format, 
								    pBuffer   , m_cbFrameBytes, m_pitch,
								    pRefBuffer, m_cbFrameBytes, m_pitch,
								    &psnr);
		}

		if(g_mem_type == MEM_GPU)
		{
		  mem_release(memTmpBuffer);
		  mem_release(memTmpRefBuffer);
		}

        if(g_cudaContext)
          cuCtxPopCurrent(NULL);
        
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

