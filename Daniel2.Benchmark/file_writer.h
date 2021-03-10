#include "implement_iunknown_static.h"
#ifdef __LINUX__
#include <malloc.h>
#endif
#include <vector>
#include <utility>
#include <atomic>

//------------------------------------------------------------------
class C_FileWriter : public ICC_ByteStreamCallback, public ICC_DataWriterEx
//------------------------------------------------------------------
{
  _IMPLEMENT_IUNKNOWN_2_STATICALLY(ICC_ByteStreamCallback, ICC_DataWriterEx)

  FILE *m_File;

  size_t m_NumFrames;
  std::atomic<long long> m_TotalDataSize;

  bool m_bCatchFrames;
  size_t m_MinSeqLength;

  typedef std::pair<BYTE*,size_t> CodedFrameData;
  std::vector< CodedFrameData > m_CaughtFrames;

public:
  C_FileWriter(FILE *f, bool bCatchFrames = true, size_t min_seq_length = 1) : m_File(f), m_bCatchFrames(bCatchFrames), m_MinSeqLength(min_seq_length)
  {
    m_NumFrames = 0;
    m_TotalDataSize = 0;
  }

  //-----------------------------------------------------------------------------
  virtual ~C_FileWriter()
  //-----------------------------------------------------------------------------
  {
  }

  //-----------------------------------------------------------------------------
  STDMETHOD(ProcessData)(const BYTE *pbData, CC_AMOUNT cbSize, CC_TIME, IUnknown *pProducer)
  //-----------------------------------------------------------------------------
  {
    if(m_bCatchFrames)
    {
	  com_ptr<ICC_VideoEncoder> pVProd;
	  if(S_OK == pProducer->QueryInterface(IID_ICC_VideoEncoder, (void**)&pVProd))
      {
        com_ptr<ICC_VideoFrameInfo> pFrameInfo;
        pVProd->GetVideoFrameInfo(&pFrameInfo);

        CC_FRAME_TYPE frame_type;
        pFrameInfo->get_FrameType(&frame_type);

        if(frame_type == CC_I_FRAME && !m_CaughtFrames.empty() && m_NumFrames >= m_MinSeqLength)
        {
          printf("\n@@: collected %zd frame(s)\n", m_CaughtFrames.size());
          m_bCatchFrames = false;
        }
        else
        {
          BYTE *ptr = (BYTE*)malloc(cbSize);
          memcpy(ptr, pbData, cbSize);
          m_CaughtFrames.push_back( std::make_pair(ptr, cbSize) );
        }
      }
    }

  	return Write(pbData, cbSize);
  }

  size_t GetCodedSequenceLength() const
  {
    return m_CaughtFrames.size();
  }

  CodedFrameData GetCodedFrame(size_t i)
  {
    return m_CaughtFrames[i];
  }

  //-----------------------------------------------------------------------------
  STDMETHOD(Write)(const BYTE *pbData, CC_AMOUNT cbSize)
  //-----------------------------------------------------------------------------
  {
    if(m_NumFrames == 0)
      g_EncoderTimeFirstFrameOut = system_clock::now();

  	m_TotalDataSize += cbSize;
  	m_NumFrames++;
  	//InterlockedCompareExchange(&g_target_bitrate, 

  	if(m_File == NULL)
  	  return S_FALSE;

  	if(fwrite(pbData, 1, cbSize, m_File) != cbSize)
  	  return E_FAIL;
    	
  	return S_OK;
  }

  //-----------------------------------------------------------------------------
  STDMETHOD(WriteDirect)(long long offs, const BYTE *pbData, CC_AMOUNT cbSize)
  //-----------------------------------------------------------------------------
  {
  	if(m_File == NULL)
  	  return S_FALSE;

    long long prev = _ftelli64(m_File);
    _fseeki64(m_File, offs, SEEK_SET);
    HRESULT hr = Write(pbData, cbSize);
    _fseeki64(m_File, prev, SEEK_SET);
    return hr;
  }

  //-----------------------------------------------------------------------------
  STDMETHOD(GetCurrentOffset)(long long *ppos)
  //-----------------------------------------------------------------------------
  {
  	if(m_File == NULL)
  	  return E_ACCESSDENIED;

  	if(ppos == NULL)
  	  return E_POINTER;

  	*ppos = _ftelli64(m_File);

  	return S_OK;
  }

  //-----------------------------------------------------------------------------
  long long GetTotalBytesWritten()
  //-----------------------------------------------------------------------------
  {
    return m_TotalDataSize;
  }
};
