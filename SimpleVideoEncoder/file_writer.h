//------------------------------------------------------------------
class C_FileWriter : public C_Unknown, public ICC_ByteStreamCallback
//------------------------------------------------------------------
{
  FILE *m_File;
  __int64& m_total_size;

public:
  C_FileWriter(FILE *f, __int64 *p_total_size) : m_File(f), m_total_size(*p_total_size)
  {
  }

  //-----------------------------------------------------------------------------
  virtual ~C_FileWriter()
  //-----------------------------------------------------------------------------
  {
  }

  _IMPLEMENT_IUNKNOWN_1(ICC_ByteStreamCallback);                                 \

  //-----------------------------------------------------------------------------
  STDMETHOD(ProcessData)(const BYTE *pbData, CC_AMOUNT cbSize, CC_TIME, IUnknown*)
  //-----------------------------------------------------------------------------
  {
  	if(m_File == NULL)
  	  return E_ACCESSDENIED;

  	if(fwrite(pbData, 1, cbSize, m_File) != cbSize)
  	  return E_FAIL;
  	
  	m_total_size += cbSize;

  	return S_OK;
  }
};
