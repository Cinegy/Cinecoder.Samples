//------------------------------------------------------------------
class C_FileWriter : public C_Unknown, public ICC_ByteStreamCallback
//------------------------------------------------------------------
{
  FILE *m_File;

public:
  C_FileWriter(FILE *f) : m_File(f)
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
  	
  	return S_OK;
  }
};
