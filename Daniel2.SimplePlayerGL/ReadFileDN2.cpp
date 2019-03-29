#include "stdafx.h"
#include "ReadFileDN2.h"

ReadFileDN2::ReadFileDN2() :
	m_bProcess(false),
	m_bReadFile(false),
	m_bSeek(false),
	m_iSpeed(1)
{
#ifdef __FILE_READ__
	m_file = nullptr;
#endif
}

ReadFileDN2::~ReadFileDN2()
{
	StopPipe();

	CloseFile();

	m_queueFrames.Free();
	m_queueFrames_free.Free();

	m_listFrames.clear();
}

int ReadFileDN2::OpenFile(const char* filename)
{
	CloseFile();

	////////////////////////////

#ifdef __STD_READ__
	m_file.open(filename, std::ofstream::in | std::ifstream::binary);

	if (!m_file.is_open())
		return -1;
#elif __FILE_READ__
	std::string strA(filename);
	std::wstring strW(strA.begin(), strA.end());

	_wfopen_s(&m_file, strW.c_str(), L"rb");

	if (!m_file)
		return -1;
#endif

	////////////////////////////

	HRESULT hr = S_OK;

	com_ptr<ICC_ClassFactory> piFactory;

	Cinecoder_CreateClassFactory((ICC_ClassFactory**)&piFactory);
	if (FAILED(hr)) return hr;

	if (SUCCEEDED(hr)) hr = piFactory->CreateInstance(CLSID_CC_MvxFile, IID_ICC_MvxFile, (IUnknown**)&m_fileMvx);

#if defined(__WIN32__)
	CC_STRING file_name_str = _com_util::ConvertStringToBSTR(filename);
#elif defined(__APPLE__) || defined(__LINUX__)
	CC_STRING file_name_str = const_cast<CC_STRING>(filename);
#endif

	if (SUCCEEDED(hr)) hr = m_fileMvx->Open(file_name_str);

	if (!SUCCEEDED(hr))
		return -1;

	CC_UINT lenght = 0;
	hr = m_fileMvx->get_Length(&lenght);

	m_frames = (CC_UINT)lenght;

	////////////////////////////

	size_t iCountFrames = 7;

	for (size_t i = 0; i < iCountFrames; i++)
	{
		m_listFrames.emplace_back(CodedFrame());
		m_queueFrames_free.Queue(&m_listFrames.back());
	}

	return 0;
}

int ReadFileDN2::CloseFile()
{
	if (m_fileMvx)
		m_fileMvx->Close();

#ifdef __STD_READ__
	m_file.close();
#elif __FILE_READ__
	if (m_file) 
		fclose(m_file);
#endif
	return 0;
}

int ReadFileDN2::StartPipe()
{
	Create();

	return 0;
}

int ReadFileDN2::StopPipe()
{
	m_bProcess = false;
	m_bReadFile = false;

	Close();

	return 0;
}

int ReadFileDN2::ReadFrame(size_t frame, std::vector<unsigned char> & buffer, size_t & size)
{
	C_AutoLock lock(&m_critical_read);

	if (frame >= m_frames)
		return -1;

#ifdef __STD_READ__
	if (m_file.is_open())
#elif __FILE_READ__
	if (m_file)
#endif
	{
		CC_MVX_ENTRY Idx;
		if (!SUCCEEDED(m_fileMvx->FindEntryByCodingNumber((CC_UINT)frame, &Idx)))
			return -1;

		size_t offset = (size_t)Idx.Offset;
		size = Idx.Size;

		if (buffer.size() < size)
			buffer.resize(size);

#ifdef __STD_READ__
		m_file.seekg(offset, m_file.beg);
		m_file.read((char*)buffer.data(), size);
#elif __FILE_READ__
		_fseeki64(m_file, offset, SEEK_SET);
		fread(buffer.data(), size, 1, m_file);
#endif
		return 0;
	}

	return -1;
}

CodedFrame* ReadFileDN2::MapFrame()
{
	C_AutoLock lock(&m_critical_queue);

	CodedFrame *pFrame = nullptr;

	m_queueFrames.Get(&pFrame, m_evExit);

	return pFrame;
}

void ReadFileDN2::UnmapFrame(CodedFrame* pFrame)
{
	C_AutoLock lock(&m_critical_queue);

	if (pFrame)
	{
		m_queueFrames_free.Queue(pFrame);
	}
}

long ReadFileDN2::ThreadProc()
{
	int iCurEncodedFrame = 0;

	m_bProcess = true;
	m_bReadFile = true;

	data_rate = 0;

	int res = 0;
	bool bSeek = false;

	while (m_bProcess)
	{
		CodedFrame* frame = nullptr;

		m_queueFrames_free.Get(&frame, m_evExit);

		if (frame)
		{
			res = 0;

			if (m_bReadFile)
			{
				res = ReadFrame(iCurEncodedFrame, frame->coded_frame, frame->coded_frame_size);
				data_rate += frame->coded_frame_size;
			}
			
			frame->flags = 0;
			if (bSeek) { frame->flags = 1; bSeek = false; }

			frame->frame_number = iCurEncodedFrame;
			m_queueFrames.Queue(frame);

			if (res != 0)
			{
				assert(0);
				printf("ReadFrame failed res=%d coded_frame_size=%zu coded_frame=%p\n", res, frame->coded_frame_size, frame->coded_frame.data());
			}
		}

		iCurEncodedFrame += m_iSpeed;

		if (iCurEncodedFrame >= (int)m_frames)
			iCurEncodedFrame = 0;
		else if (iCurEncodedFrame < 0)
			iCurEncodedFrame = (int)m_frames - 1;

		if (m_bSeek)
		{
			{
				C_AutoLock lock(&m_critical_queue);
				for (size_t i = 0; i < m_queueFrames.GetCount(); ++i)
				{
					CodedFrame *pFrame = nullptr;
					m_queueFrames.Get(&pFrame, m_evExit);
					if (pFrame)
						m_queueFrames_free.Queue(pFrame);
				}
			}

			iCurEncodedFrame = (int)m_iSeekFrame;
			m_bSeek = false;
			bSeek = true;
		}
	}

	return 0;
}

