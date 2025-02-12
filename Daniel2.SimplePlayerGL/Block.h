#pragma once

typedef void* cudaPtr;

class C_Block
{
public:
	C_Block();
	~C_Block();

	C_Block(const C_Block&) = default;
	C_Block(C_Block&&) = default;

private:
	//C_Block(const C_Block&);
	C_Block& operator=(const C_Block&);

private:
	size_t			iWidth;
	size_t			iHeight;
	size_t			iPitch;
	size_t			iSizeFrame;

	bool			bRotateFrame;

public:
	size_t			iFrameNumber;

	size_t			iMatrixCoeff_YUYtoRGBA;

#if defined(__WIN32__)
private:
	ID3D11Resource*		m_pResource;
	com_ptr<IDXGIKeyedMutex> m_pKeyedMutex;
public:
	ID3D11Resource* GetD3DX11ResourcePtr() { return m_pResource; }
	void InitD3DResource(ID3D11Resource* pResource, size_t _iWidth, size_t _iHeight, size_t _iStride, size_t _iSize)
	{ 
		Destroy();

		iWidth = _iWidth;
		iHeight = _iHeight;

		iPitch = _iStride;
		iSizeFrame = iPitch * iHeight;

		m_pKeyedMutex = nullptr;

		HRESULT  hr = S_OK;

		com_ptr<ID3D11Buffer> pBuffer;
		hr = pResource->QueryInterface(__uuidof(ID3D11Buffer), reinterpret_cast<void**>(&pBuffer));

		D3D11_BUFFER_DESC bufDesc = {};
		if (SUCCEEDED(hr) && pBuffer) pBuffer->GetDesc(&bufDesc);
		if (SUCCEEDED(hr) && (bufDesc.MiscFlags & D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX))
		{
			hr = pResource->QueryInterface(__uuidof(IDXGIKeyedMutex), reinterpret_cast<void**>(&m_pKeyedMutex));
		}

		m_pResource = pResource;		
	}
	HRESULT D3DX11ResourceLock() { if (m_pKeyedMutex) return m_pKeyedMutex->AcquireSync(0, INFINITE); else return S_OK; }
	HRESULT D3DX11ResourceUnLock() { if (m_pKeyedMutex) return m_pKeyedMutex->ReleaseSync(0); else return S_OK; }
#endif

private:
	unsigned char* frame_buffer;

	cudaPtr	pKernelDataOut;

private:
	void Initialize();

public:
	int AddRef() { return 2; }
	int Release() { return 1; }

public:
	unsigned char* DataPtr() 
	{ 
		return frame_buffer;
	}

	unsigned char* DataGPUPtr()
	{
		return (unsigned char*)pKernelDataOut;
	}

	size_t Width() { return iWidth; }
	size_t Height() { return iHeight; }
	size_t Pitch() { return iPitch; }
	size_t Size() { return iSizeFrame; }

	void SetRotate(bool bRotate) { bRotateFrame = bRotate; }
	bool GetRotate() { return bRotateFrame; }

	long Init(size_t _iWidth, size_t _iHeight, size_t _iStride, size_t _iSize, bool bUseCuda = false);

	int CopyToGPU();
	int CopyToCPU();

	void Destroy();
};

