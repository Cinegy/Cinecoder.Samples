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
public:
	ID3D11Resource* GetD3DX11ResourcePtr() { return m_pResource; }
	void InitD3DResource(ID3D11Resource* pResource, size_t _iWidth, size_t _iHeight, size_t _iStride, size_t _iSize)
	{ 
		Destroy();

		iWidth = _iWidth;
		iHeight = _iHeight;

		iPitch = _iStride;
		iSizeFrame = iPitch * iHeight;

		m_pResource = pResource;
	}
#endif

private:
	std::vector<unsigned char> frame_buffer;

	cudaPtr	pKernelDataOut;

private:
	void Initialize();

public:
	int AddRef() { return 2; }
	int Release() { return 1; }

public:
	unsigned char* DataPtr() 
	{ 
		return frame_buffer.data(); 
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
	void Destroy();
};

