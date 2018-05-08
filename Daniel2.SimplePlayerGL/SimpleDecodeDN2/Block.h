#pragma once

#include <vector>

class C_Block
{
public:
	C_Block();
	~C_Block();

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

private:
	std::vector<unsigned char> frame_buffer;

private:
	void Initialize();

public:
	int AddRef() { return 2; }
	int Release() { return 1; }

public:
	unsigned char* DataPtr() { return frame_buffer.data(); }

	size_t Width() { return iWidth; }
	size_t Height() { return iHeight; }
	size_t Pitch() { return iPitch; }
	size_t Size() { return iSizeFrame; }

	void SetRotate(bool bRotate) { bRotateFrame = bRotate; }
	bool GetRotate() { return bRotateFrame; }

	long Init(size_t _iWidth, size_t _iHeight, size_t _iStride);

	void Destroy();
};

