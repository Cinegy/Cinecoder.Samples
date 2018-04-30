#include "stdafx.h"
#include "Block.h"

C_Block::C_Block()
{
	Initialize();
}

C_Block::~C_Block()
{
	Destroy();
}

void C_Block::Initialize()
{
	iWidth = iHeight = iPitch = iSizeFrame = 0;

	bRotateFrame = false;
}

long C_Block::Init(size_t _iWidth, size_t _iHeight, size_t _iStride)
{
	Destroy();

	iWidth = _iWidth;
	iHeight = _iHeight;

	iPitch = _iStride;
	iSizeFrame = iPitch * iHeight;

	frame_buffer.resize(iSizeFrame);

	if (frame_buffer.size() != iSizeFrame)
		return -1;

	return 0;
}

void C_Block::Destroy()
{
	frame_buffer.clear();

	Initialize();
}