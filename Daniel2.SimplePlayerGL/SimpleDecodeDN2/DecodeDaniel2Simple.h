#pragma once

#include "Block.h"

enum IMAGE_FORMAT { IMAGE_FORMAT_RGBA8BIT, IMAGE_FORMAT_RGBA16BIT, IMAGE_FORMAT_RGB30 };

class DecodeDaniel2Simple
{
private:
	std::wstring m_filename;

	size_t m_width;
	size_t m_height;
	size_t m_stride;
	IMAGE_FORMAT m_outputImageFormat;

	bool m_bProcess;
	bool m_bInitDecoder;

	C_Event m_hExitEvent;
	C_QueueT<C_Block> m_queueFrames;

	std::list<C_Block> m_listBlocks;

public:
	int OpenFile(const char* const filename, size_t iMaxCountDecoders = 2)
	{
		m_outputImageFormat = IMAGE_FORMAT_RGBA8BIT;

		m_width = 720;
		m_height = 576;
		m_stride = m_width * 4;

		InitValues();

		return 0;
	}
	int StartDecode()
	{
		m_bProcess = true;
		return 0;
	}
	int StopDecode()
	{
		m_bProcess = false;
		return 0;
	}
	size_t GetImageWidth() { return m_width; }
	size_t GetImageHeight() { return m_height; }
	IMAGE_FORMAT GetImageFormat() { return m_outputImageFormat; }

	C_Block* MapFrame()
	{
		C_Block *pBlock = nullptr;

		m_queueFrames.Get(&pBlock, m_hExitEvent);

		static int si = 0;

		if (si > 255) si = 0; 
		si++;

		for (size_t i = 0; i < pBlock->Size(); i += 4)
		{
			*(pBlock->DataPtr() + i + 0) = si;
			*(pBlock->DataPtr() + i + 1) = si;
			*(pBlock->DataPtr() + i + 2) = si;
			*(pBlock->DataPtr() + i + 3) = 0xFF;
		}

		return pBlock;
	}
	void  UnmapFrame(C_Block* pBlock)
	{
		if (pBlock)
		{
			m_queueFrames.Queue(pBlock);
		}
	}

	bool isProcess() { return m_bProcess; }

private:
	int InitValues()
	{
		size_t iCountBlocks = 4; // set count of blocks in queue

		int res = 0;

		for (size_t i = 0; i < iCountBlocks; i++)
		{
			m_listBlocks.push_back(C_Block());

			m_listBlocks.back().Init(m_width, m_height, m_stride);

			if (res != 0)
			{
				printf("InitBlocks: Init() return error - %d", res);
				return res;
			}

			m_queueFrames.Queue(&m_listBlocks.back()); // add free pointers to queue
		}

		return 0;
	}
};

