#pragma once

// Cinecoder
#include <Cinecoder_h.h>

// License
#include "../cinecoder_license_string.h"

// Cinegy utils
#include "utils/comptr.h"

#if defined(__WIN32__) || defined(_WIN32)
#include <Al/al.h>
#include <Al/alc.h>
#else
#include <OpenAL/OpenAL.h>
#endif

#ifndef NDEBUG
	#define __al { \
	ALuint alRes = alGetError(); \
	if (alRes != AL_NO_ERROR) { \
	printf("al error = 0x%x line = %d\n", alRes, __LINE__); \
	return alRes; \
	} }
#else
	#define __al
#endif

#define NUM_BUFFERS 6

class AudioSource
{
private:
	bool m_bInitialize;

	com_ptr<ICC_MediaReader> m_pMediaReader;
	com_ptr<ICC_AudioStreamInfo> m_pAudioStreamInfo;

	std::vector<BYTE> audioChunk;

	size_t m_iSampleCount;
	size_t m_iSampleRate;
	size_t m_iSampleBytes;
	size_t m_iNumChannels;
	size_t m_iBitsPerSample;

	int m_iSpeed;

	CC_FRAME_RATE m_FrameRate;

	ALCdevice *device;
	ALCcontext *context;

	ALuint source;
	ALuint buffers[NUM_BUFFERS];

	bool m_bAudioPause;

public:
	AudioSource();
	~AudioSource();

public:
	int Init(CC_FRAME_RATE video_framerate);
	int OpenFile(const char* const filename);
	int PlayFrame(size_t iFrame);
	int SetPause(bool bPause);

	bool IsPause() { return m_bAudioPause; }

	void SetSpeed(int iSpeed)
	{
		m_iSpeed = iSpeed;
	}

private:
	int InitOpenAL();
	int DestroyOpenAL();
	int PrintVersionAL();
};
