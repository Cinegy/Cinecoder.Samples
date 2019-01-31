#pragma once

// Cinecoder
#include <Cinecoder_h.h>

// License
#include "../common/cinecoder_license_string.h"

// Cinegy utils
#include "utils/comptr.h"

#if defined(__WIN32__)
#include <Al/al.h>
#include <Al/alc.h>
#elif defined(__APPLE__)
#include <OpenAL/OpenAL.h>
#elif defined(__LINUX__)
#include <AL/al.h>
#include <AL/alc.h>
#endif

#ifndef NDEBUG
	#define __al { \
	ALuint alRes = alGetError(); \
	if (alRes != AL_NO_ERROR) { \
	printf("al error = 0x%x line = %d\n", alRes, __LINE__); \
	return alRes; \
	} }

#define __al_void { \
	ALuint alRes = alGetError(); \
	if (alRes != AL_NO_ERROR) { \
	printf("al error = 0x%x line = %d\n", alRes, __LINE__); \
	} }
#else
	#define __al
	#define __al_void
#endif

#define NUM_BUFFERS 6

class AudioSource : public C_SimpleThread<AudioSource>
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
	size_t m_iBlockAlign;

	int m_iSpeed;

	CC_FRAME_RATE m_FrameRate;

	ALCdevice *device;
	ALCcontext *context;

	ALuint source;
	ALuint buffers[NUM_BUFFERS];

	bool m_bAudioPause;
	bool m_bProcess;

	C_CritSec m_CritSec;
	std::queue<size_t> queueFrames;

public:
	AudioSource();
	~AudioSource();

public:
	int Init(CC_FRAME_RATE video_framerate);
	int OpenFile(const char* const filename);
	int PlayFrame(size_t iFrame);
	int SetPause(bool bPause);

	bool IsPause() { return m_bAudioPause; }
	bool IsInitialize() { return m_bInitialize; }

	void SetSpeed(int iSpeed)
	{
		m_iSpeed = iSpeed;
	}

	void SetVolume(float fVolume)
	{
		if (fVolume >= 0.f && fVolume <= 1.f)
		{
			alSourcef(source, AL_GAIN, fVolume); __al_void
		}
	}

	float GetVolume()
	{
		float fVolume = 0;
		alGetSourcef(source, AL_GAIN, &fVolume); __al_void

		return fVolume;
	}

private:
	int InitOpenAL();
	int DestroyOpenAL();
	int PrintVersionAL();

	HRESULT UpdateAudioChunk(size_t iFrame, ALvoid** data, ALsizei* size);

private:
	friend class C_SimpleThread<AudioSource>;
	long ThreadProc();
};

