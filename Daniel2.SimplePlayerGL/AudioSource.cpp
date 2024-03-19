#include "stdafx.h"
#include "AudioSource.h"

#if defined(__APPLE__) || defined(__LINUX__)
typedef signed char INT8;
typedef signed short INT16;
//typedef long long UINT64;
#endif

static void ReverseSamples(BYTE *p, int iSize, int nBlockAlign)
{
	long lActual = iSize;

	if (lActual == 0) { _assert(0); return; };
	if (nBlockAlign != 4) { _assert(0); return; };

	UINT64 *p_src = (UINT64 *)(p);
	UINT64 *p_dst = ((UINT64 *)(p + lActual)) - 1;
	UINT64 temp;

	while (p_src < p_dst)
	{
		temp = *p_src;
		*p_src++ = *p_dst;
		*p_dst-- = temp;
	};
}

static void AliasingSamples(BYTE *p, int iSize, int nBlockAlign, int nChannels)
{
	long lActual = iSize;

	const long iMaxValue = 32;

	if (lActual == 0) { _assert(0); return; };
	if (nBlockAlign != 4) { _assert(0); return; };

	if (lActual / nBlockAlign < 2 * iMaxValue) return;

	INT16 *p_Beg = (INT16 *)(p);
	INT16 *p_End = ((INT16 *)(p + lActual)) - 1;
	float ftemp;

	for (long i = 0; i <= iMaxValue; i++)
	{
		ftemp = ((float)i / (float)iMaxValue);
		for (long ic = 0; ic < nChannels; ic++)
		{
			*p_Beg = (INT16)((float)*p_Beg * ftemp); p_Beg++;
			*p_End = (INT16)((float)*p_End * ftemp); p_End--;
		}
	}
}

static void list_audio_devices(const ALCchar *devices)
{
	const ALCchar *device = devices, *next = devices + 1;
	size_t len = 0;

	fprintf(stdout, "Devices list:\n");
	fprintf(stdout, "----------\n");
	while (device && *device != '\0' && next && *next != '\0') {
		fprintf(stdout, "%s\n", device);
		len = strlen(device);
		device += (len + 1);
		next += (len + 2);
	}
	fprintf(stdout, "----------\n");
}

AudioSource::AudioSource() :
	m_bInitialize(false),
	device(nullptr),
	context(nullptr),
	source(0),
	buffers{ 0 },
	m_FrameRate{ 25, 1 },
	m_bAudioPause(false),
	m_iSpeed(1),
	m_bProcess(false)
{
	m_iSampleCount = 0;
	m_iSampleRate = 0;
	m_iSampleBytes = 0;
	m_iNumChannels = 0;
	m_iBitsPerSample = 0;
	m_iBlockAlign = 0;

	m_AudioFormat = CAF_PCM16;
	ALformat = AL_FORMAT_STEREO16;
}

AudioSource::~AudioSource()
{
	m_bProcess = false;

	Close(); // closing thread <ThreadProc>

	if (m_bInitialize)
		DestroyOpenAL();
}

int AudioSource::Init(CC_FRAME_RATE video_framerate)
{
	m_FrameRate = video_framerate;

	return 0;
}

int AudioSource::InitOpenAL()
{
	//list_audio_devices(alcGetString(NULL, ALC_DEVICE_SPECIFIER));

	device = alcOpenDevice(NULL);
	//device = alcOpenDevice((ALchar*)"DirectSound3D");

	context = alcCreateContext(device, NULL);

	if (!alcMakeContextCurrent(context))
		return -1;

	alGetError(); /* clear error */

	ALfloat listenerOri[] = { 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f };

	alListener3f(AL_POSITION, 0, 0, 1.0f); __al
	alListener3f(AL_VELOCITY, 0, 0, 0); __al
	alListenerfv(AL_ORIENTATION, listenerOri); __al

	alGenBuffers(NUM_BUFFERS, buffers); __al
	alGenSources(1, &source); __al

	alSourcef(source, AL_PITCH, 1); __al
	alSourcef(source, AL_GAIN, 1); __al
	alSource3f(source, AL_POSITION, 0, 0, 0); __al
	alSource3f(source, AL_VELOCITY, 0, 0, 0); __al
	alSourcei(source, AL_LOOPING, AL_FALSE); __al

	for (size_t i = 0; i < NUM_BUFFERS; i++)
	{
		HRESULT hr = S_OK;
		DWORD cbRetSize = 0;

		BYTE* pb = audioChunk.data();
		DWORD cb = static_cast<DWORD>(audioChunk.size());
		
		//hr = m_pMediaReader->GetAudioSamples(m_AudioFormat, i * m_iSampleCount, (CC_UINT)m_iSampleCount, pb, cb, &cbRetSize);
		cbRetSize = cb;
		if (SUCCEEDED(hr) && cbRetSize > 0)
		{
			ALvoid* data = pb;
			ALsizei size = static_cast<ALsizei>(cbRetSize);
			ALsizei frequency = static_cast<ALsizei>(m_iSampleRate);
			ALenum  format = ALformat;

			memset(data, 0x00, size);

			alBufferData(buffers[i], format, data, size, frequency); __al
		}
	}

	alSourceQueueBuffers(source, NUM_BUFFERS, buffers); __al

	alSourcePlay(source); __al

	m_bInitialize = true;

	PrintVersionAL();

	Create(); // creating thread <ThreadProc>

	return 0;
}

int AudioSource::DestroyOpenAL()
{
	alSourceStop(source); __al

	alDeleteSources(1, &source); __al
	alDeleteBuffers(NUM_BUFFERS, buffers); __al

	device = alcGetContextsDevice(context); __al
	alcMakeContextCurrent(NULL); __al
	alcDestroyContext(context); __al
	alcCloseDevice(device); __al

	return 0;
}

int AudioSource::PrintVersionAL()
{
	if (!device)
		return -1;

	static const ALchar alVendor[] = "OpenAL Community";
	static const ALchar alVersion[] = "1.1 ALSOFT ";
	static const ALchar alRenderer[] = "OpenAL Soft";

	const ALCchar* _alVendor = alcGetString(device, AL_VENDOR); __al
	const ALCchar* _alVersion = alcGetString(device, AL_VERSION); __al
	const ALCchar* _alRenderer = alcGetString(device, AL_RENDERER); __al

	printf("OpenAL vendor : %s\n", _alVendor == nullptr ? alVendor : _alVendor);
	printf("OpenAL renderer : %s\n", _alVersion == nullptr ? alVersion : _alVersion);
	printf("OpenAL version : %s\n", _alRenderer == nullptr ? alRenderer : _alRenderer);

	printf("-------------------------------------\n");

	return 0;
}

int AudioSource::OpenFile(const char* const filename)
{
	HRESULT hr = S_OK;

	com_ptr<ICC_ClassFactory> piFactory;

	Cinecoder_CreateClassFactory((ICC_ClassFactory**)&piFactory); // get Factory
	if (FAILED(hr)) return hr;

	hr = piFactory->AssignLicense(COMPANYNAME, LICENSEKEY); // set license
	if (FAILED(hr))
		return printf("AudioSource::OpenFile: AssignLicense failed!\n"), hr;

	hr = piFactory->CreateInstance(CLSID_CC_MediaReader, IID_ICC_MediaReader, (IUnknown**)&m_pMediaReader);
	if (FAILED(hr)) return hr;

#if defined(__WIN32__)
	CC_STRING file_name_str = _com_util::ConvertStringToBSTR(filename);
#elif defined(__APPLE__) || defined(__LINUX__)
	CC_STRING file_name_str = const_cast<CC_STRING>(filename);
#endif

	hr = m_pMediaReader->Open(file_name_str);
	if (FAILED(hr)) return hr;

	CC_INT numAudioTracks = 0;
	hr = m_pMediaReader->get_NumberOfAudioTracks(&numAudioTracks);
	if (FAILED(hr)) return hr;

	if (numAudioTracks == 0)
	{
		printf("numAudioTracks == 0\n");
		m_pMediaReader = nullptr;
		return -1;
	}

	CC_INT iCurrentAudioTrackNumber = 0;

	for (CC_INT i = 0; i < numAudioTracks; i++)
	{
		hr = m_pMediaReader->put_CurrentAudioTrackNumber(i);
		if (FAILED(hr)) return hr;

		com_ptr<ICC_AudioStreamInfo> pAudioStreamInfo;
		hr = m_pMediaReader->get_CurrentAudioTrackInfo((ICC_AudioStreamInfo**)&pAudioStreamInfo);
		if (FAILED(hr)) return hr;

		CC_TIME Duration = 0;
		hr = m_pMediaReader->get_Duration(&Duration);
		if (FAILED(hr)) return hr;

		CC_INT FrameCount = 0;
		hr = m_pMediaReader->get_NumberOfFrames(&FrameCount);
		if (FAILED(hr)) return hr;

		CC_FRAME_RATE FrameRateMR;
		hr = m_pMediaReader->get_FrameRate(&FrameRateMR);
		if (FAILED(hr)) return hr;

		CC_BITRATE BitRate;
		CC_UINT BitsPerSample;
		CC_UINT ChannelMask;
		CC_FRAME_RATE FrameRateAS;
		CC_UINT NumChannels;
		CC_UINT SampleRate;
		CC_ELEMENTARY_STREAM_TYPE StreamType;

		hr = pAudioStreamInfo->get_BitRate(&BitRate);
		if (FAILED(hr)) return hr;

		hr = pAudioStreamInfo->get_BitsPerSample(&BitsPerSample);
		if (FAILED(hr)) return hr;

		hr = pAudioStreamInfo->get_ChannelMask(&ChannelMask);
		if (FAILED(hr)) return hr;

		hr = pAudioStreamInfo->get_FrameRate(&FrameRateAS);
		if (FAILED(hr)) return hr;

		hr = pAudioStreamInfo->get_NumChannels(&NumChannels);
		if (FAILED(hr)) return hr;

		hr = pAudioStreamInfo->get_SampleRate(&SampleRate);
		if (FAILED(hr)) return hr;

		hr = pAudioStreamInfo->get_StreamType(&StreamType);
		if (FAILED(hr)) return hr;

		printf("audio track #%d: ", i);
		switch (StreamType)
		{
		case CC_ES_TYPE_AUDIO_AAC: printf("AAC / "); break;
		case CC_ES_TYPE_AUDIO_AC3: printf("AC3 / "); break;
		case CC_ES_TYPE_AUDIO_AC3_DVB: printf("AC3_DVB / "); break;
		case CC_ES_TYPE_AUDIO_AES3: printf("AES3 / "); break;
		case CC_ES_TYPE_AUDIO_D_E: printf("DOLBY E / "); break;
		case CC_ES_TYPE_AUDIO_DTS: printf("DTS / "); break;
		case CC_ES_TYPE_AUDIO_LATM: printf("LATM / "); break;
		case CC_ES_TYPE_AUDIO_LPCM: printf("LPCM / "); break;
		case CC_ES_TYPE_AUDIO_MPEG1: printf("MPEG1 / "); break;
		case CC_ES_TYPE_AUDIO_MPEG2: printf("MPEG2 / "); break;
		case CC_ES_TYPE_AUDIO_SMPTE302: printf("SMPTE302 / "); break;
		}
		if (NumChannels == 1) printf("1 channel / ");
		else printf("%lu channels / ", static_cast<unsigned long>(NumChannels));
		printf("%.2f kHz / ", ((double)SampleRate / 1000.0));
		printf("%lu bits", static_cast<unsigned long>(BitsPerSample));
		printf("\n");

		if (iCurrentAudioTrackNumber == i)
		{
			//BitsPerSample = 16; // always play in PCM16
			BitsPerSample = BitsPerSample;

			if (BitsPerSample == 8)
				m_AudioFormat = CAF_PCM8;
			else if (BitsPerSample == 16)
				m_AudioFormat = CAF_PCM16;

			if (FrameRateMR.num != 0)
				m_FrameRate = FrameRateMR;
			else if (FrameRateAS.num != 0)
				m_FrameRate = FrameRateAS;

			size_t sample_count = (SampleRate / (m_FrameRate.num / m_FrameRate.denom));
			size_t sample_bytes = sample_count * NumChannels * (BitsPerSample >> 3);

			m_iSampleCount = sample_count;
			m_iSampleRate = SampleRate;
			m_iSampleBytes = sample_bytes;
			m_iNumChannels = NumChannels;
			m_iBitsPerSample = BitsPerSample;
			m_iBlockAlign = (m_iNumChannels * m_iBitsPerSample) / 8;

			ALformat = (m_iNumChannels == 2) ?
				((m_AudioFormat == CAF_PCM8) ? AL_FORMAT_STEREO8 : AL_FORMAT_STEREO16)
				: ((m_AudioFormat == CAF_PCM8) ? AL_FORMAT_MONO8 : AL_FORMAT_MONO16);

			audioChunk.resize(sample_bytes);
		}
	}

	printf("-------------------------------------\n");

	hr = m_pMediaReader->put_CurrentAudioTrackNumber(iCurrentAudioTrackNumber);
	if (FAILED(hr)) return hr;

	if (m_iBitsPerSample != 16)
	{
		printf("error: BitsPerSample != 16 bits (support only 16 bits)\n");
		m_pMediaReader = nullptr;
		return -1;
	}

	return InitOpenAL();
}

int AudioSource::PlayFrame(size_t iFrame)
{
	if (!m_bInitialize)
		return -1;

	if (!m_bAudioPause)
	{
		C_AutoLock lock(&m_CritSec);

		if (queueFrames.size() > NUM_BUFFERS)
			queueFrames.pop();

		queueFrames.push(iFrame);
	}

	return 0;
}

int AudioSource::SetPause(bool bPause)
{
	if (!m_bInitialize)
		return -1;

	m_bAudioPause = bPause;

	if (m_bAudioPause)
	{
		alSourcePause(source); __al
	}

	return 0;
}

long AudioSource::ThreadProc()
{
	m_bProcess = true;

	size_t iCurFrame = NUM_BUFFERS;

	while (m_bProcess)
	{
		ALint numProcessed = 0;
		alGetSourcei(source, AL_BUFFERS_PROCESSED, &numProcessed); __al

		ALint source_state = 0;
		alGetSourcei(source, AL_SOURCE_STATE, &source_state); __al

		if (numProcessed > 0)
		{
			if (queueFrames.size() > 0)
			{
				{
					C_AutoLock lock(&m_CritSec);

					iCurFrame = queueFrames.front();
					queueFrames.pop();
				}

				ALvoid* data = nullptr;
				ALsizei size = 0;
				if (UpdateAudioChunk(iCurFrame, &data, &size) == S_OK && data && size > 0)
				{
					ALsizei frequency = static_cast<ALsizei>(m_iSampleRate);
					ALenum  format = ALformat;
					ALuint buffer;

					alSourceUnqueueBuffers(source, 1, &buffer); __al
					alBufferData(buffer, format, data, size, frequency); __al
					alSourceQueueBuffers(source, 1, &buffer); __al
				}
			}
		}

		if (source_state != AL_PLAYING && !m_bAudioPause)
		{
			alSourcePlay(source); __al
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}

	return 0;
}

HRESULT AudioSource::UpdateAudioChunk(size_t iFrame, ALvoid** data, ALsizei* size)
{
	if (!m_pMediaReader)
		return E_FAIL;

	HRESULT hr = S_OK;
	DWORD cbRetSize = 0;

	BYTE* pb = audioChunk.data();
	DWORD cb = static_cast<DWORD>(audioChunk.size());

	hr = m_pMediaReader->GetAudioSamples(m_AudioFormat, iFrame * m_iSampleCount, (CC_UINT)m_iSampleCount, pb, cb, &cbRetSize);
	if (SUCCEEDED(hr) && cbRetSize > 0)
	{
		// if we playing in the opposite direction we need reverse audio samples
		if (m_iSpeed < 0)
			ReverseSamples(pb, (int)cbRetSize, (int)m_iBlockAlign);

		// if we playing with speed > 1 we need aliasing our audio samples
		if (abs(m_iSpeed) > 1)
			AliasingSamples(pb, (int)cbRetSize, (int)m_iBlockAlign, (int)m_iNumChannels);

		*data = pb;
		*size = static_cast<ALsizei>(cbRetSize);
	}

	return hr;
}
