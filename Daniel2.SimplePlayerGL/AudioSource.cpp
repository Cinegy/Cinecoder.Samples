#include "stdafx.h"
#include "AudioSource.h"

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
	m_bAudioPause(false)
{
}

AudioSource::~AudioSource()
{
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

		hr = m_pMediaReader->GetAudioSamples(CAF_PCM16, i * m_iSampleCount, (CC_UINT)m_iSampleCount, pb, cb, &cbRetSize);
		if (SUCCEEDED(hr) && cbRetSize > 0)
		{
			ALvoid* data = pb;
			ALsizei size = static_cast<ALsizei>(cbRetSize);
			ALsizei frequency = static_cast<ALsizei>(m_iSampleRate);
			ALenum  format = (m_iNumChannels == 2) ? AL_FORMAT_STEREO16 : AL_FORMAT_MONO16;

			alBufferData(buffers[i], format, data, size, frequency); __al
		}
	}

	alSourceQueueBuffers(source, NUM_BUFFERS, buffers); __al

	alSourcePlay(source); __al

	m_bInitialize = true;

	return 0;
}

int AudioSource::DestroyOpenAL()
{
	alDeleteSources(1, &source); __al
	alDeleteBuffers(NUM_BUFFERS, buffers); __al

	device = alcGetContextsDevice(context); __al
	alcMakeContextCurrent(NULL); __al
	alcDestroyContext(context); __al
	alcCloseDevice(device); __al

	return 0;
}

int AudioSource::OpenFile(const char* const filename)
{
	HRESULT hr = S_OK;

	com_ptr<ICC_ClassFactory> piFactory;

	Cinecoder_CreateClassFactory((ICC_ClassFactory**)&piFactory); // get Factory
	if (FAILED(hr)) return hr;

	hr = piFactory->AssignLicense(COMPANYNAME, LICENSEKEY); // set license
	if (FAILED(hr)) return hr;

	hr = piFactory->CreateInstance(CLSID_CC_MediaReader, IID_ICC_MediaReader, (IUnknown**)&m_pMediaReader);
	if (FAILED(hr)) return hr;

#if defined(__WIN32__) || defined(_WIN32)
	CC_STRING file_name_str = _com_util::ConvertStringToBSTR(filename);
#elif defined(__APPLE__)
	CC_STRING file_name_str = const_cast<CC_STRING>(filename);
#endif

	hr = m_pMediaReader->Open(file_name_str);
	hr = E_FAIL;
	if (FAILED(hr)) return hr;

	CC_INT numAudioTracks = 0;
	hr = m_pMediaReader->get_NumberOfAudioTracks(&numAudioTracks);
	if (FAILED(hr)) return hr;

	if (numAudioTracks == 0)
	{
		m_pMediaReader = nullptr;
		return 0;
	}

	CC_UINT iCurrentAudioTrackNumber = 0;

	hr = m_pMediaReader->put_CurrentAudioTrackNumber(iCurrentAudioTrackNumber);
	if (FAILED(hr)) return hr;

	hr = m_pMediaReader->get_CurrentAudioTrackInfo((ICC_AudioStreamInfo**)&m_pAudioStreamInfo);
	if (FAILED(hr)) return hr;

	CC_TIME Duration = 0;
	hr = m_pMediaReader->get_Duration(&Duration);
	if (FAILED(hr)) return hr;

	CC_BITRATE BitRate;
	CC_UINT BitsPerSample;
	CC_UINT ChannelMask;
	CC_FRAME_RATE FrameRate;
	CC_UINT NumChannels;
	CC_UINT SampleRate;
	CC_ELEMENTARY_STREAM_TYPE StreamType;

	hr = m_pAudioStreamInfo->get_BitRate(&BitRate);
	if (FAILED(hr)) return hr;

	hr = m_pAudioStreamInfo->get_BitsPerSample(&BitsPerSample);
	if (FAILED(hr)) return hr;

	hr = m_pAudioStreamInfo->get_ChannelMask(&ChannelMask);
	if (FAILED(hr)) return hr;

	hr = m_pAudioStreamInfo->get_FrameRate(&FrameRate);
	if (FAILED(hr)) return hr;

	hr = m_pAudioStreamInfo->get_NumChannels(&NumChannels);
	if (FAILED(hr)) return hr;

	hr = m_pAudioStreamInfo->get_SampleRate(&SampleRate);
	if (FAILED(hr)) return hr;

	hr = m_pAudioStreamInfo->get_StreamType(&StreamType);
	if (FAILED(hr)) return hr;

	BitsPerSample = 16; // always play in PCM16

	size_t sample_count = (SampleRate / (m_FrameRate.num / m_FrameRate.denom));
	size_t sample_bytes = sample_count * NumChannels * (BitsPerSample >> 3);

	m_iSampleCount = sample_count;
	m_iSampleRate = SampleRate;
	m_iSampleBytes = sample_bytes;
	m_iNumChannels = NumChannels;

	audioChunk.resize(sample_bytes);

	return InitOpenAL();
}

int AudioSource::PlayFrame(size_t iFrame)
{
	if (!m_bInitialize)
		return -1;

	ALint numProcessed = 0;
	alGetSourcei(source, AL_BUFFERS_PROCESSED, &numProcessed); __al

	ALint source_state = 0;
	alGetSourcei(source, AL_SOURCE_STATE, &source_state); __al

	if (numProcessed > 0)
	{
		size_t iFrames = (iFrame + NUM_BUFFERS);

		HRESULT hr = S_OK;
		DWORD cbRetSize = 0;

		BYTE* pb = audioChunk.data();
		DWORD cb = static_cast<DWORD>(audioChunk.size());

		hr = m_pMediaReader->GetAudioSamples(CAF_PCM16, iFrames * m_iSampleCount, (CC_UINT)m_iSampleCount, pb, cb, &cbRetSize);
		if (SUCCEEDED(hr) && cbRetSize > 0)
		{
			ALvoid* data = pb;
			ALsizei size = static_cast<ALsizei>(cbRetSize);
			ALsizei frequency = static_cast<ALsizei>(m_iSampleRate);
			ALenum  format = (m_iNumChannels == 2) ? AL_FORMAT_STEREO16 : AL_FORMAT_MONO16;

			ALuint buffer = buffers[numProcessed - 1];

			alSourceUnqueueBuffers(source, 1, &buffer); __al
			alBufferData(buffer, format, data, size, frequency); __al
			alSourceQueueBuffers(source, 1, &buffer); __al
		}
	}

	if (source_state != AL_PLAYING && !m_bAudioPause)
	{
		alSourcePlay(source); __al
	}

	return 0;
}

int AudioSource::Pause(bool bPause)
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

