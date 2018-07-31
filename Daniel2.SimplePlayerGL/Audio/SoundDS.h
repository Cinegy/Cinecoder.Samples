#pragma once

#include <vector>

// Cinegy utils
#include "../utils/comptr.h"

#include "Sound.h"

class C_SoundDS : public C_Sound
{
public:
	//C_SoundDS( DWORD Sources );
	C_SoundDS();
	~C_SoundDS();

	int Init(DWORD Sources);

	BOOL Play( DWORD Source, BOOL Advanced = FALSE );
	BOOL SetSourceVolume( DWORD Source, float db, BOOL Advanced = FALSE );
	BOOL AddNewSource( BOOL Advanced = FALSE );
	BOOL AddSampleToQueue( DWORD Source, C_SoundSample *Sample, BOOL AdvancedQueue = FALSE );
	void ClearSamples( DWORD Source, BOOL Advanced = FALSE );

	void SetWAVEFORMATEX(WAVEFORMATEX format)
	{
		m_format = format;
	}

private:
	HWND								m_DummyHWND;
	com_ptr<IDirectSound>				m_Device;
	std::vector<IDirectSoundBuffer*>	m_Buffer;
	std::vector<DWORD>					m_LastPos;
	std::vector<IDirectSoundBuffer*>	m_BufferAdvanced;
	std::vector<DWORD>					m_LastPosAdvanced;

	WAVEFORMATEX						m_format;

	BOOL CreateSoundBuffer( IDirectSoundBuffer **SoundBuffer );
};