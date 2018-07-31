#include "stdafx.h"

#include "Sound.h"

/////////////////////////////////////////////////////////////////////////////
// C_SoundSample
/////////////////////////////////////////////////////////////////////////////

C_SoundSample::C_SoundSample( __int16 *AudioData, DWORD AudioSize )
{
	if ( AudioSize > 0 )
	{
		m_AudioSize = AudioSize;
		m_AudioData = (__int16*)_aligned_malloc( m_AudioSize, 16 );
		memcpy( m_AudioData, AudioData, m_AudioSize );
		m_Offset = 0;
	}
	else
	{
		m_AudioData = NULL;
		m_AudioSize = 0;
		m_Offset = 0;
	}
}

C_SoundSample::~C_SoundSample()
{
	_aligned_free( m_AudioData );
}

/////////////////////////////////////////////////////////////////////////////

__int16 * C_SoundSample::GetData()
{
	return (__int16*)( (BYTE*)m_AudioData + m_Offset );
}

DWORD C_SoundSample::GetSize()
{
	return m_AudioSize;
}

DWORD C_SoundSample::GetOffset()
{
	return m_Offset;
}

void C_SoundSample::SetOffset( DWORD Offset )
{
	m_Offset = Offset;
}

DWORD C_SoundSample::GetRemainingData()
{
	return m_AudioSize - m_Offset;
}