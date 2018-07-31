#pragma once

#include <vector>

//const int SND_FREQUENCY		= 48000;
//const int SND_BITS			= 16;
//const int SND_BUFFER_SIZE	= 32 * 1024;

//const int SND_BUFFER_SIZE = 192000;

#define DSVOLUME_TO_DB(volume) ((DWORD)(-30 * (100 - volume)))

class C_SoundSample
{
public:
	C_SoundSample( __int16 *AudioData, DWORD AudioSize );
	~C_SoundSample();

	__int16 * GetData();
	DWORD GetSize();
	DWORD GetOffset();
	void SetOffset( DWORD Offset );
	DWORD GetRemainingData();
private:
	__int16	*m_AudioData;
	DWORD	m_AudioSize;
	DWORD	m_Offset;
};

typedef std::vector<std::vector<C_SoundSample*>> SoudnSampleQuee;

class C_Sound
{
public:
	C_Sound()
	{
	}
	virtual ~C_Sound()
	{
		for ( unsigned int i = 0; i < m_LockSource.size(); i++ )
		{
			ClearSamples( i );
			delete m_LockSource[ i ];
		}

		for ( unsigned int i = 0; i < m_LockAdvancedSource.size(); i++ )
		{
			ClearSamples( i, TRUE );
			delete m_LockAdvancedSource[ i ];
		}
	}

	virtual BOOL Play( DWORD Source, BOOL Advanced = FALSE ) = 0;
	virtual BOOL SetSourceVolume( DWORD Source, float db, BOOL Advanced = FALSE ) = 0;
	virtual BOOL AddNewSource( BOOL Advanced = FALSE )
	{
		if ( !Advanced )
			m_LockSource.push_back( new C_CritSec() );
		else
			m_LockAdvancedSource.push_back( new C_CritSec() );

		return TRUE;
	};
	virtual BOOL AddSampleToQueue( DWORD Source, C_SoundSample *Sample, BOOL AdvancedQueue = FALSE ) = 0;
	virtual int GetSamples( DWORD Source, BOOL AdvancedQueue = FALSE )
	{
		if ( !AdvancedQueue )
		{
			if ( Source < m_SampleQueue.size() )
				return (int)m_SampleQueue[ Source ].size();
		}
		else
		{
			if ( Source < m_AdvancedSampleQueue.size() )
				return (int)m_AdvancedSampleQueue[ Source ].size();
		}

		return 0;
	}
	virtual void ClearSamples( DWORD Source, BOOL AdvancedQueue = FALSE )
	{
		if ( !AdvancedQueue )
		{
			if ( Source < m_SampleQueue.size() )
			{
				C_AutoLock lock( m_LockSource[ Source ] );

				for ( unsigned int i = 0; i < m_SampleQueue[ Source ].size(); i++ )
					delete m_SampleQueue[ Source ][ i ];
				m_SampleQueue[ Source ].clear();
			}
		}
		else
		{
			if ( Source < m_AdvancedSampleQueue.size() )
			{
				C_AutoLock lock( m_LockAdvancedSource[ Source ] );

				for ( unsigned int i = 0; i < m_AdvancedSampleQueue[ Source ].size(); i++ )
					delete m_AdvancedSampleQueue[ Source ][ i ];
				m_AdvancedSampleQueue[ Source ].clear();
			}
		}
	}

protected:
	std::vector<C_CritSec*>	m_LockSource;
	std::vector<C_CritSec*>	m_LockAdvancedSource;

	BOOL					m_Active;
	SoudnSampleQuee			m_SampleQueue;
	SoudnSampleQuee			m_AdvancedSampleQueue;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class C_SoundEx
{
public:
	C_SoundEx()
	{
	}
	virtual ~C_SoundEx()
	{
		for (unsigned int i = 0; i < m_LockSource.size(); i++)
		{
			ClearSamples(i);
			delete m_LockSource[i];
		}
	}

	virtual BOOL Play(DWORD Source) = 0;
	virtual BOOL SetSourceVolume(DWORD Source, float db) = 0;
	virtual BOOL AddNewSource()
	{
		m_LockSource.push_back(new C_CritSec());

		return TRUE;
	};
	virtual BOOL AddSampleToQueue(DWORD Source, C_SoundSample *Sample, int iNumber) = 0;
	virtual int GetSamples(DWORD Source)
	{
		if (Source < m_SampleQueue.size())
			return (int)m_SampleQueue[Source].size();

		return 0;
	}
	virtual void ClearSamples(DWORD Source)
	{
		if (Source < m_SampleQueue.size())
		{
			C_AutoLock lock(m_LockSource[Source]);

			for (unsigned int i = 0; i < m_SampleQueue[Source].size(); i++)
				delete m_SampleQueue[Source][i];
			m_SampleQueue[Source].clear();
		}
	}

protected:
	std::vector<C_CritSec*>	m_LockSource;

	BOOL					m_Active;
	SoudnSampleQuee			m_SampleQueue;
};