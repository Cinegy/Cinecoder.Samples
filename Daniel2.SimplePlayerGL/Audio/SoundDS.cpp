#include "stdafx.h"

#include "SoundDS.h"

static int SND_BUFFER_SIZE = 32 * 1024;

/////////////////////////////////////////////////////////////////////////////
// C_SoundDS
/////////////////////////////////////////////////////////////////////////////

C_SoundDS::C_SoundDS()
{
	m_DummyHWND = CreateWindowEx( 0, L"EDIT", NULL, 0, 0, 0, 0, 0, 0, 0, 0, NULL );
	m_Active = DirectSoundCreate(NULL, (LPDIRECTSOUND*)&m_Device, NULL) == DS_OK && m_Device->SetCooperativeLevel(m_DummyHWND, DSSCL_PRIORITY) == DS_OK;

	//Log::Log( ( m_Active ? KIND_TRACE : KIND_ERROR ), 1, ( m_Active ? "DirectSound: Initialized" : "DirectSound: Failed to initialize" ) );
}

int C_SoundDS::Init(DWORD Sources)
{
	for (DWORD i = 0; i < Sources; i++)
	{
		AddNewSource();
		Play(i);
	}

	return 0;
}

C_SoundDS::~C_SoundDS()
{
	m_LastPos.clear();
	m_LastPosAdvanced.clear();

	for ( unsigned int i = 0; i < m_Buffer.size(); i++ )
		m_Buffer[ i ]->Release();
	m_Buffer.clear();

	for ( unsigned int i = 0; i < m_BufferAdvanced.size(); i++ )
		m_BufferAdvanced[ i ]->Release();
	m_BufferAdvanced.clear();

	m_Device = NULL;

	DestroyWindow( m_DummyHWND );
}

/////////////////////////////////////////////////////////////////////////////

BOOL C_SoundDS::Play( DWORD Source, BOOL Advanced )
{
	if ( !Advanced )
	{
		if ( !m_Active || Source >= m_Buffer.size() )
			return FALSE;

		m_LastPos[ Source ] = 0;

		m_Buffer[ Source ]->SetCurrentPosition( 0 );
		m_Buffer[ Source ]->Play( 0, 0, DSBPLAY_LOOPING );
		//m_Buffer[ Source ]->SetFrequency( SND_FREQUENCY );
		m_Buffer[Source]->SetFrequency(m_format.nSamplesPerSec);
		//m_Buffer[ Source ]->SetVolume( LONG(g_Settings->GetSoundDefaultVolume() * 100) ); <!>
	}
	else
	{
		if ( !m_Active || Source >= m_BufferAdvanced.size() )
			return FALSE;

		m_LastPosAdvanced[ Source ] = 0;

		m_BufferAdvanced[ Source ]->SetCurrentPosition( 0 );
		m_BufferAdvanced[ Source ]->Play( 0, 0, DSBPLAY_LOOPING );
		//m_BufferAdvanced[ Source ]->SetFrequency( SND_FREQUENCY );
		m_Buffer[Source]->SetFrequency(m_format.nSamplesPerSec);
		//m_BufferAdvanced[ Source ]->SetVolume( LONG(g_Settings->GetSoundDefaultVolume() * 100) ); <!>
	}

	return TRUE;
}

BOOL C_SoundDS::SetSourceVolume( DWORD Source, float db, BOOL Advanced )
{
	if ( !Advanced )
	{
		if ( !m_Active || Source >= m_Buffer.size() )
			return FALSE;

		m_Buffer[ Source ]->SetVolume( LONG(db * 100) );
	}
	else
	{
		if ( !m_Active || Source >= m_BufferAdvanced.size() )
			return FALSE;

		m_BufferAdvanced[ Source ]->SetVolume( LONG(db * 100) );
	}

	return TRUE;
}

BOOL C_SoundDS::CreateSoundBuffer( IDirectSoundBuffer **SoundBuffer )
{
	WAVEFORMATEX format;
	DSBUFFERDESC bufferDesc;

	memset( &format, 0, sizeof( WAVEFORMATEX ) );

	format = m_format;

	format.wFormatTag		= WAVE_FORMAT_PCM;
	//format.nChannels		= 2;
	//format.wBitsPerSample	= SND_BITS;
	//format.nSamplesPerSec	= SND_FREQUENCY;
	//format.nBlockAlign		= ( format.wBitsPerSample / 8 ) * format.nChannels;
	//format.nAvgBytesPerSec	= format.nSamplesPerSec * format.nBlockAlign;
	format.cbSize			= sizeof( WAVEFORMATEX );

	//format.nChannels		= m_format.nChannels;
	//format.wBitsPerSample	= m_format.wBitsPerSample;
	//format.nSamplesPerSec	= m_format.nSamplesPerSec;
	//format.nBlockAlign		= ( m_format.wBitsPerSample / 8 ) * m_format.nChannels;
	//format.nAvgBytesPerSec	= m_format.nSamplesPerSec * m_format.nBlockAlign;

	memset( &bufferDesc, 0, sizeof( DSBUFFERDESC ) );
	bufferDesc.dwSize			= sizeof( DSBUFFERDESC );
	bufferDesc.dwFlags			= DSBCAPS_LOCSOFTWARE | DSBCAPS_CTRLPAN | DSBCAPS_CTRLVOLUME | DSBCAPS_CTRLFREQUENCY | DSBCAPS_GETCURRENTPOSITION2 | DSBCAPS_GLOBALFOCUS;
	bufferDesc.dwBufferBytes	= SND_BUFFER_SIZE;
	bufferDesc.lpwfxFormat		= &format;

	if ( FAILED( m_Device->CreateSoundBuffer( &bufferDesc, SoundBuffer, NULL ) ) )
	{
		//Log::Error( 1, "DirectSound: Failed to create a sound buffer" );
		return FALSE;
	}

	return TRUE;
}

BOOL C_SoundDS::AddNewSource( BOOL Advanced )
{
	if ( __super::AddNewSource( Advanced ) && m_Active )
	{
		IDirectSoundBuffer *channel = NULL;
		if ( !CreateSoundBuffer( &channel ) )
			return FALSE;

		if ( !Advanced )
		{
			m_Buffer.push_back( channel );
			m_LastPos.push_back( 0 );
			m_SampleQueue.resize( m_SampleQueue.size() + 1 );
		}
		else
		{
			m_BufferAdvanced.push_back( channel );
			m_LastPosAdvanced.push_back( 0 );
			m_AdvancedSampleQueue.resize( m_AdvancedSampleQueue.size() + 1 );
		}

		return TRUE;
	}

	return FALSE;
}

// TODO: rewrite here everything
BOOL C_SoundDS::AddSampleToQueue( DWORD Source, C_SoundSample *Sample, BOOL AdvancedQueue )
{
	//static FILE *pd1 = NULL;
	//if ( !pd1 )
	//	pd1 = fopen("C:\\raw.pcm", "w+b");

	SoudnSampleQuee &sampleQueue = AdvancedQueue ? m_AdvancedSampleQueue : m_SampleQueue;

	if ( !AdvancedQueue )
	{
		if ( !m_Active || Source >= m_Buffer.size() || Source >= sampleQueue.size() )
		{
			delete Sample;
			return FALSE;
		}
	}
	else
	{
		if ( !m_Active || Source >= m_BufferAdvanced.size() || Source >= sampleQueue.size() )
		{
			delete Sample;
			return FALSE;
		}
	}

	C_AutoLock lock( AdvancedQueue ? m_LockAdvancedSource[ Source ] : m_LockSource[ Source ]  );

	DWORD status;
	IDirectSoundBuffer *buffer = AdvancedQueue ? m_BufferAdvanced[ Source ] : m_Buffer[ Source ];
	DWORD &lastPos = AdvancedQueue ? m_LastPosAdvanced[ Source ] : m_LastPos[ Source ];
	if ( buffer->GetStatus( &status ) == DS_OK )
	{
		if ( ( status & DSBSTATUS_BUFFERLOST ) && ( buffer->Restore() != DS_OK ) )
		{
			delete Sample;
			return FALSE;
		}
	}
	else
	{
		delete Sample;
		return FALSE;
	}

	if ( ( Sample->GetSize() > 0 ) || ( sampleQueue[ Source ].size() > 0 ) )
	{
		if ( Sample->GetSize() > 0 )
			sampleQueue[ Source ].push_back( Sample );
		else
			delete Sample;

		DWORD fillSize, position;
		void *block1, *block2;
		DWORD b1Size, b2Size;

		while ( buffer->GetCurrentPosition( &position, NULL ) == DSERR_BUFFERLOST )
			buffer->Restore();

		if ( lastPos <= position )
  			fillSize = position - lastPos;
		else
			fillSize = SND_BUFFER_SIZE - lastPos + position;

		block1 = NULL;
		block2 = NULL;
		b1Size = 0;
		b2Size = 0;

		C_SoundSample *sample = sampleQueue[ Source ][ 0 ];

		if ( ( fillSize > 0 ) && buffer->Lock( lastPos, fillSize, &block1, &b1Size, &block2, &b2Size, 0 ) == DS_OK )
		{
			if ( b1Size + b2Size >= sample->GetRemainingData() )
			{
				DWORD bytesRead = 0;
				DWORD bytesRead2 = 0;
				DWORD bytesLeft = b1Size;
				DWORD bytesWrite = sample->GetRemainingData();

				while ( bytesRead < b1Size )
				{
					bytesWrite = sample->GetRemainingData();
					if ( bytesWrite > bytesLeft )
						bytesWrite = bytesLeft;

					memcpy( (BYTE*)block1 + bytesRead, sample->GetData(), bytesWrite );
					bytesRead += bytesWrite;
					bytesLeft -= bytesWrite;
					sample->SetOffset( sample->GetOffset() + bytesWrite );

					if ( sample->GetRemainingData() == 0 )
					{
						delete sample;
						sample = NULL;
						sampleQueue[ Source ].erase( sampleQueue[ Source ].begin(), sampleQueue[ Source ].begin() + 1 );
					}

					if ( sampleQueue[ Source ].size() <= 0 || bytesLeft <= 0 )
						break;
					else
						sample = sampleQueue[ Source ][ 0 ];
				}

				if ( sample )
				{
					bytesRead2 = 0;
					bytesLeft = b2Size;
					bytesWrite = sample->GetRemainingData();

					while ( bytesRead2 < b2Size )
					{
						bytesWrite = sample->GetRemainingData();
						if ( bytesWrite > bytesLeft )
							bytesWrite = bytesLeft;

						memcpy( (BYTE*)block2 + bytesRead2, sample->GetData(), bytesWrite );
						bytesRead2 += bytesWrite;
						bytesLeft -= bytesWrite;
						sample->SetOffset( sample->GetOffset() + bytesWrite );

						if ( sample->GetRemainingData() == 0 )
						{
							delete sample;
							sampleQueue[ Source ].erase( sampleQueue[ Source ].begin(), sampleQueue[ Source ].begin() + 1 );
						}

						if ( sampleQueue[ Source ].size() <= 0 || bytesLeft <= 0 )
							break;
						else
							sample = sampleQueue[ Source ][ 0 ];
					}
				}

				buffer->Unlock( block1, bytesRead, block2, bytesRead2 );

				//fwrite((void *)(block1), sizeof(BYTE), bytesRead, pd1);
				//fwrite((void *)(block2), sizeof(BYTE), bytesRead2, pd1);

				lastPos = ( lastPos + bytesRead + bytesRead2 ) % SND_BUFFER_SIZE;
			}
			else
			{
				memcpy( block1, sample->GetData(), b1Size );
				sample->SetOffset( sample->GetOffset() + b1Size );
				if ( b2Size > 0 )
				{
					memcpy( block2, sample->GetData(), b2Size );
					sample->SetOffset( sample->GetOffset() + b2Size );
				}

				buffer->Unlock( block1, b1Size, block2, b2Size );

				//fwrite((void *)(block1), sizeof(BYTE), b1Size, pd1);
				//fwrite((void *)(block2), sizeof(BYTE), b2Size, pd1);

				lastPos = ( lastPos + b1Size + b2Size ) % SND_BUFFER_SIZE;
			}
		}
	}
	else
	{
		delete Sample;
		return FALSE;
	}

	if ( !AdvancedQueue && ( sampleQueue[ Source ].size() > 4 ) )
	{
		DWORD queueLength = 0;
		for ( unsigned int i = 0; i < sampleQueue[ Source ].size() - 1; i++ )
		{
			queueLength += sampleQueue[ Source ][ i ]->GetRemainingData();
			delete sampleQueue[ Source ][ i ];
		}
		sampleQueue[ Source ].erase( sampleQueue[ Source ].begin(), sampleQueue[ Source ].begin() + sampleQueue[ Source ].size() - 1 );

		//Log::Trace( 3, "DirectSound: Source #%i has too big queue, skip ~%i ms...", Source, queueLength * 1000 / ( 48000 * 4 ) );
	}

	return TRUE;
}

void C_SoundDS::ClearSamples( DWORD Source, BOOL Advanced )
{
	if ( !Advanced )
	{
		if ( Source < m_SampleQueue.size() )
		{
			__super::ClearSamples( Source, Advanced );

			C_AutoLock lock( m_LockSource[ Source ] );

			m_LastPos[ Source ] = 0;

			LPVOID block1, block2;
			DWORD b1Size, b2Size;
			m_Buffer[ Source ]->Stop();
			m_Buffer[ Source ]->Lock( m_LastPos[ Source ], SND_BUFFER_SIZE, &block1, &b1Size, &block2, &b2Size, 0 );
			memset( block1, 0, b1Size );
			if ( b2Size > 0 )
				memset( block2, 0, b2Size );
			m_Buffer[ Source ]->Unlock( block1, b1Size, block2, b2Size );
			m_Buffer[ Source ]->SetCurrentPosition( 0 );
			m_Buffer[ Source ]->Play( 0, 0, DSBPLAY_LOOPING );
		}
	}
	else
	{
		if ( Source < m_AdvancedSampleQueue.size() )
		{
			__super::ClearSamples( Source, Advanced );

			C_AutoLock lock( m_LockAdvancedSource[ Source ] );

			m_LastPosAdvanced[ Source ] = 0;

			LPVOID block1, block2;
			DWORD b1Size, b2Size;
			m_BufferAdvanced[ Source ]->Stop();
			m_BufferAdvanced[ Source ]->Lock( m_LastPosAdvanced[ Source ], SND_BUFFER_SIZE, &block1, &b1Size, &block2, &b2Size, 0 );
			memset( block1, 0, b1Size );
			if ( b2Size > 0 )
				memset( block2, 0, b2Size );
			m_BufferAdvanced[ Source ]->Unlock( block1, b1Size, block2, b2Size );
			m_BufferAdvanced[ Source ]->SetCurrentPosition( 0 );
			m_BufferAdvanced[ Source ]->Play( 0, 0, DSBPLAY_LOOPING );
		}
	}
}