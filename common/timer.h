#pragma once

#include <iostream>
#include <chrono>
#include <ctime>
#include <cmath>

class C_Timer
{
	typedef std::chrono::time_point<std::chrono::high_resolution_clock> timer;

	typedef std::chrono::high_resolution_clock clock;
	typedef std::chrono::duration<double, std::milli> duration;

private:
	timer m_start_time;
	timer m_end_time;

public:
	void StartTimer()
	{
		m_start_time = clock::now();
	}

	void StopTimer()
	{
		m_end_time = clock::now();
	}

	double GetTime() // ms
	{
		return (double)(((duration)(m_end_time - m_start_time)).count());

		//duration elapsed = m_end_time - m_start_time;
		//return (double)elapsed.count();
	}

	double GetElapsedTime() // ms
	{
		timer end_time = clock::now();

		return (double)(((duration)(end_time - m_start_time)).count());

		//duration elapsed = end_time - m_start_time;
		//return (double)elapsed.count();
	}
};
