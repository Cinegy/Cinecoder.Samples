#pragma once

#include <iostream>
#include <chrono>
#include <ctime>
#include <cmath>

class C_Timer
{
	typedef std::chrono::time_point<std::chrono::system_clock> timer;

private:
	timer m_start_time;
	timer m_end_time;

public:
	void StartTimer()
	{
		m_start_time = std::chrono::system_clock::now();
	}

	void StopTimer()
	{
		m_end_time = std::chrono::system_clock::now();
	}

	double GetTime() // ms
	{
		return static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(m_end_time - m_start_time).count());
	}

	double GetElapsedTime() // ms
	{
		timer end_time = std::chrono::system_clock::now();

		return static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(end_time - m_start_time).count());
	}
};
