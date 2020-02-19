#pragma once

#if defined(_WIN32)

#include <windows.h>

class CpuLoadMeter
{
	unsigned long long prev_IdleTime, prev_KernelTime, prev_UserTime;

public:
	CpuLoadMeter()
	{
		GetSystemTimes((FILETIME*)&prev_IdleTime, (FILETIME*)&prev_KernelTime, (FILETIME*)&prev_UserTime);
	}

	float GetLoad()
	{
        unsigned long long IdleTime, KernelTime, UserTime;
        GetSystemTimes((FILETIME*)&IdleTime, (FILETIME*)&KernelTime, (FILETIME*)&UserTime);

        auto usr = UserTime   - prev_UserTime;
        auto ker = KernelTime - prev_KernelTime;
        auto idl = IdleTime   - prev_IdleTime;
        auto sys = ker + usr;

        prev_IdleTime   = IdleTime;
        prev_KernelTime = KernelTime;
        prev_UserTime   = UserTime;

        return 100.f * (sys - idl) / sys;
	}
};

#elif defined(__linux__)

class CpuLoadMeter
{
    long long prev_busy, prev_work;

public:
    CpuLoadMeter()
    {
        ReadProcStat(&prev_busy, &prev_work);
    }

    float GetLoad()
    {
        long long busy, work;

        ReadProcStat(&busy, &work);

        float usage = work == prev_work ? 0.0 : 100.0 * (busy - prev_busy) / (work - prev_work);

        prev_work = work;
        prev_busy = busy;

        return usage;
    }

    static void ReadProcStat(long long *busy, long long *work)
    {
        long long dummy, cpu, nice, sys, idle;

        FILE *f = fopen("/proc/stat", "rt");
        fscanf(f, "cpu %lld %lld %lld %lld", &cpu, &nice, &sys, &idle);
        fclose(f);

        *busy = cpu + nice + sys;
        *work = *busy + idle;
    }
};

#else

class CpuLoadMeter
{
public:
    float GetLoad()
    {
        return -1;
    }
};

#endif
