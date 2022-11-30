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

        return 100.0f * (sys - idl) / sys;
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

		if (busy == -1 && work == -1) 
			return -1;

        float usage = work == prev_work ? 0.0 : 100.0 * (busy - prev_busy) / (work - prev_work);

        prev_work = work;
        prev_busy = busy;

        return usage;
    }

    static void ReadProcStat(long long *busy, long long *work)
    {
		if (FILE *f = fopen("/proc/stat", "rt"))
		{
	        long long dummy, cpu, nice, sys, idle;

			[&]() { return fscanf(f, "cpu %lld %lld %lld %lld", &cpu, &nice, &sys, &idle); }; // lambda for fix "warning: ignoring return value of ‘int fscanf(.."
			fclose(f);

			*busy = cpu + nice + sys;
			*work = *busy + idle;
		}
		else
		{
			*busy = *work = -1;
		}
    }
};

#elif defined(__APPLE__)

#include <mach/mach_init.h>
#include <mach/mach_error.h>
#include <mach/mach_host.h>

class CpuLoadMeter
{
	unsigned long long prev_total_ticks = 0;
	unsigned long long prev_idle_ticks = 0;
	float              prev_load = 0;

public:
    float GetLoad()
    {
	    host_cpu_load_info_data_t cpuinfo;

	    mach_msg_type_number_t count = HOST_CPU_LOAD_INFO_COUNT;

	    if (KERN_SUCCESS != host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, (host_info_t)&cpuinfo, &count))
	    	return -1;

	    unsigned long long curr_idle_ticks  = cpuinfo.cpu_ticks[CPU_STATE_IDLE];
	    unsigned long long curr_total_ticks = 0;
	        
	    for(int i = 0; i < CPU_STATE_MAX; i++)
	        curr_total_ticks += cpuinfo.cpu_ticks[i];

	    unsigned long long total_ticks = curr_total_ticks - prev_total_ticks;
	    unsigned long long idle_ticks  = curr_idle_ticks  - prev_idle_ticks;

	    if(total_ticks == 0)
	    	return prev_load;

	    float curr_load = (1 - float(idle_ticks) / total_ticks) * 100;

	    prev_total_ticks = curr_total_ticks;
	    prev_idle_ticks  = curr_idle_ticks;
	    prev_load        = curr_load;

	    return curr_load;
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
