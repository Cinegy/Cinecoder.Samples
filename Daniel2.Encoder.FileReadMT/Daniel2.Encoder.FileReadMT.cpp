// Daniel2.Encoder.FileReadMT.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "CEncoderTest.h"

#include <cinecoder_i.c>

#include "../cinecoder_license_string.h"

int parse_args(int argc, TCHAR *argv[], TEST_PARAMS *args);
int print_help();

//---------------------------------------------------------------
int _tmain(int argc, TCHAR *argv[])
//---------------------------------------------------------------
{
    printf("Daniel2 Encoder Test App # 1.00. Copyright (c) 2018 Cinegy LLC\n\n");

    if(argc < 3)
    	return print_help();

	CC_VERSION_INFO ver = Cinecoder_GetVersion();
	printf("The Cinecoder version is %d.%02d.%02d.%d\n", ver.VersionHi, ver.VersionLo, ver.EditionNo, ver.RevisionNo);

	HRESULT hr;

	TEST_PARAMS par = {};

	par.CompanyName = COMPANYNAME;
	par.LicenseKey  = LICENSEKEY;

	if(FAILED(hr = parse_args(argc, argv, &par)))
		return hr;

	CEncoderTest Test;
	if (FAILED(hr = Test.AssignParameters(par)))
		return print_error(hr, "EncoderTest.AssignParameters error");

	LARGE_INTEGER t0, freq;
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&t0);

	ENCODER_STATS s0 = {};

	Test.Run();
	printf("\n");

	while (Test.IsActive())
	{
		if (_kbhit() && _getch() == 27)
		{
			puts("\nCancelled\n");
			Test.Cancel();
			break;
		}

		Sleep(1000);

		LARGE_INTEGER t1;
		QueryPerformanceCounter(&t1);

		ENCODER_STATS s1 = {};
		Test.GetCurrentEncodingStats(&s1);

		double dT = double(t1.QuadPart - t0.QuadPart) / freq.QuadPart;

		double Rspeed = (s1.NumBytesRead - s0.NumBytesRead) / (1024.0*1024.0*1024.0) / dT;
		double Wspeed = (s1.NumBytesWritten - s0.NumBytesWritten) / (1024.0*1024.0*1024.0) / dT;
		double Rfps = (s1.NumFramesRead - s0.NumFramesRead) / dT;
		double Wfps = (s1.NumFramesWritten - s0.NumFramesWritten) / dT;
		int queue_fill_level = s1.NumFramesRead - s1.NumFramesWritten;

		printf("\rdT = %.0f ms, R = %.3f GB/s (%.3f fps), W = %.3f GB/s (%.3f fps), Q=%d  ",
			dT * 1000,
			Rspeed, Rfps,
			Wspeed, Wfps,
			queue_fill_level);

		t0 = t1; s0 = s1;
	}

	hr = Test.GetResult();

	if (FAILED(hr))
	{
		printf("Test failed, code = %08xh\n", Test.GetResult());
	}
	else
	{
		Test.GetCurrentEncodingStats(&s0);
		printf("\nDone.\nFrames processed: %d\n", s0.NumFramesWritten);
	}
	
	return hr;
}

//---------------------------------------------------------------
int print_error(int err, const char *str)
//---------------------------------------------------------------
{
	if(str) fprintf(stderr, "%s, ", str);

	if(SUCCEEDED(err))
	{
		fprintf(stderr, "code=%08xh\n", err);
	}
	else if(LPCSTR errstr = Cinecoder_GetErrorString(err))
	{
		fprintf(stderr, "code=%08xh (%s)", err, errstr);
	}
	else
	{
		char buf[1024] = {0};
		FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM, 0, err, 0, buf, sizeof(buf), 0);
		fprintf(stderr, "code=%08xh (%s)", err, buf);
	}

	return err;
}

//---------------------------------------------------------------
int print_help()
//---------------------------------------------------------------
{
	printf(
		"This test application is intended to show the fastest encoding\n"
		"approach, where input file(s) are read directly into CUDA pinned\n"
		"memory, and then uploaded into the GPU for encoding.\n"
		"It also uses several reading threads and no reading cache to get\n"
		"the max possible reading bandwidth.\n"
		"As a side result, it is a tool for conversion from raw video files\n"
		"or DPX or BMP sequences into Cinegy's Daniel2 packed format.\n"
		"\n"
		"Usage: d2enc.exe <inputfile(s)> <output_file> <switches>\n"
		"\n"
		"Where the parameters are:\n"
		"  inputfile  - a single raw filename or a wildcard for an image sequence\n"
		"  outputfile - the output filename (can be NUL)\n"
		"\n"
		"  /fmt=WxH@F - the video format, W=width, H=height, F=fps\n"
		"  /type=t    - raw color type, can be one of {yuy2,v210}\n"
		"  /start=#   - the number of first frame to encode ([0])\n"
		"  /stop=#    - the number of last frame to encode\n"
		"  /looped    - loop the input (for perf metering)\n"
		"\n"
		"  /queue=#   - the reading queue size in frames ([8])\n"
		"  /nread=#   - the number of reading threads ([4])\n"
		"  /cache=#   - switching system reading cahe on/off ([off])\n"
		"\n"
		"  /method=#  - the encoding method ([0],2)\n"
		"  /chroma=#  - the chroma format (420,[422],rgba)\n"
		"  /bits=#    - the target bitdepth\n"
		"  /cbr=#     - the Constant Bitrate mode, the value is in Mbps\n"
		"  /cq=#      - the Constant Quality mode, the value is the quant_scale\n"
		"  /nenc=#    - the number of frame encoders working in a loop ([4])\n"
		"  /device=#  - the number of NVidia card for encoding ([0],-1=CPU)\n"
	);

	printf("\n\nPress Enter to Exit\n");
	char ch;
	scanf_s("%c", &ch);
	
	return 0;
}

//---------------------------------------------------------------
int parse_args(int argc, TCHAR *argv[], TEST_PARAMS *par)
//---------------------------------------------------------------
{
	if (!par)
		return E_UNEXPECTED;

	if (argc < 3)
		return E_INVALIDARG;

	par->InputFileName = argv[1];
	par->OutputFileName = argv[2];

	par->BitDepth = 10;
	par->ChromaFormat = CC_CHROMA_422;
	par->BitrateMode = CC_CQ;
	par->QuantScale = 16;
	par->NumSingleEncoders = 4;
	par->NumReadThreads = 4;
	par->QueueSize = 8;
	par->StopFrameNum = -1;

	for (int i = 3; i < argc; i++)
	{
		if (0 == _tcsncmp(argv[i], _T("/fmt="), 5))
		{
			int w, h; float f;

			if (_stscanf_s(argv[i] + 5, _T("%dx%d@%f"), &w, &h, &f) != 3)
				return _ftprintf(stderr, _T("incorrect /fmt switch: %s"), argv[i]), -i;

			par->Width = w;
			par->Height = h;
			par->FrameRateN = int(f + 0.5);
			par->FrameRateD = 1;

			if (par->FrameRateN != int(f))
			{
				par->FrameRateN *= 1000;
				par->FrameRateD = int(par->FrameRateN / f + 0.5);
			}
		}

		else if (0 == _tcsicmp(argv[i], _T("/type=YUY2")))
			par->InputColorFormat = CCF_YUY2;
		else if (0 == _tcsicmp(argv[i], _T("/type=UYVY")))
			par->InputColorFormat = CCF_UYVY;
		else if (0 == _tcsicmp(argv[i], _T("/type=V210")))
			par->InputColorFormat = CCF_V210;

		else if (0 == _tcsncmp(argv[i], _T("/start="), 7))
			par->StartFrameNum = _tstoi(argv[i] + 7);
		else if (0 == _tcsncmp(argv[i], _T("/stop="), 6))
			par->StopFrameNum = _tstoi(argv[i] + 6);
		else if (0 == _tcsicmp(argv[i], _T("/looped")))
			par->Looped = true;
		else if (0 == _tcsncmp(argv[i], _T("/queue="), 7))
			par->QueueSize = _tstoi(argv[i] + 7);
		else if (0 == _tcsncmp(argv[i], _T("/nread="), 7))
			par->NumReadThreads = _tstoi(argv[i] + 7);

		else if (0 == _tcsicmp(argv[i], _T("/cache=on")))
			par->UseCache = true;
		else if (0 == _tcsicmp(argv[i], _T("/cache=off")))
			par->UseCache = false;

		else if (0 == _tcsncmp(argv[i], _T("/method="), 8))
			par->CodingMethod = (CC_DANIEL2_CODING_METHOD)_tstoi(argv[i] + 8);

		else if (0 == _tcsicmp(argv[i], _T("/chroma=420")))
			par->ChromaFormat = CC_CHROMA_420;
		else if (0 == _tcsicmp(argv[i], _T("/chroma=422")))
			par->ChromaFormat = CC_CHROMA_422;
		else if (0 == _tcsicmp(argv[i], _T("/chroma=RGBA")))
			par->ChromaFormat = CC_CHROMA_RGBA;

		else if (0 == _tcsncmp(argv[i], _T("/bits="), 6))
			par->BitDepth = _tstoi(argv[i] + 6);

		else if (0 == _tcsncmp(argv[i], _T("/cbr="), 5))
		{
			par->Bitrate = _tstoi(argv[i] + 5) * 1000000ULL;
			par->BitrateMode = CC_CBR;
		}
		else if (0 == _tcsncmp(argv[i], _T("/cq="), 4))
		{
			par->QuantScale = (float)_tstof(argv[i] + 4);
			par->BitrateMode = CC_CQ;
		}

		else if (0 == _tcsncmp(argv[i], _T("/nenc="), 6))
			par->NumSingleEncoders = _tstoi(argv[i] + 6);
		else if (0 == _tcsncmp(argv[i], _T("/device="), 8))
			par->DeviceId = _tstoi(argv[i] + 8);

		else
			return _ftprintf(stderr, _T("unknown switch or incorrect switch format: %s"), argv[i]), -i;
	}

	return 0;
}
