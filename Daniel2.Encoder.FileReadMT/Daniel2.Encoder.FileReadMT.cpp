// Daniel2.Encoder.FileReadMT.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "CEncoderTest.h"

#include <Cinecoder_i.c>

#include "../cinecoder_license_string.h"
#include "../cinecoder_error_handler.h"

int parse_args(int argc, TCHAR *argv[], TEST_PARAMS *encpar);
int print_help();
int check_for_dpx(TEST_PARAMS *encpar);

//---------------------------------------------------------------
int _tmain(int argc, TCHAR *argv[])
//---------------------------------------------------------------
{
    printf("Daniel2 Encoder Test App # 1.00. Copyright (c) 2018 Cinegy LLC\n\n");

    if(argc < 3)
    	return print_help();

	Cinecoder_SetErrorHandler(&g_ErrorHandler);

	CC_VERSION_INFO ver = Cinecoder_GetVersion();
	printf("The Cinecoder version is %d.%02d.%02d.%d\n", ver.VersionHi, ver.VersionLo, ver.EditionNo, ver.RevisionNo);

	HRESULT hr;

	TEST_PARAMS par = {};

	par.CompanyName = COMPANYNAME;
	par.LicenseKey  = LICENSEKEY;

	if(FAILED(hr = parse_args(argc, argv, &par)))
		return hr;

	if(FAILED(hr = check_for_dpx(&par)))
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

		Sleep(500);

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

		printf("\rdT = %.0f ms, R = %.3f GB/s (%.3f fps), W = %.3f GB/s (%.3f fps), Q=%d [%-*.*s] ",
			dT * 1000,
			Rspeed, Rfps,
			Wspeed, Wfps,
			queue_fill_level,
			par.QueueSize, queue_fill_level,
			"################"
		);

		t0 = t1; s0 = s1;
	}

	hr = Test.GetResult();

	if (FAILED(hr))
	{
		printf("Test failed, code = %08lxh\n", Test.GetResult());
	}
	else
	{
		Test.GetCurrentEncodingStats(&s0);
		printf("\nDone.\nFrames processed: %ld\n", s0.NumFramesWritten);
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
	else if(const LPCSTR errstr = Cinecoder_GetErrorString(err))
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
		"  /raw=t:WxH - the color type, can be one of {yuy2,v210}, W=width, H=height\n"
		"  /fps=#     - the frame rate (i.e. 25, 29.97, [60], etc)\n"
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
		"\n"
		"Sample usage (raw dile):\n"
		"> Daniel2.DPXEncoder.exe old_town_cross_2160p50.yuy2 test.dn2 /raw=V210:3840x2160 /bits=10 /fps=60 /method=0 /chroma=422 /cbr=600\n"
		"Set of DPX files:"
		"> Daniel2.DPXEncoder.exe Animation_#####.dpx test.dn2 /start=0 /stop=9999 /bits=12 /fps=60 /method=2 /chroma=RGBA /cq=16 /nread=8\n"
	);

	printf("\n\nPress Enter to Exit\n");
	std::cin.get();
	
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

	par->FrameRateN = 60;
	par->FrameRateD = 1;
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
		if (0 == _tcsnicmp(argv[i], _T("/raw="), 5))
		{
				 if (0 == _tcsnicmp(argv[i]+5, _T("YUY2:"), 5))	par->InputColorFormat = CCF_YUY2;
			else if (0 == _tcsnicmp(argv[i]+5, _T("UYVY:"), 5))	par->InputColorFormat = CCF_UYVY;
			else if (0 == _tcsnicmp(argv[i]+5, _T("V210:"), 5))	par->InputColorFormat = CCF_V210;
			else return _ftprintf(stderr, _T("Unknown /raw color format: %4.4s"), argv[i]+5), -i;

			if (_stscanf_s(argv[i] + 10, _T("%dx%d"), &par->Width, &par->Height) != 2)
				return _ftprintf(stderr, _T("incorrect /raw frame size specification: %s"), argv[i]+10), -i;
		}
		else if (0 == _tcsncmp(argv[i], _T("/fps="), 5))
		{
			double f = _tstof(argv[i] + 5);

			par->FrameRateN = int(f + 0.5);
			par->FrameRateD = 1;

			if (par->FrameRateN != int(f))
			{
				par->FrameRateN *= 1000;
				par->FrameRateD = int(par->FrameRateN / f + 0.5);
			}
		}

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

//---------------------------------------------------------------
void ConvertFileMask(const TCHAR *mask, TCHAR *buf)
//---------------------------------------------------------------
{
	while(*mask && *mask != '#')
		*buf++ = *mask++;
	
	*buf = 0;

	if(!*mask) return;

	int len = 0;
	
	while(*mask && *mask == '#')
		len++, mask++;
	
	_stprintf(buf, _T("%%0%dd%s"), len, mask);
}

#include "dpx_file.h"

//---------------------------------------------------------------
int check_for_dpx(TEST_PARAMS *par)
//---------------------------------------------------------------
{
    TCHAR ext[MAX_PATH] = {};
	_tsplitpath(par->InputFileName, NULL, NULL, NULL, ext);

	bool is_dpx= _tcsicmp(ext, _T(".DPX")) == 0;
	if(!is_dpx)
		return S_FALSE;

	static TCHAR dpx_filemask[MAX_PATH] = {}; // hack! par->InputFileName can point to this buffer
	ConvertFileMask(par->InputFileName, dpx_filemask);

	TCHAR dpx_filename[MAX_PATH] = {};
	_stprintf(dpx_filename, dpx_filemask, par->StartFrameNum);

    HANDLE hFile = CreateFile(dpx_filename, GENERIC_READ, FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, 0, NULL);
	if(hFile == INVALID_HANDLE_VALUE)
		return _ftprintf(stderr, _T("Can't open '%s'"), dpx_filename), HRESULT_FROM_WIN32(GetLastError());

    dpx_file_header_t dpx_hdr; DWORD r;
    if(!ReadFile(hFile, &dpx_hdr, sizeof(dpx_hdr), &r, NULL))
    	return _ftprintf(stderr, _T("Can't read DPX header from '%s'"), dpx_filename), HRESULT_FROM_WIN32(GetLastError());

    if(dpx_hdr.file.magic_num != 'XPDS' && dpx_hdr.file.magic_num != 'SDPX')
    	return _ftprintf(stderr, _T("Wrong MAGIC_NUMBER")), E_UNEXPECTED;

	bool BE = dpx_hdr.file.magic_num == 'XPDS';

	if(dpx_hdr.file.encryption_key != 0xFFFFFFFF)
		return _ftprintf(stderr, _T("DPX: encryped, key=%08x. Unsupported."), SWAP4(BE, dpx_hdr.file.encryption_key)), E_UNEXPECTED;

	if(SWAP2(BE,dpx_hdr.image.channels_per_image) != 1)
		return _ftprintf(stderr, _T("DPX: only 1 channel per image is supported")), E_UNEXPECTED;

	if(SWAP2(BE,dpx_hdr.image.channel[0].packing) > 1)
		return _ftprintf(stderr, _T("DPX: unsupported packing %d"), SWAP2(BE, dpx_hdr.image.channel[0].packing)), E_UNEXPECTED;

	if (dpx_hdr.image.channel[0].encoding != 0)
		return _ftprintf(stderr, _T("DPX: unsupported encoding %d"), dpx_hdr.image.channel[0].encoding), E_UNEXPECTED;

	int bits = dpx_hdr.image.channel[0].bits_per_pixel;
	if (bits != 10 && bits != 12 && bits != 16)
		return _ftprintf(stderr, _T("DPX: unsupported bit_depth %d"), bits), E_UNEXPECTED;

	int dpx_w = SWAP4(BE,dpx_hdr.image.pixels_per_line);
	int dpx_h = SWAP4(BE,dpx_hdr.image.lines_per_image);
	int dpx_size = GetFileSize(hFile, NULL);// SWAP4(BE, dpx_hdr.file.file_size);
	int dpx_offset = SWAP4(BE,dpx_hdr.file.data_offset);
	int dpx_padding = SWAP4(BE, dpx_hdr.image.channel[0].line_padding);

	switch (dpx_hdr.image.channel[0].designator)
	{
	case 50: // RGB
		switch (bits)
		{
		case 10: 
			par->InputColorFormat = BE ? CCF_A2RGB30_BE : CCF_A2RGB30;
			par->InputPitch = dpx_w * 32 / 8 + dpx_padding;
			break;
		case 12:
			par->InputColorFormat = BE ? CCF_RGB36_BE : CCF_RGB36;
			par->InputPitch = dpx_w * 36 / 8 + dpx_padding;
			break;
		case 16:
			par->InputColorFormat = BE ? CCF_RGB48_BE : CCF_RGB48;
			par->InputPitch = dpx_w * 48 / 8 + dpx_padding;
			break;
		}
		break;

	default:
		return _ftprintf(stderr, _T("DPX: non-RGB data (designator=%d)"), dpx_hdr.image.channel[0].designator), E_UNEXPECTED;
	}

	switch (dpx_hdr.image.orientation)
	{
	case 0: par->PictureOrientation = CC_PO_DEFAULT; break;
	case 1: par->PictureOrientation = CC_PO_FLIP_HORIZONTAL; break;
	case 2: par->PictureOrientation = CC_PO_FLIP_VERTICAL; break;
	case 3: par->PictureOrientation = CC_PO_ROTATED_180DEG; break;
	default: return _ftprintf(stderr, _T("DPX: unsupported picture_orientation %d"), dpx_hdr.image.orientation), E_UNEXPECTED;
	}

	par->InputFileName = dpx_filemask;
	par->Width = dpx_w;
	par->Height = dpx_h;
	par->SetOfFiles = TRUE;
	par->FileSize = dpx_size;
	par->DataOffset = dpx_offset;
	par->ChromaFormat = CC_CHROMA_RGBA;

	return S_OK;
}
