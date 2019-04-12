// Daniel2.Encoder.FileReadMT.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "CEncoderTest.h"

#include <Cinecoder_i.c>
#include <Cinecoder.Plugin.Multiplexers_i.c>

#include "../common/cinecoder_license_string.h"
#include "../common/cinecoder_error_handler.h"

#include <stdio.h>

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

	auto t0 = system_clock::now();

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

		std::this_thread::sleep_for(500ms);

		auto t1 = system_clock::now();

		ENCODER_STATS s1 = {};
		Test.GetCurrentEncodingStats(&s1);

		double dT = duration<double>(t1 - t0).count();

		double Rspeed = (s1.NumBytesRead - s0.NumBytesRead) / (1024.0*1024.0*1024.0) / dT;
		double Wspeed = (s1.NumBytesWritten - s0.NumBytesWritten) / (1024.0*1024.0*1024.0) / dT;
		double Rfps = (s1.NumFramesRead - s0.NumFramesRead) / dT;
		double Wfps = (s1.NumFramesWritten - s0.NumFramesWritten) / dT;
		int queue_fill_level = s1.NumFramesRead - s1.NumFramesWritten;

		fprintf(stderr, "\rframe # %d, Q=%d [%-*.*s], R = %.3f GB/s (%.3f fps), W = %.3f GB/s (%.3f fps) ",
			s1.NumFramesRead,
			queue_fill_level,
			par.QueueSize, queue_fill_level,
			"################",
			Rspeed, Rfps,
			Wspeed, Wfps
		);

		t0 = t1; s0 = s1;
	}

	hr = Test.GetResult();

	if (FAILED(hr))
	{
		return print_error(hr, "Test failed");
	}
	else
	{
		Test.GetCurrentEncodingStats(&s0);
		printf("\nDone.\nFrames processed: %d\n", s0.NumFramesWritten);
	}

	Test.Close();

	return hr;
}

//---------------------------------------------------------------
int print_error(int err, const char *str)
//---------------------------------------------------------------
{
	fprintf(stderr, "\nError: %s%s", str?str:"", str?", ":"");

	if(SUCCEEDED(err))
	{
		fprintf(stderr, "code=%08xh\n", err);
	}
	else if(const LPCSTR errstr = Cinecoder_GetErrorString(err))
	{
		fprintf(stderr, "code=%08xh (%s)\n", err, errstr);
	}
	else
	{
#ifdef _WIN32
		char buf[1024] = {0};
		FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM, 0, err, 0, buf, sizeof(buf), 0);
		fprintf(stderr, "code=%08xh (%s)\n", err, buf);
#else
		fprintf(stderr, "code=%08xh (%s)\n", err, strerror(err & ~0x80000000u));
#endif
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
		"Usage: Daniel2.DPXEncoder.exe <inputfile(s)> <output_file> <switches>\n"
		"\n"
		"Where the parameters are:\n"
		"  inputfile  - a single raw filename or a wildcard for an image sequence\n"
		"  outputfile - the output filename (.MXF or .DN2)\n"
		"\n"
		"  /raw=t:WxH - the color type, can be one of {yuy2,uyvy,v210,rgb30}, W=width, H=height\n"
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
		"  /chroma=#  - the chroma format (420,[422],444,4444,rgb,rgba)\n"
		"  /bits=#    - the target bitdepth\n"
		"  /cbr=#     - the Constant Bitrate mode, the value is in Mbps\n"
		"  /cq=#      - the Constant Quality mode, the value is the quant_scale\n"
		"  /nenc=#    - the number of frame encoders working in a loop ([4])\n"
		"  /device=#  - the number of NVidia card for encoding ([0],-1=CPU)\n"
		"\n"
		"Sample usage (raw file):\n"
		"> Daniel2.DPXEncoder.exe RawSample_2160.yuy2 test.DN2 /raw=V210:3840x2160 /bits=10 /fps=60 /method=0 /chroma=422 /cbr=600\n"
		"Set of DPX files:\n"
		"> Daniel2.DPXEncoder.exe Animation_#####.dpx test.MXF /start=0 /stop=9999 /bits=12 /fps=60 /method=2 /chroma=RGBA /cq=16 /nread=8\n"
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
	//par->ChromaFormat = CC_CHROMA_422;
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
			int wxh_offs = 0;
				 if (0 == _tcsnicmp(argv[i]+5, _T("YUY2:"),  5)) { par->InputColorFormat = CCF_YUY2;  wxh_offs = 10; }
			else if (0 == _tcsnicmp(argv[i]+5, _T("UYVY:"),  5)) { par->InputColorFormat = CCF_UYVY;  wxh_offs = 10; }
			else if (0 == _tcsnicmp(argv[i]+5, _T("V210:"),  5)) { par->InputColorFormat = CCF_V210;  wxh_offs = 10; }
			else if (0 == _tcsnicmp(argv[i]+5, _T("RGB30:"), 6)) { par->InputColorFormat = CCF_RGB30; wxh_offs = 11; }
			else return _ftprintf(stderr, _T("Unknown /raw color format: %s"), argv[i]+5), -i;

			if (_stscanf_s(argv[i] + wxh_offs, _T("%dx%d"), &par->Width, &par->Height) != 2)
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
		else if (0 == _tcsicmp(argv[i], _T("/chroma=444")))
			par->ChromaFormat = CC_CHROMA_444;
		else if (0 == _tcsicmp(argv[i], _T("/chroma=4444")))
			par->ChromaFormat = CC_CHROMA_4444;
		else if (0 == _tcsicmp(argv[i], _T("/chroma=RGB")))
			par->ChromaFormat = CC_CHROMA_RGB;
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

constexpr uint32_t fourcc( char const p[5] )
{
  return (p[0] << 24) | (p[1] << 16) | (p[2] << 8) | p[3];
}

//---------------------------------------------------------------
int check_for_dpx(TEST_PARAMS *par)
//---------------------------------------------------------------
{
    const TCHAR *ext = _tcsrchr(par->InputFileName, '.');

	if(!ext || _tcsicmp(ext, _T(".DPX")) != 0)
		return S_FALSE;

	static TCHAR dpx_filemask[MAX_PATH] = {}; // hack! par->InputFileName can point to this buffer
	ConvertFileMask(par->InputFileName, dpx_filemask);

	TCHAR dpx_filename[MAX_PATH] = {};
	_stprintf(dpx_filename, dpx_filemask, par->StartFrameNum);

    FILE *hFile = _tfopen(dpx_filename, _T("rb"));
	if(!hFile)
		return _ftprintf(stderr, _T("Can't open '%s'"), dpx_filename), -1;

    dpx_file_header_t dpx_hdr;
    if(fread(&dpx_hdr, 1, sizeof(dpx_hdr), hFile) != sizeof(dpx_hdr))
    	return _ftprintf(stderr, _T("Can't read DPX header from '%s'"), dpx_filename), -2;

    if(dpx_hdr.file.magic_num != fourcc("XPDS") && dpx_hdr.file.magic_num != fourcc("SDPX"))
    	return _ftprintf(stderr, _T("Wrong MAGIC_NUMBER")), E_UNEXPECTED;

	bool BE = dpx_hdr.file.magic_num == fourcc("XPDS");

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
	fseek(hFile, 0, SEEK_END);
	int dpx_size = ftell(hFile);// SWAP4(BE, dpx_hdr.file.file_size);
	int dpx_offset = SWAP4(BE,dpx_hdr.file.data_offset);
	int dpx_padding = SWAP4(BE, dpx_hdr.image.channel[0].line_padding);

	switch (dpx_hdr.image.channel[0].designator)
	{
	case 50: // RGB
		switch (bits)
		{
		case 10: 
			par->InputColorFormat = BE ? CCF_RGB30X2_BE : CCF_RGB30X2;
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
	//par->ChromaFormat = CC_CHROMA_RGBA;

    fclose(hFile);

	return S_OK;
}
