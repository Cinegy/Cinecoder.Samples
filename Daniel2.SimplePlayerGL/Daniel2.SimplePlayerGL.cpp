// SimpleDecodeDN2.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

bool g_bPause = false;
bool g_bVSync = false;
bool g_bRotate = true;
bool g_bLastRotate = g_bRotate;
bool g_useCuda = false;
bool g_useQuickSync = false;
bool g_useAMF = false;
bool g_useNVDEC = false;
bool g_useOpenCL = false;
bool g_useDirectX11 = false;
bool g_useModernOGL = false;
bool g_useCinecoderD3D11 = false;

bool g_bMaxFPS = false;
bool g_bVSyncHand = true;
bool g_bShowTexture = true;

bool g_bFramebuffer = false;

#if defined(__USE_GLUT_RENDER__)
#include "SimplePlayerGL.h"
#endif

///////////////////////////////////////////////////////

inline const char * getRefStrArgShift(const char *argv, char ch)
{
	const char *str_argv = argv;

	size_t shift = 0;

	while (str_argv[shift] == ch) shift++;

	if (shift < strlen(str_argv))
		str_argv += shift;

	return str_argv;
}

inline bool checkCmdLineArg(const int argc, const char **argv, const char *str_ref)
{
	bool bFound = false;

	for (int i = 1; i < argc; i++)
	{
		if (strcmp(getRefStrArgShift(argv[i], '-'), str_ref) == 0)
		{
			bFound = true;
			break;
		}
	}

	return bFound;
}

inline bool getCmdLineArgStr(const int argc, const char **argv, const char *str_ref, char **str_ret)
{
	bool bFound = false;

	for (int i = 1; i < argc; i++)
	{
		if (strcmp(getRefStrArgShift(argv[i], '-'), str_ref) == 0)
		{
			if ((i + 1) < argc)
			{
				*str_ret = (char *)argv[i + 1];
				bFound = true;
			}
			break;
		}
	}

	if (!bFound)
		*str_ret = nullptr;

	return bFound;
}

///////////////////////////////////////////////////////////////////////////
// Print help screen
///////////////////////////////////////////////////////////////////////////

void printHelp(void)
{
	printf("Usage: Daniel2.SimplePlayerGL [OPTION]...\n");
	printf("Test the decode DANIEL2 format file (DN2) using OpenGL(GLUT)\nNow with added Cinecoder power to decode MXFs with various other codecs (because we can!)");
	printf("\n");
	printf("Command line example: <Daniel2.SimplePlayerGL.exe> <file_path> -decoders 2 -d3d11 nvidia -vsync -rotate_frame -output_format RGBA32\n");
	printf("\n");
	printf("Options:\n");
	printf("-help               display this help menu\n");
	printf("-decoders <N>       max count of decoders [1..4] (default: 2)\n");
	//printf("-vsync              enable vertical synchronisation (default - disable)\n");
	printf("-fpsmax             enable maximum playing fps (default: disable)\n");
	printf("-rotate_frame       enable rotate frame (default: disable)\n");
#ifdef USE_CUDA_SDK
	printf("-cuda               enable CUDA decoding (default: disable)\n");
#endif
#if defined(__USE_GLUT_RENDER__)
#ifdef USE_OPENCL_SDK
	printf("-opencl             enable OpenCL decoding (default: disable)\n");
#endif
#endif
	printf("-quicksync          enable QuickSync H264/HEVC GPU decoding (default: disable)\n");
	printf("-amf                enable AMF H264/HEVC GPU decoding (default: disable)\n");
	printf("-nvdec              enable NVIDIA H264/HEVC GPU decoding (default: disable)\n");
#if defined(__WIN32__)
	printf("-d3d11 [adapter]   enable DirectX11 pipeline (default: OpenGL)\n");
	printf("    any:            Any Graphics Adapter (without cuda)\n");
	printf("    intel:          IntelHD Graphics Adapter (without cuda)\n");
	printf("    nvidia:         NVIDIA Adapter (set by default)\n");
	printf("-cinecoderD3D11    enable Cinecode+DirectX11 pipeline\n");
#endif
#if !defined(__WIN32__)
	printf("-ogl33              enable modern OpenGL 3.3 (default use OpenGL 1.1)\n");
	printf("-nogl               start without OpenGL (default use OpenGL)\n");
#endif
#if defined(__LINUX__)
	printf("-framebuffer        start without OpenGL and use framebuffer mode (work in Linux text console)\n");
#endif
	printf("-output_format      output texture format (default: RGBA32 for 8 bit and RGBA64 for more 8 bit)\n");
	printf("-scale <N>          scale output buffer 1(x2) 2(x4) 3(x8) (only without CUDA, default: 0)\n");
	printf("\nCommands:\n");
	printf("'ESC':              exit\n");
	printf("'p' or 'SPACE':     on/off pause\n");
	printf("'v':                on/off vertical synchronisation\n");
	printf("'m':                on/off maximum playing fps\n");
	printf("'r':                on/off rotate image\n");
	printf("'f':                on/off fullscreen mode\n");
	printf("'t':                on/off copy result to texture\n");
	printf("'o':                on/off show texture\n");
	printf("'d':                on/off decoder\n");
	printf("'n':                on/off read file\n");
	printf("'+'/'-':            change audio volume (+/- 10%%)\n");
	printf("'J'/'K'/'L':        change direction video or pause\n");
	printf("'right'/'left':     show next/prev (+/- 1 frame)\n");
	printf("'HOME':             seek to first frame\n");
	printf("'END':              seek to last frame\n");

	printf("\n\nPress Enter to Exit\n");
	std::cin.get();
}

///////////////////////////////////////////////////////////////////////////

// GPU decoding pipeline on NVIDIA: -cuda -cinecoderD3D11 -d3d11

int main(int argc, char **argv)
{
	// Process command line args
	if (checkCmdLineArg(argc, (const char **)argv, "help") ||
		checkCmdLineArg(argc, (const char **)argv, "h") ||
		argc == 1)
	{
		printHelp(); // Print help info
		return 0;
	}

	std::string filename;

	size_t iMaxCountDecoders = 2;
	size_t iScale = 0;

	char *str = nullptr;

	filename = argv[1];

	if (getCmdLineArgStr(argc, (const char **)argv, "decoders", &str))
	{
		iMaxCountDecoders = atoi(str); // max count of decoders [1..4] (default: 2)
	}

	if (getCmdLineArgStr(argc, (const char **)argv, "scale", &str))
	{
		iScale = atoi(str);
	}

	if (checkCmdLineArg(argc, (const char **)argv, "vsync"))
	{
		g_bVSync = true; // on/off vertical synchronisation
	}

	if (checkCmdLineArg(argc, (const char **)argv, "fpsmax"))
	{
		g_bMaxFPS = true; // on/off maximum playing fps
	}

	if (checkCmdLineArg(argc, (const char **)argv, "rotate_frame"))
	{
		g_bRotate = true; // on/off rotate image
	}

#ifdef USE_CUDA_SDK
	if (checkCmdLineArg(argc, (const char **)argv, "cuda"))
	{
		g_useCuda = true; // use CUDA decoder rather than CPU decoder
	}
#ifdef CUDA_WRAPPER
	if (g_useCuda)
	{
		if (__InitCUDA() != 0) // init CUDA SDK
		{
			printf("Error: cannot initialize CUDA! Please check if the <cudart> file exists!\n");
			return 0;
		}
	}
#endif
#endif

#ifdef USE_OPENCL_SDK
	if (checkCmdLineArg(argc, (const char **)argv, "opencl"))
	{
		g_useOpenCL = true; // use OpenCL decoder rather than CPU decoder
	}

	if (g_useCuda && g_useOpenCL)
	{
		g_useOpenCL = false;
		printf("Info: CUDA and OpenCL are not uses at the same time! (parameter <opencl> was ignored)\n");
	}

#ifdef OPENCL_WRAPPER
	if (g_useOpenCL)
	{
		if (LoadOpenClLib() != 0) // init OpenCL SDK
		{
			printf("Error: cannot initialize OpenCL! Please check if the <OpenCL.dll/libOpenCL.so> file exists!\n");
			return 0;
		}
	}
#endif
#endif

	if (checkCmdLineArg(argc, (const char **)argv, "quicksync"))
	{
		g_useQuickSync = true; // use QuickSync decoder
	}

	if (checkCmdLineArg(argc, (const char**)argv, "amf"))
	{
		g_useAMF = true; // use AMF decoder
	}	

	if (checkCmdLineArg(argc, (const char**)argv, "nvdec"))
	{
		g_useNVDEC = true; // use AMF decoder
	}

	size_t gpuDevice = 1;
#if defined(__WIN32__)
	if (checkCmdLineArg(argc, (const char **)argv, "d3d11"))
	{
		g_useDirectX11 = true; // use DirectX11 pipeline

		if (getCmdLineArgStr(argc, (const char **)argv, "d3d11", &str))
		{
			if (strcmp(str, "nvidia") == 0)
			{
				gpuDevice = 1;
			}
			else if (strcmp(str, "intel") == 0)
			{
				if (!g_useCuda)
					gpuDevice = 2;
				else
					printf("ignored: option \"-d3d11 intel\" was ignored because set param \"-cuda\"\n");
			}
			else if (strcmp(str, "any") == 0)
			{
				if (!g_useCuda)
					gpuDevice = 0;
				else
					printf("ignored: option \"-d3d11 any\" was ignored because set param \"-cuda\"\n");
			}
		}
	}

	if (checkCmdLineArg(argc, (const char **)argv, "cinecoderD3D11"))
	{
		g_useCinecoderD3D11 = true;
	}
#endif	

#if !defined(__WIN32__)
	if (checkCmdLineArg(argc, (const char **)argv, "ogl33"))
	{
		g_useModernOGL = true; // use modern OpenGL 3.3
	}
	if (checkCmdLineArg(argc, (const char **)argv, "nogl"))
	{
		g_bGlutWindow = false;
	}
#endif
#if defined(__LINUX__)
	if (checkCmdLineArg(argc, (const char **)argv, "framebuffer"))
	{
		g_bFramebuffer = true;
		g_bGlutWindow = false;
	}
#endif

	IMAGE_FORMAT outputFormat = IMAGE_FORMAT_UNKNOWN;

	if (getCmdLineArgStr(argc, (const char **)argv, "output_format", &str))
	{
		if (strcmp(str, "RGBA32") == 0)
			outputFormat = IMAGE_FORMAT_RGBA8BIT;
		else if (strcmp(str, "RGBA64") == 0)
			outputFormat = IMAGE_FORMAT_RGBA16BIT;
		else 
			printf("output_format set incorrect!\n");
	}

#if defined(__LINUX__)
	if (g_bFramebuffer)
	{
		outputFormat = IMAGE_FORMAT_RGBA8BIT;
		printf("<for this mode (framebuffer): parameter output_format was set in RGBA32!>\n");
	}
#endif

	int res = 0;
	
	ST_VIDEO_DECODER_PARAMS dec_params;

	dec_params.max_count_decoders = iMaxCountDecoders;
	dec_params.scale_factor = (CC_VDEC_SCALE_FACTOR)iScale;
	dec_params.outputFormat = outputFormat;
	dec_params.type = VD_TYPE_CPU;
	dec_params.use_cinecoder_d3d11 = g_useCinecoderD3D11;

	if (g_useCuda)
		dec_params.type = VD_TYPE_CUDA;

	if (g_useQuickSync)
		dec_params.type = VD_TYPE_QuickSync;

	if (g_useAMF)
		dec_params.type = VD_TYPE_AMF;

	if (g_useNVDEC)
		dec_params.type = VD_TYPE_NVDEC;

#if defined(__WIN32__) && !defined(__USE_GLUT_RENDER__)
	std::shared_ptr<BaseGPURender> render = nullptr; // pointer to render (OpenGL or DirectX11)

	if (g_useDirectX11)
		render = std::make_shared<GPURenderDX>(); // create object of render using DirectX11
	else
		render = std::make_shared<GPURenderGL>(); // create object of render using OpenGL

	if (render && !render->IsInit())
	{
		int res = render->Init(filename, dec_params, gpuDevice); // init render

		if (res == 0)
			res = render->SetParameters(g_bVSync, g_bRotate, g_bMaxFPS); // set startup parameters

		if (res == 0)
		{
			render->StartPipe(); // wait until the exit
		}
	}

	render = nullptr; // destroy render
#else
	decodeD2 = std::make_shared<DecodeDaniel2>(); // Create decoder for decoding DN2 files

	if (!decodeD2)
	{
		printf("Cannot create create decoder!\n");
		return 0;
	}

	res = decodeD2->OpenFile(filename.c_str(), dec_params); // Open input DN2 file

	if (res != 0)
	{
		printf("Cannot open input file <%s> or create decoder!\n", filename.c_str());
		return 0;
	}

	image_width = (unsigned int)decodeD2->GetImageWidth();	// Get image width
	image_height = (unsigned int)decodeD2->GetImageHeight(); // Get image height

	iAllFrames = decodeD2->GetCountFrames(); // Get count of frames

	ValueFPS = decodeD2->GetFrameRate(); // get frame rate

	if (!g_bFramebuffer)
	{
		InitAudioTrack(filename, decodeD2->GetFrameRateValue());
	}

#if defined(__USE_GLUT_RENDER__)
	if (g_bGlutWindow)
	{
		if (!gpu_initGLUT(&argc, argv)) // Init GLUT
		{
			printf("Error: cannot init GLUT!\n");
			return 0;
		}

		if (!gpu_initGLBuffers()) // Init GL buffers
		{
			printf("Error: cannot init GL buffers!\n");
			return 0;
		}

		get_versionGLandGLUT(); // print version of OpenGL and freeGLUT
	}
#endif
	// Start timer
	timer.StartTimer();

	timerqFPSMode.StartTimer();

#if defined(__USE_GLUT_RENDER__)
	if (g_bGlutWindow)
	{
		decodeD2->StartDecode(); // Start decoding

		// Start mainloop
		glutMainLoop(); // Wait
	}
	else
#endif
	{
		bool bRotate = g_bRotate;

		//g_bMaxFPS = true;
		g_bShowTexture = false;
		g_bCopyToTexture = false;
		//g_bDecoder = false;

		//decodeD2->SetDecode(g_bDecoder);
		//decodeD2->SetReadFile(false);

#if defined(__LINUX__)
		size_t page_size = 0;
		unsigned char* pb = nullptr;
		
		std::thread thMouse;

		if (g_bFramebuffer)
		{
			res = frame_buffer.Init();

			if (res != 0)
			{
				printf("Error: cannot initialize framebuffer!\n");
				return 0;
			}
			g_bCopyToTexture = true;

			page_size = frame_buffer.SizeBuffer();
			g_var_info = frame_buffer.GetVInfo();

			thMouse = std::thread(CLI_OnMouseMove);
		}
#endif
		decodeD2->StartDecode(); // Start decoding

		while (true)
		{
			if (!g_bMaxFPS && g_bVSyncHand)
			{
				double timestep = 1000.0 / ValueFPS;

				double ms_elapsed = timerqFPSMode.GetElapsedTime();

				int dT = (int)(timestep - ms_elapsed);

				if (dT > 1)
					std::this_thread::sleep_for(std::chrono::milliseconds(dT));

				timerqFPSMode.StartTimer();
			}

			if (!g_bPause)
			{
#if defined(__LINUX__)
				if (g_bFramebuffer)
				{
					pb = frame_buffer.GetPtr();

					res = copy_to_framebuffer(pb, page_size);

					CLI_Draw(pb);

					frame_buffer.SwapBuffers();
				}
				else
#endif
				{
					// Copy data from queue to texture
					res = gpu_generateImage(bRotate);
				}

				if (res < 0)
					printf("Load texture from decoder failed!\n");

				ComputeFPS(); // Calculate fps
			}
			else
			{
#if defined(__LINUX__)
				if (g_bFramebuffer)
				{
                    pb = frame_buffer.GetPtr();

                    res = copy_to_framebuffer(pb, page_size);

                    CLI_Draw(pb);

                    frame_buffer.SwapBuffers();
				}
#endif
				std::this_thread::sleep_for(std::chrono::milliseconds(1)); // to unload CPU when paused
			}

			if (_kbhit())
			{
				char ch = _getch();
				if (ch == 27) break;
				else Keyboard(ch, 0, 0);
			}
		}

		if (decodeD2)
			decodeD2 = nullptr; // destroy video decoder

		if (decodeAudio)
			decodeAudio = nullptr; // destroy audio decoder

#if defined(__LINUX__)
		if (g_bFramebuffer)
		{
			thMouse.detach();
			frame_buffer.Destroy();
		}
#endif

	}
#endif

#ifdef USE_CUDA_SDK
#ifdef CUDA_WRAPPER
	if (g_useCuda)
	{
		__DestroyCUDA(); // destroy CUDA SDK
	}
#endif
#endif

#ifdef USE_OPENCL_SDK
#ifdef OPENCL_WRAPPER
	if (g_useOpenCL)
	{
		UnLoadOpenClLib(); // destroy OpenCL SDK
	}
#endif
#endif

	return 0;
}
