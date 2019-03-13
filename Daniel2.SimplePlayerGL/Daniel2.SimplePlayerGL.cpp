// SimpleDecodeDN2.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

bool g_bPause = false;
bool g_bVSync = false;
bool g_bRotate = true;
bool g_bLastRotate = g_bRotate;
bool g_useCuda = false;
bool g_useDirectX11 = false;

bool g_bMaxFPS = false;
bool g_bVSyncHand = true;

#if defined(__WIN32__)
#include "GPURenderGL.h"
#include "GPURenderDX.h"
#else
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
	printf("Command line example: <Daniel2.SimplePlayerGL.exe> <file_path> -decoders 2 -vsync -rotate_frame\n");
	printf("\n");
	printf("Options:\n");
	printf("-help               display this help menu\n");
	printf("-decoders <N>       max count of decoders [1..4] (default: 2)\n");
	//printf("-vsync              enable vertical synchronisation (default - disable)\n");
	printf("-fpsmax             enable maximum playing fps (default - disable)\n");
	printf("-rotate_frame       enable rotate frame (default - disable)\n");
#ifdef USE_CUDA_SDK
	printf("-cuda               enable CUDA decoding (default - disable, PC only)\n");
#endif
#if defined(__WIN32__)
	printf("-d3d11              enable DirectX11 pipeline (default - OpenGL)\n");
#endif
	printf("\nCommands:\n");
	printf("'ESC':              exit\n");
	printf("'p' or 'SPACE':     on/off pause\n");
	printf("'v':                on/off vertical synchronisation\n");
	printf("'m':                on/off maximum playing fps\n");
	printf("'r':                on/off rotate image\n");
	printf("'f':                on/off fullscreen mode\n");
	printf("'t':                on/off copy result to texture\n");
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

	char *str = nullptr;

	filename = argv[1];

	if (getCmdLineArgStr(argc, (const char **)argv, "decoders", &str))
	{
		iMaxCountDecoders = atoi(str); // max count of decoders [1..4] (default: 2)
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

	if (g_useCuda)
	{
		if (initCUDA() != 0) // init CUDA SDK
		{
			printf("Error: cannot initialize CUDA! Please check if the %s file exists!\n", CUDART_FILENAME);
			return 0;
		}
	}
#endif

#if defined(__WIN32__)
	if (checkCmdLineArg(argc, (const char **)argv, "d3d11"))
	{
		g_useDirectX11 = true; // use DirectX11 pipeline
	}
#endif	

	int res = 0;
	
#if defined(__WIN32__)
	std::shared_ptr<BaseGPURender> render = nullptr; // pointer to render (OpenGL or DirectX11)

	if (g_useDirectX11)
		render = std::make_shared<GPURenderDX>(); // create object of render using DirectX11
	else
		render = std::make_shared<GPURenderGL>(); // create object of render using OpenGL

	if (render && !render->IsInit())
	{
		int res = render->Init(filename, iMaxCountDecoders, g_useCuda); // init render

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

	res = decodeD2->OpenFile(filename.c_str(), iMaxCountDecoders, g_useCuda); // Open input DN2 file

	if (res != 0)
	{
		printf("Cannot open input file <%s> or create decoder!\n", filename.c_str());
		return 0;
	}

	image_width = (unsigned int)decodeD2->GetImageWidth();	// Get image width
	image_height = (unsigned int)decodeD2->GetImageHeight(); // Get image height

	iAllFrames = decodeD2->GetCountFrames(); // Get count of frames

	ValueFPS = decodeD2->GetFrameRate(); // get frame rate

	InitAudioTrack(filename, decodeD2->GetFrameRateValue());

	gpu_initGLUT(&argc, argv); // Init GLUT

	gpu_initGLBuffers(); // Init GL buffers

	get_versionGLandGLUT(); // print version of OpenGL and freeGLUT

	// Start timer
	timer.StartTimer();

	timerqFPSMode.StartTimer();

	decodeD2->StartDecode(); // Start decoding

	// Start mainloop
	glutMainLoop(); // Wait

	if (decodeD2)
        decodeD2 = nullptr; // destroy video decoder

    if (decodeAudio)
        decodeAudio = nullptr; // destroy audio decoder
#endif

#ifdef USE_CUDA_SDK
	if (g_useCuda)
	{
		destroyCUDA(); // destroy CUDA SDK
	}
#endif

	return 0;
}
