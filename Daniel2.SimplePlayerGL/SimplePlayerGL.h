#pragma once

#if defined(__WIN32__)
#include <GL/freeglut.h> // GLUT framework
#elif defined(__APPLE__)
#include <GLUT/glut.h> // GLUT framework
#include <OpenGL/OpenGL.h> // OpenGL framework
#include <ApplicationServices/ApplicationServices.h> // CoreGraphics
#define sprintf_s sprintf
#elif defined(__LINUX__)
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <GL/glx.h>
#include <GL/freeglut.h> // GLUT framework
#endif

#include "Timer.h"
#include "DecodeDaniel2.h"
#include "AudioSource.h"

///////////////////////////////////////////////////////

inline bool CheckErrorGL(const char *file, const int line)
{
	bool ret_val = true;

	// check for error
	GLenum gl_error = glGetError();

	if (gl_error != GL_NO_ERROR)
	{
		char tmpStr[255];
		sprintf_s(tmpStr, "\n%s(%i) : GL Error : %s\n\n", file, line, gluErrorString(gl_error));

		fprintf(stderr, "%s", tmpStr);
		printf("%s", tmpStr);
		ret_val = false;
	}

	return ret_val;
}

#define OGL_CHECK_ERROR_GL() \
	CheckErrorGL(__FILE__, __LINE__)

///////////////////////////////////////////////////////

#if defined(__WIN32__)
typedef BOOL(WINAPI *PFNWGLSWAPINTERVALEXTPROC_GLOBAL)(int interval);
PFNWGLSWAPINTERVALEXTPROC_GLOBAL g_wglSwapInterval;
#endif

#define GL_CLAMP_TO_EDGE 0x812F
#define GL_UNSIGNED_INT_2_10_10_10_REV 0x8368
#define GL_UNSIGNED_INT_10_10_10_2 0x8036

#define GL_TEXTURE_SWIZZLE_RGBA 0x8E46

///////////////////////////////////////////////////////

#define TITLE_WINDOW_APP "TestApp OGL(Decode Daniel2)"

bool g_bCopyToTexture = true;
bool g_bDecoder = true;

bool g_bFullScreen = false;
bool g_bShowSlider = false;

int window_width = 1280; // start value for window width
int window_height = 720; // start value for window height

int iGLUTWindowHandle = 0; // Handle to the GLUT window

float start_ortho_w = 0;
float start_ortho_h = 0;

float stop_ortho_w = (float)window_width;
float stop_ortho_h = (float)window_height;

int nViewportWidth = window_width;
int nViewportHeight = window_height;

size_t iAllFrames = 1;
size_t iCurPlayFrameNumber = 0;
float edgeLineX = 10.f;
float edgeLineY = 30.f;
float sizeSquare = 10.f;
float sizeSquare2 = sizeSquare / 2;

int g_mouse_state = -1;
int g_mouse_button = -1;

C_CritSec g_mutex; // global mutex

///////////////////////////////////////////////////////

#ifdef USE_CUDA_SDK
cudaGraphicsResource_t cuda_tex_result_resource = nullptr;
#endif

GLuint tex_result;  // Where we will copy result

unsigned int image_width = 1280; // start value for image width
unsigned int image_height = 720; // start value for image height

GLint g_internalFormat = GL_RGBA;
GLenum g_format = GL_BGRA_EXT;
GLenum g_type = GL_UNSIGNED_BYTE;

// Timer
static int fpsCount = 0;
C_Timer timer;

C_Timer timerqFPSMode;
double ValueFPS = 60.0;

///////////////////////////////////////////////////////

std::shared_ptr<DecodeDaniel2> decodeD2; // Wrapper class which demonstration decoding Daniel2 format

std::shared_ptr<AudioSource> decodeAudio; // Audio decoder

///////////////////////////////////////////////////////

int InitAudioTrack(std::string filename, CC_FRAME_RATE frameRate)
{
	int res = 0;

	decodeAudio = std::make_shared<AudioSource>(); // Create decoder for audio track

	if (!decodeAudio)
	{
		printf("Cannot create create audio decoder!\n");
		return 0;
	}

	res = decodeAudio->Init(frameRate); // Init audio decoder

	if (res != 0)
		return res;

#if defined(__WIN32__)
	res = decodeAudio->OpenFile(filename.c_str()); // Open audio stream

	if (res == 0)
		printf("Audio track: Yes\n");
	else
		printf("Audio track: No (error = 0x%x)\n", res);

	printf("-------------------------------------\n");
#endif
	return res;
}

///////////////////////////////////////////////////////

void RenderWindow();
void Keyboard(unsigned char key, int x, int y);
void SpecialKeyboard(int key, int x, int y);
void Cleanup();
void Reshape(int x, int y);
void AnimateScene(void);

void OnMouseClick(int button, int state, int x, int y);
void OnMouseMove(int x, int y);

void SetPause(bool bPause);
void SetVerticalSync(bool bVerticalSync);
void ComputeFPS();

void SeekToFrame(int x, int y);
void SeekToFrame(size_t iFrame);

///////////////////////////////////////////////////////

void get_versionGLandGLUT()
{
	char *versionGL = nullptr;
	versionGL = (char *)(glGetString(GL_VERSION));

	printf("OpenGL version: %s\n", versionGL);

#if defined(__WIN32__)
	GLint versionFreeGlutInt = 0;
	versionFreeGlutInt = (glutGet(GLUT_VERSION));

	if (versionFreeGlutInt > 0)
	{
		std::string versionFreeGlutString = std::to_string((long long)(versionFreeGlutInt));
		versionFreeGlutString.insert(1, "."); // transforms 30000 into 3.0000
		versionFreeGlutString.insert(4, "."); // transforms 3.0000 into 3.00.00

		printf("FreeGLUT version: %s\n", versionFreeGlutString.c_str());
	}
#endif

	printf("-------------------------------------\n");
}

bool gpu_initGLUT(int *argc, char **argv)
{
#if defined(__WIN32__)
	int iWinW = GetSystemMetrics(SM_CXSCREEN);
	int iWinH = GetSystemMetrics(SM_CYSCREEN);
#elif defined(__APPLE__)
	CGRect mainMonitor = CGDisplayBounds(CGMainDisplayID());
	CGFloat monitorHeight = CGRectGetHeight(mainMonitor);
	CGFloat monitorWidth = CGRectGetWidth(mainMonitor);

	int iWinW = (int)monitorWidth;
	int iWinH = (int)monitorHeight;
#elif defined(__LINUX__)
	Display* dpy = XOpenDisplay(NULL);
	Screen*  screen = DefaultScreenOfDisplay(dpy);
	int iWinW = screen->width;
	int iWinH = screen->height;
	XCloseDisplay(dpy);
#endif

	float fKoeffDiv = (float)image_width / ((float)iWinW / 3.f);

	// Correction start of window size using global height of monitor
	while (window_height >= iWinH)
		fKoeffDiv *= 1.5f;

	window_width = static_cast<unsigned int>((float)(image_width) / fKoeffDiv);
	window_height = static_cast<unsigned int>((float)(image_height) / fKoeffDiv);

	// Create GL context
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(window_width, window_height);
	iGLUTWindowHandle = glutCreateWindow(TITLE_WINDOW_APP);

	// Register callbacks
	glutDisplayFunc(RenderWindow);
	glutKeyboardFunc(Keyboard);
	glutReshapeFunc(Reshape);
	glutIdleFunc(AnimateScene);

	glutMouseFunc(OnMouseClick);
	glutMotionFunc(OnMouseMove);
	glutPassiveMotionFunc(OnMouseMove);
	glutSpecialFunc(SpecialKeyboard);

#if defined(__APPLE__)
	atexit(Cleanup);
#else
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
	glutCloseFunc(Cleanup);
#endif

	glutPositionWindow(100, 100); // Start position window

	SetVerticalSync(g_bVSync); // Set value of vertical synchronisation (on/off)

	OGL_CHECK_ERROR_GL();

	return true;
}

void gpu_initGLBuffers()
{
	// Create texture

	glGenTextures(1, &tex_result);
	glBindTexture(GL_TEXTURE_2D, tex_result);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	if (decodeD2->GetImageFormat() == IMAGE_FORMAT_RGBA8BIT || decodeD2->GetImageFormat() == IMAGE_FORMAT_BGRA8BIT) // RGBA 8 bit
	{
		g_internalFormat = GL_RGBA;
		g_type = GL_UNSIGNED_BYTE;
		//g_format = GL_RGBA;      // this one is 2x faster
		//if (decodeD2->GetImageFormat() == IMAGE_FORMAT_BGRA8BIT) g_format = GL_RGBA;
	}
	else if (decodeD2->GetImageFormat() == IMAGE_FORMAT_RGB30) // R10G10B10A2 fromat
	{
		g_internalFormat = GL_RGB10;
		g_format = GL_RGBA;
		//g_type = GL_UNSIGNED_INT_10_10_10_2;
		g_type = GL_UNSIGNED_INT_2_10_10_10_REV;	// this one is 2x faster
	}
	else if (decodeD2->GetImageFormat() == IMAGE_FORMAT_RGBA16BIT || decodeD2->GetImageFormat() == IMAGE_FORMAT_BGRA16BIT) // RGBA 16 bit
	{
		g_internalFormat = GL_RGBA16;
		g_type = GL_UNSIGNED_SHORT;
		//if (decodeD2->GetImageFormat() == IMAGE_FORMAT_BGRA16BIT) g_format = GL_RGBA;
	}
	else
	{
		printf("Image format is invalid!\n");
	}

	g_format = GL_RGBA;

	glTexImage2D(GL_TEXTURE_2D, 0, g_internalFormat, image_width, image_height, 0, g_format, g_type, NULL);

	if (decodeD2->GetImageFormat() == IMAGE_FORMAT_BGRA8BIT || decodeD2->GetImageFormat() == IMAGE_FORMAT_BGRA16BIT)
	{
		GLint swizzleMask[] = { GL_BLUE, GL_GREEN, GL_RED, GL_ALPHA };
		glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzleMask);
	}

	glBindTexture(GL_TEXTURE_2D, 0);

	OGL_CHECK_ERROR_GL();

#ifdef USE_CUDA_SDK
	// register this textures with CUDA

	cuda_tex_result_resource = nullptr;

	if (g_useCuda)
	{
		cudaError cuErr;

		cuErr = cudaGraphicsGLRegisterImage(&cuda_tex_result_resource, tex_result, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard); __vrcu
	}
#endif

	OGL_CHECK_ERROR_GL();
}

void gpu_UpdateGLSettings()
{
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f); // Clear buffer
	glEnable(GL_TEXTURE_2D); // Allow texture mapping

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(start_ortho_w, stop_ortho_w, stop_ortho_h, start_ortho_h, -1, 1);

	glMatrixMode(GL_MODELVIEW);
	glViewport(0, 0, nViewportWidth, nViewportHeight);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	glClear(GL_COLOR_BUFFER_BIT);

	// Initialization for transparency
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glClearColor(0.0, 0.0, 0.0, 0.0);

	OGL_CHECK_ERROR_GL();
}

#ifdef USE_CUDA_SDK
int gpu_generateCUDAImage(C_Block* pBlock)
{
	// We want to copy cuda_dest_resource data to the texture
	// map buffer objects to get CUDA device pointers

	if (!cuda_tex_result_resource)
		return -1;

	const cudaPtr cuda_dest_resource = pBlock->DataGPUPtr();

	cudaArray *texture_ptr;
	cudaGraphicsMapResources(1, &cuda_tex_result_resource, 0); __vrcu
	cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_result_resource, 0, 0); __vrcu
	cudaMemcpy2DToArray(texture_ptr, 0, 0, cuda_dest_resource, pBlock->Pitch(), pBlock->Pitch(), pBlock->Height(), cudaMemcpyDeviceToDevice); __vrcu
	cudaGraphicsUnmapResources(1, &cuda_tex_result_resource, 0); __vrcu

	return 0;
}
#endif

int gpu_generateImage(bool & bRotateFrame)
{
	if (!decodeD2->isProcess() || decodeD2->isPause()) // check for pause or process
		return 1;

	C_Block *pBlock = decodeD2->MapFrame(); // Get poiter to picture after decoding

	if (!pBlock)
		return -1;

#ifdef USE_CUDA_SDK
	if (g_useCuda)
	{
		if (g_bCopyToTexture)
		{
			gpu_generateCUDAImage(pBlock);
		}
	}
	else
#endif
	{
		if (g_bCopyToTexture)
		{
			glTexImage2D(GL_TEXTURE_2D, 0, g_internalFormat, image_width, image_height, 0, g_format, g_type, pBlock->DataPtr()); // coping decoded frame into the GL texture
		}
	}

	bRotateFrame = pBlock->GetRotate() ? !bRotateFrame : bRotateFrame; // Rotate frame

	g_bLastRotate = pBlock->GetRotate(); // Save frame rotation value

	iCurPlayFrameNumber = pBlock->iFrameNumber; // Save currect frame number

	decodeD2->UnmapFrame(pBlock); // Add free pointer to queue

	return 0;
}

void RenderWindow()
{
	C_AutoLock lock(&g_mutex);

	if (!g_bVSync && !g_bMaxFPS && g_bVSyncHand)
	{
		double timestep = 1000.0 / ValueFPS;

		double ms_elapsed = timerqFPSMode.GetElapsedTime();

		int dT = (int)(timestep - ms_elapsed);

		if (dT > 1)
			std::this_thread::sleep_for(std::chrono::milliseconds(dT));

		timerqFPSMode.StartTimer();
	}

	bool bRotate = g_bRotate;

	int res = 1;

	if (!g_bPause)
	{
		// Copy data from queue to texture
		res = gpu_generateImage(bRotate);

		if (res < 0)
			printf("Load texture from decoder failed!\n");
	}
	else
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(100)); // to unload CPU when paused
	}

	if (res != 0)
	{
		bRotate = g_bLastRotate ? !bRotate : bRotate; // Rotate frame
	}

	float nCoordL = 0.0f;
	float nCoordR = (float)nViewportWidth;
	float fBottom = (float)nViewportHeight;
	float fTop = 0.0f;

	// Update GL settings
	gpu_UpdateGLSettings();

	if (g_bShowTexture)
	{
		// Draw texture
		glBindTexture(GL_TEXTURE_2D, tex_result);
		glBegin(GL_QUADS);

		if (bRotate)
		{
			glTexCoord2f(0.0, 0.0);
			glVertex2f(nCoordL, fTop);
			glTexCoord2f(1.0, 0.0);
			glVertex2f(nCoordR, fTop);
			glTexCoord2f(1.0, 1.0);
			glVertex2f(nCoordR, fBottom);
			glTexCoord2f(0.0, 1.0);
			glVertex2f(nCoordL, fBottom);
		}
		else
		{
			glTexCoord2f(0.0, 0.0);
			glVertex2f(nCoordL, fBottom);
			glTexCoord2f(1.0, 0.0);
			glVertex2f(nCoordR, fBottom);
			glTexCoord2f(1.0, 1.0);
			glVertex2f(nCoordR, fTop);
			glTexCoord2f(0.0, 1.0);
			glVertex2f(nCoordL, fTop);
		}

		glEnd();
		glDisable(GL_TEXTURE_2D);
	}

	if (decodeAudio && decodeAudio->IsInitialize())
		decodeAudio->PlayFrame(iCurPlayFrameNumber); // play audio

	if (g_bShowSlider) // draw slider
	{
		GLint w = glutGet(GLUT_WINDOW_WIDTH); // Width in pixels of the current window
		GLint h = glutGet(GLUT_WINDOW_HEIGHT); // Height in pixels of the current window

		sizeSquare2 = (float)w / 100;
		edgeLineY = sizeSquare2 * 4;
		edgeLineX = sizeSquare2 * 2;

		float xCoord = edgeLineX + ((((float)w - (2.f * edgeLineX)) / (float)(iAllFrames - 1)) * (float)iCurPlayFrameNumber);
		float yCoord = (float)h - edgeLineY;

		glColor4f(0.f, 0.f, 0.f, 0.5);
		glBegin(GL_QUADS);
		glVertex2f(0, (float)h);
		glVertex2f((float)w, (float)h);
		glVertex2f((float)w, (float)h - (edgeLineY * 2));
		glVertex2f(0, (float)h - (edgeLineY * 2));
		glEnd();

		glLineWidth(2);
		glColor4f(0.f, 1.f, 0.f, 0.5);

		if ((xCoord - sizeSquare2) > (0 + edgeLineX))
		{
			glBegin(GL_LINES);
			glVertex2f(0 + edgeLineX, yCoord);
			glVertex2f(xCoord - sizeSquare2, yCoord);
			glEnd();
		}

		glBegin(GL_LINES);
		glVertex2f(xCoord - sizeSquare2, yCoord - sizeSquare2);
		glVertex2f(xCoord + sizeSquare2, yCoord - sizeSquare2);

		glVertex2f(xCoord + sizeSquare2, yCoord - sizeSquare2);
		glVertex2f(xCoord + sizeSquare2, yCoord + sizeSquare2);

		glVertex2f(xCoord + sizeSquare2, yCoord + sizeSquare2);
		glVertex2f(xCoord - sizeSquare2, yCoord + sizeSquare2);

		glVertex2f(xCoord - sizeSquare2, yCoord + sizeSquare2);
		glVertex2f(xCoord - sizeSquare2, yCoord - sizeSquare2);
		glEnd();

		if ((xCoord + sizeSquare2) < ((float)w - edgeLineX))
		{
			glColor4f(1.f, 1.f, 1.f, 0.5);
			glBegin(GL_LINES);
			glVertex2f(xCoord + sizeSquare2, yCoord);
			glVertex2f((float)w - edgeLineX, yCoord);
			glEnd();
		}

		glColor4f(0.f, 0.f, 0.f, 1.0);
	}

	glutSwapBuffers(); // swaps the buffers of the current window if double buffered.

	OGL_CHECK_ERROR_GL();

	ComputeFPS(); // Calculate fps
}

void ComputeFPS()
{
	// Update fps counter, fps/title display
	fpsCount++;
	double ms_elapsed = timer.GetElapsedTime();

	if (ms_elapsed >= 1000.0f)
	{
		static bool bInit = true;

		double fps = (fpsCount / (ms_elapsed / 1000.0));

		size_t data_rate = decodeD2->GetDataRate(true);
		double fDataRate = bInit ? 0.0, bInit = false : (data_rate * 1000) / ms_elapsed / (1024 * 1024);

		char cString[256];
		std::string cTitle;

		GLint w = glutGet(GLUT_WINDOW_WIDTH); // Width in pixels of the current window
		GLint h = glutGet(GLUT_WINDOW_HEIGHT); // Height in pixels of the current window

		if (g_bPause)
			sprintf_s(cString, "%s (%d x %d): (Pause)", TITLE_WINDOW_APP, w, h);
		else
			sprintf_s(cString, "%s (%d x %d): %.0f fps data_rate = %.2f MB/s", TITLE_WINDOW_APP, w, h, fps, fDataRate);

		cTitle = cString;
		switch (g_internalFormat)
		{
		case GL_RGBA: cTitle += " fmt=RGBA32"; break;
		case GL_RGB10: cTitle += " fmt=RGB30"; break;
		case GL_RGBA16: cTitle += " fmt=RGBA64"; break;
		default: break;
		}

		cTitle += " cur_frm=";
		cTitle += std::to_string((long long)iCurPlayFrameNumber); // print current frame number

		glutSetWindowTitle(cTitle.c_str());

		fpsCount = 0;

		timer.StartTimer();
	}
}

void Keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	switch (key)
	{
	case 27:
	{
#if defined(__APPLE__)
		exit(0); // On MacOS we have error <Use of undeclared identifier 'glutLeaveMainLoop'> so we call exit(0)
#else
		glutLeaveMainLoop();
#endif
		break;
	}

	case '+':
	case '-':
	{
		if (decodeAudio && decodeAudio->IsInitialize())
		{
			float volume = decodeAudio->GetVolume() + (key == '+' ? 0.1f : -0.1f);
			if (volume > 1.f) volume = 1.f;
			else if (volume < 0.f) volume = 0;
			decodeAudio->SetVolume(volume);
			printf("audio volume = %.0f %%\n", decodeAudio->GetVolume() * 100.f);
		}
		break;
	}

	case 32:
	case 'p':
	{
		if (g_bPause)
			SeekToFrame(iCurPlayFrameNumber);

		SetPause(!g_bPause);
		break;
	}

	case 'j':
	{
		int iSpeed = decodeD2->GetSpeed();

		if (iSpeed < 0)
			decodeD2->SetSpeed(iSpeed * 2);
		else
			decodeD2->SetSpeed(-1);

		if (g_bPause)
		{
			SeekToFrame(iCurPlayFrameNumber);
			SetPause(!g_bPause);
		}

		if (decodeAudio && decodeAudio->IsInitialize())
			decodeAudio->SetSpeed(decodeD2->GetSpeed());

		printf("press J (speed: %dx)\n", decodeD2->GetSpeed());
		break;
	}
	case 'k':
	{
		int iSpeed = decodeD2->GetSpeed();

		if (iSpeed > 0)
			decodeD2->SetSpeed(1);
		else
			decodeD2->SetSpeed(-1);

		if (g_bPause)
			SeekToFrame(iCurPlayFrameNumber);

		SetPause(!g_bPause);

		if (decodeAudio && decodeAudio->IsInitialize())
			decodeAudio->SetSpeed(decodeD2->GetSpeed());

		printf("press K (speed: %dx)\n", decodeD2->GetSpeed());
		break;
	}
	case 'l':
	{
		int iSpeed = decodeD2->GetSpeed();

		if (iSpeed > 0)
			decodeD2->SetSpeed(iSpeed * 2);
		else
			decodeD2->SetSpeed(1);

		if (g_bPause)
		{
			SeekToFrame(iCurPlayFrameNumber);
			SetPause(!g_bPause);
		}

		if (decodeAudio && decodeAudio->IsInitialize())
			decodeAudio->SetSpeed(decodeD2->GetSpeed());

		printf("press L (speed: %dx)\n", decodeD2->GetSpeed());
		break;
	}

	case 'v':
	{
		g_bVSync = !g_bVSync;
		SetVerticalSync(g_bVSync);

		if (g_bVSync)
			printf("vertical synchronisation: on\n");
		else
			printf("vertical synchronisation: off\n");

		break;
	}

	case 'm':
	{
		g_bMaxFPS = !g_bMaxFPS;

		if (g_bMaxFPS)
			printf("maximum playing fps: on\n");
		else
			printf("maximum playing fps: off\n");

		break;
	}

	case 'r':
	{
		g_bRotate = !g_bRotate;

		if (g_bRotate)
			printf("rotate image: on\n");
		else
			printf("rotate image: off\n");

		break;
	}

	case 'f':
	{
		g_bFullScreen = !g_bFullScreen;

		if (g_bFullScreen)
			glutFullScreen(); // if uses freeglut 3.0 and 4K image -> GL error invalid framebuffer operation
		else
		{
			glutPositionWindow(100, 100); // Start position window
			glutReshapeWindow(window_width, window_height); // requests a change to the size of the current window.
		}

		if (g_bFullScreen)
			printf("fullscreen mode: on\n");
		else
			printf("fullscreen mode: off\n");

		break;
	}

	case 't':
	{
		g_bCopyToTexture = !g_bCopyToTexture;

		if (g_bCopyToTexture)
			printf("copy result to texture: on\n");
		else
			printf("copy result to texture: off\n");

		break;
	}

	case 'o':
	{
		g_bShowTexture = !g_bShowTexture;

		if (g_bShowTexture)
			printf("show texture: on\n");
		else
			printf("show texture: off\n");

		break;
	}

	case 'd':
	{
		g_bDecoder = !g_bDecoder;
		decodeD2->SetDecode(g_bDecoder);

		if (g_bDecoder)
			printf("decoder: on\n");
		else
			printf("decoder: off\n");

		break;
	}

	case 'n':
	{
		bool bReadFile = decodeD2->GetReadFile();
		decodeD2->SetReadFile(!bReadFile);

		bReadFile = decodeD2->GetReadFile();

		if (bReadFile)
			printf("read file: on\n");
		else
			printf("read file: off\n");

		break;
	}

	default:
		break;
	}
}

void SpecialKeyboard(int key, int x, int y)
{
	switch (key)
	{
	case GLUT_KEY_HOME:
	{
		SeekToFrame(0);
		break;
	}
	case GLUT_KEY_END:
	{
		SeekToFrame(iAllFrames - 1);
		break;
	}
	case GLUT_KEY_RIGHT:
	{
		if (iCurPlayFrameNumber <= iAllFrames - 2)
			SeekToFrame(iCurPlayFrameNumber + 1);
		break;
	}
	case GLUT_KEY_LEFT:
	{
		if (iCurPlayFrameNumber >= 1)
			SeekToFrame(iCurPlayFrameNumber - 1);
		break;
	}
	default:
		break;
	}

	RenderWindow(); // update frame
}

void Cleanup()
{
	// Stop decode pipe
	decodeD2->StopDecode();

	// Delete GL texture
	glDeleteTextures(1, &tex_result);

#ifdef USE_CUDA_SDK
	if (cuda_tex_result_resource)
	{
		cudaGraphicsUnregisterResource(cuda_tex_result_resource); __vrcu
	}
#endif
	OGL_CHECK_ERROR_GL();
}

void Reshape(int x, int y)
{
	nViewportWidth = x;
	nViewportHeight = y;

	stop_ortho_w = (float)nViewportWidth;
	stop_ortho_h = (float)nViewportHeight;
}

void AnimateScene(void)
{
	// Force redraw
	glutPostRedisplay();
}

void OnMouseClick(int button, int state, int x, int y)
{
	g_mouse_state = state;
	g_mouse_button = button;

	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
	{
		if (g_bShowSlider)
		{
			SeekToFrame(x, y);
		}
		else
		{
			SetPause(!g_bPause);
		}
	}
	RenderWindow(); // update frame to improve performance of scrubbing
}

void OnMouseMove(int x, int y)
{
	GLint w = glutGet(GLUT_WINDOW_WIDTH); // Width in pixels of the current window
	GLint h = glutGet(GLUT_WINDOW_HEIGHT); // Height in pixels of the current window

	float fStartY = (float)h - (edgeLineY * 2.f);
	float fStopY = (float)h;

	if (((float)y >= fStartY) && ((float)y <= fStopY))
	{
		if (g_mouse_state == GLUT_DOWN && g_mouse_button == GLUT_LEFT_BUTTON)
		{
			SeekToFrame(x, y);
			RenderWindow(); // update frame to improve performance of scrubbing
		}

		g_bShowSlider = true;
	}
	else
	{
		g_bShowSlider = false;
	}
}

void SetPause(bool bPause)
{
	g_bPause = bPause;

	if (g_bPause)
		printf("pause: on\n");
	else
		printf("pause: off\n");

	if (decodeAudio && decodeAudio->IsInitialize())
		decodeAudio->SetPause(g_bPause);
}

void SetVerticalSync(bool bVerticalSync)
{
#if defined(__WIN32__)
	if (!g_wglSwapInterval)
	{
		g_wglSwapInterval = (PFNWGLSWAPINTERVALEXTPROC_GLOBAL)wglGetProcAddress("wglSwapIntervalEXT");

		if (!g_wglSwapInterval)
			g_wglSwapInterval = (PFNWGLSWAPINTERVALEXTPROC_GLOBAL)wglGetProcAddress("wglSwapInterval");
	}

	if (g_wglSwapInterval)
		g_wglSwapInterval(bVerticalSync);
#elif defined(__APPLE__)
	GLint                       sync = bVerticalSync;
	CGLContextObj               ctx = CGLGetCurrentContext();

	CGLSetParameter(ctx, kCGLCPSwapInterval, &sync);
#elif defined(__LINUX__)
	void(*swapInterval)(int) = (void(*)(int)) glXGetProcAddress((const GLubyte*) "glXSwapIntervalSGI");
	if (!swapInterval)
		swapInterval = (void(*)(int)) glXGetProcAddress((const GLubyte*) "glXSwapIntervalMESA");

	if (swapInterval)
		swapInterval(bVerticalSync ? 1 : 0);
#endif
	OGL_CHECK_ERROR_GL();
}

void SeekToFrame(size_t iFrame)
{
	C_AutoLock lock(&g_mutex);

	decodeD2->SeekFrame(iFrame); // Setting the reading of the input file from the expected frame (from frame number <iFrame>)

	size_t nReadFrame = 0;
	C_Block *pBlock = nullptr;

	C_Timer seek_timer;
	seek_timer.StartTimer();

	while (seek_timer.GetElapsedTime() <= 5000) // max wait 5 sec
	{
		pBlock = decodeD2->MapFrame(); // Get poiter to picture after decoding
		nReadFrame = pBlock->iFrameNumber; // Get currect frame number

		if (!pBlock)
			break;

		if (nReadFrame == iFrame) // Search for the expected frame
		{
#ifdef USE_CUDA_SDK
			if (g_useCuda)
			{
				if (g_bCopyToTexture)
				{
					gpu_generateCUDAImage(pBlock);
				}
			}
			else
#endif
			{
				if (g_bCopyToTexture)
				{
					glTexImage2D(GL_TEXTURE_2D, 0, g_internalFormat, image_width, image_height, 0, g_format, g_type, pBlock->DataPtr()); // coping decoded frame into the GL texture
				}
			}
			iCurPlayFrameNumber = iFrame; // Save currect frame number
			decodeD2->UnmapFrame(pBlock); // Add free pointer to queue
			SetPause(true); // Set pause enable
			printf("seek to frame %zu\n", iFrame);
			break;
		}

		if (pBlock)
			decodeD2->UnmapFrame(pBlock); // Add free pointer to queue
	}

	if (nReadFrame != iFrame) // The expected frame was not found
	{
		printf("Cannot seek to frame %zu\n", iFrame);
	}
}

void SeekToFrame(int x, int y)
{
	GLint w = glutGet(GLUT_WINDOW_WIDTH); // Width in pixels of the current window
	GLint h = glutGet(GLUT_WINDOW_HEIGHT); // Height in pixels of the current window

	sizeSquare2 = (float)w / 100;
	edgeLineY = sizeSquare2 * 4;
	edgeLineX = sizeSquare2 * 2;

	int iStartX = (int)0;
	int iStopX = (int)(w - (int)(edgeLineX * 2));
	int iStartY = (int)(h - ((int)edgeLineY * 2));
	int iStopY = (int)h;

	x -= (int)edgeLineX;

	if ((x >= iStartX) && (x <= iStopX) &&
		(y >= iStartY) && (y <= iStopY)) // Get the frame number based on the coordinates (x,y)
	{
		size_t iFrame = (size_t)(((float)x * (float)(iAllFrames - 1)) / ((float)w - (2.f * edgeLineX)));

		SeekToFrame(iFrame); // Seek to frame number <iFrame>
	}
}

