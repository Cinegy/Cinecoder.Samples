#pragma once

#if defined(__WIN32__)
//#include <GL/glew.h> // GLEW framework
//#include <GL/freeglut.h> // GLUT framework
#elif defined(__APPLE__)
#ifndef GL_SILENCE_DEPRECATION
#define GL_SILENCE_DEPRECATION
#endif
#include <GL/glew.h> // GLEW framework
#include <GLUT/glut.h> // GLUT framework
#include <OpenGL/OpenGL.h> // OpenGL framework
#include <ApplicationServices/ApplicationServices.h> // CoreGraphics
#define sprintf_s sprintf
#elif defined(__LINUX__)
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <GL/glew.h> // GLEW framework
#include <GL/glx.h> // GLX framework
#include <GL/freeglut.h> // GLUT framework

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

#include "../common/framebuffer.h"
CFrameBuffer frame_buffer;
#endif

#include "../common/dib_draw.h"
#include "../common/cpu_load_meter.h"

#include "DecodeDaniel2.h"
#include "AudioSource.h"

#if defined(__WIN32__)
#include<conio.h>
#elif defined(__LINUX__)
#include <stdio.h>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

#define _getch getchar

int _kbhit(void)
{
	struct termios oldt, newt;
	int ch;
	int oldf;

	tcgetattr(STDIN_FILENO, &oldt);
	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);
	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

	ch = getchar();

	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);

	if (ch != EOF)
	{
		ungetc(ch, stdin);
		return 1;
	}

	return 0;
}
#endif

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
		exit(0);
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

bool g_bGlutWindow = true;

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

#define textdib_size 128*8*2
int textdib[textdib_size*16*2];
int text_size_x = 0, text_size_y = 0;
char textbuf[1024];

//char g_FpsText[256] = {};
CpuLoadMeter cpuLoadMeter;

C_CritSec g_mutex; // global mutex

struct mousecoord
{
	mousecoord() { x = y = 0; }
	mousecoord(int x_, int y_) { x = x_; y = y_; }
	int x, y;
};

std::deque<mousecoord> g_queuecoord;

///////////////////////////////////////////////////////

#ifdef USE_CUDA_SDK
cudaGraphicsResource_t cuda_tex_result_resource = nullptr;
unsigned int bytePerPixel;
#endif

#ifdef USE_OPENCL_SDK
cl_context context;
cl_device_id device;
cl_command_queue queue;
cl_mem imageCL;

#include "CL_ListErrors.h"
#endif

GLuint tex_result;  // Where we will copy result

unsigned int image_width = 1280; // start value for image width
unsigned int image_height = 720; // start value for image height
unsigned int image_size;

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

GLuint fbo = 0;
GLuint rbo = 0;

GLuint vbo = 0;
GLuint pbo = 0; // OpenGL pixel buffer object
#ifdef USE_CUDA_SDK
struct cudaGraphicsResource *cuda_pbo_resource = nullptr; // CUDA Graphics Resource (to transfer PBO)
#endif

//std::vector<unsigned char> pBuffer;
void PBO_to_Texture2D(unsigned char* pData, size_t size);

///////////////////////////////////////////////////////

GLhandleARB program = 0;         // program handles
GLhandleARB vertexShader = 0;
GLhandleARB fragmentShader = 0;

GLint myTextureSampler;
GLuint RotateID;
GLuint SwapRB;
GLuint DrawElement;

bool g_bSwapRB = false;

bool loadShader(GLhandleARB shader, const char * shader_source);
bool loadShaderFromFile(GLhandleARB shader, const char * fileName);

static const char *vertex_shader_source = {
"\
\n//\
\n// Simplest GLSL vertex shader\
\n//\
\n\
\n#version 330 core\
\n\
\nlayout(location = 0) in vec2 vert;\
\nlayout(location = 1) in vec2 vertTexCoord;\
\nlayout(location = 2) in vec2 vertPixCoord;\
\n\
\nout vec2 fragTexCoord;\
\n\
\nuniform int rotate = 0;\
\nuniform int draw_element = 0;\
\n\
\n//uniform mat4 MVP;\
\n\
\nvoid main(void)\
\n{\
\n	fragTexCoord = vertTexCoord;\
\n\
\n	vec2 pos = vert;\
\n\
\n	if (draw_element == 0)\
\n	{\
\n		if (rotate > 0) pos.y *= (-1); // rotate texture\
\n	}\
\n	else if (draw_element >= 1) // slider\
\n		pos = vertPixCoord;\
\n\
\n	gl_Position = vec4(pos, 0, 1);\
\n}\
"
};

static const char *fragment_shader_source = {
"\
\n//\
\n// Simplest fragment shader\
\n//\
\n\
\n#version 330 core\
\n\
\nuniform sampler2D myTextureSampler; //this is the texture\
\nin vec2 fragTexCoord; //this is the texture coord\
\nout vec4 finalColor; //this is the output color of the pixel\
\n\
\nuniform int swap_rb = 0;\
\nuniform int draw_element = 0;\
\n\
\nvoid main()\
\n{\
\n	vec4 color = vec4(0.0, 1.0, 0.0, 1.0);\
\n\
\n	if (draw_element == 0) // texture\
\n	{\
\n		color = texture2D(myTextureSampler, fragTexCoord);\
\n		if (swap_rb > 0) color = color.bgra;\
\n	}\
\n	else if (draw_element == 1) // slider background\
\n	{\
\n		color = vec4(0.0, 0.0, 0.0, 0.5);\
\n	}\
\n	else if (draw_element == 2) // slider\
\n	{\
\n		color = vec4(0.0, 1.0, 0.0, 0.5);\
\n	}\
\n	else if (draw_element == 3) // slider\
\n	{\
\n		color = vec4(1.0, 1.0, 1.0, 0.5);\
\n	}\
\n	finalColor = color;\
}\
"
};

GLuint vao;
GLuint quadvbo;
GLuint texbo;

GLuint Slider_VBObject;
std::vector<GLfloat> slider_pixels;

// X Y Z W
float quad[] = {
	// 1 triangle
	-1.0f, 1.0f, 0.0f, 1.0f,
	-1.0f, -1.0f, 0.0f, 1.0f,
	1.0f, -1.0f, 0.0f, 1.0f,
	// 2 triangle
	1.0f, -1.0f, 0.0f, 1.0f,
	1.0f, 1.0f, 0.0f, 1.0f,
	-1.0f, 1.0f, 0.0f, 1.0f,
};

float texcoord[] = {
	0.0f, 1.0f,
	0.0f, 0.0f,
	1.0f, 0.0f,
	1.0f, 0.0f,
	1.0f, 1.0f,
	0.0f, 1.0f,
};

void printInfoLog(GLhandleARB object)
{
	GLint       logLength = 0;
	GLsizei     charsWritten = 0;
	GLcharARB * infoLog;

	OGL_CHECK_ERROR_GL(); // check for OpenGL errors

	glGetObjectParameterivARB(object, GL_OBJECT_INFO_LOG_LENGTH_ARB, &logLength);

	OGL_CHECK_ERROR_GL(); // check for OpenGL errors

	if (logLength > 0)
	{
		infoLog = (GLcharARB*)malloc(logLength);

		if (infoLog == NULL)
		{
			printf("ERROR: Could not allocate InfoLog buffer\n");

			exit(1);
		}

		glGetInfoLogARB(object, logLength, &charsWritten, infoLog);

		printf("InfoLog:\n%s\n\n", infoLog);
		free(infoLog);
	}

	OGL_CHECK_ERROR_GL(); // check for OpenGL errors
}

bool loadShaderFromFile(GLhandleARB shader, const char * fileName)
{
	printf("Loading %s\n", fileName);

#if defined(__WIN32__)
	FILE * file = nullptr;
	fopen_s(&file, fileName, "rb");
#else
	FILE * file = fopen(fileName, "rb");
#endif

	if (file == NULL)
	{
		printf("Error opening %s\n", fileName);
		exit(1);
	}

	fseek(file, 0, SEEK_END);

	GLint	size = ftell(file);

	if (size < 1)
	{
		fclose(file);
		printf("Error loading file %s\n", fileName);
		exit(1);
	}

	char *buf = (char*)malloc(size);

	fseek(file, 0, SEEK_SET);

	if (fread(buf, 1, size, file) != size)
	{
		fclose(file);
		printf("Error loading file %s\n", fileName);
		exit(1);
	}

	fclose(file);

	bool res = loadShader(shader, buf);
	
	free(buf);

	return res;
}

bool loadShader(GLhandleARB shader, const char * shader_source)
{
	glShaderSourceARB(shader, 1, (const char **)&shader_source, NULL);

	// compile the particle vertex shader, and print out
	glCompileShaderARB(shader);

	OGL_CHECK_ERROR_GL(); // check for OpenGL errors

	GLint compileStatus;
	glGetObjectParameterivARB(shader, GL_OBJECT_COMPILE_STATUS_ARB, &compileStatus);
	printInfoLog(shader);

	return compileStatus != 0;
}

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

void SeekToFrame(int x, int y, bool bLock = true);
void SeekToFrame(size_t iFrame, bool bLock = true);
void SeekToFrameImpl(size_t iFrame);

int gpu_copyImage(unsigned char* pImage, size_t iSize);

///////////////////////////////////////////////////////

#if defined(__LINUX__)
#include "CLI.h"
#endif

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

	if (g_useModernOGL)
	{
#if defined(__APPLE__)
		glutInitDisplayMode(GLUT_3_2_CORE_PROFILE | GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
#elif defined(__LINUX__)
		glutInitContextVersion(3, 3);
		glutInitContextProfile(GLUT_CORE_PROFILE);
		glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
#else
		glutInitContextVersion(3, 3);
		glutInitContextProfile(GLUT_CORE_PROFILE);
		glutInitContextFlags(GLUT_FORWARD_COMPATIBLE);
#endif
	}
	else
	{
		glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
		//glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
	}

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

	glewExperimental = GL_TRUE;
	GLenum GlewInitResult = glewInit();

	if (GlewInitResult != GLEW_OK) {
		// Problem: glewInit failed, something is seriously wrong.
		printf("glewInit failed: %s\n", glewGetErrorString(GlewInitResult));
		exit(1);
	}

	if (!GLEW_ARB_shader_objects)
	{
		printf("GL_ARB_shader_objects NOT supported");
		exit(1);
	}

	OGL_CHECK_ERROR_GL();

	program = 0;

	if (g_useModernOGL)
	{
		GLint linked;

		// create a vertex shader object and a fragment shader object
		vertexShader = glCreateShaderObjectARB(GL_VERTEX_SHADER_ARB);
		fragmentShader = glCreateShaderObjectARB(GL_FRAGMENT_SHADER_ARB);

		//GLuint geomShader = glCreateShader(GL_GEOMETRY_SHADER_ARB);

		// load source code strings into shaders
		if (!loadShader(vertexShader, vertex_shader_source))
			//if (!loadShader(vertexShader, "simplest.vsh"))
			exit(1);

		if (!loadShader(fragmentShader, fragment_shader_source))
			//if (!loadShader(fragmentShader, "simplest.fsh"))
			exit(1);

		// create a program object and attach the two compiled shaders
		program = glCreateProgramObjectARB();

		glAttachObjectARB(program, vertexShader);
		glAttachObjectARB(program, fragmentShader);

		// link the program object and print out the info log
		glLinkProgramARB(program);

		OGL_CHECK_ERROR_GL(); // check for OpenGL errors

		glGetObjectParameterivARB(program, GL_OBJECT_LINK_STATUS_ARB, &linked);

		printInfoLog(program);

		if (!linked)
		{
			program = 0;
			printf("OpenGL modern: cannot build shader program!\n");
			return 0;
		}

		glUseProgram(program);
		myTextureSampler = glGetUniformLocationARB(program, "myTextureSampler");
		RotateID = glGetUniformLocation(program, "rotate");
		SwapRB = glGetUniformLocation(program, "swap_rb");
		DrawElement = glGetUniformLocation(program, "draw_element");
		glUseProgram(0);

		OGL_CHECK_ERROR_GL();

		printf("OpenGL modern: use\n");
	}

	return true;
}

int gpu_copyImage(unsigned char* pImage, size_t iSize)
{
#if defined(__LINUX__)
	if (g_bFramebuffer)
	{
		if (g_fb.size() < iSize)
			g_fb.resize(iSize);

		size_t size_copy = min(g_fb.size(), iSize);
		memcpy(g_fb.data(), pImage, size_copy);
		return 0;
	}
#endif
	if (pImage)
	{
		if (pbo)
			PBO_to_Texture2D(pImage, image_size); // coping decoded frame into the GL texture using PBO
		else
			glTexImage2D(GL_TEXTURE_2D, 0, g_internalFormat, image_width, image_height, 0, g_format, g_type, pImage); // coping decoded frame into the GL texture (synchronous call!)
	}

	return 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *vbo, int size)
{
	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);
	glBufferData(GL_ARRAY_BUFFER, size, NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	OGL_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *vbo)
{
	if (*vbo)
		glDeleteBuffers(1, vbo);
	*vbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Create PBO
////////////////////////////////////////////////////////////////////////////////
void createPBO(GLuint *pbo, int size)
{
	//pBuffer.resize(image_width * image_height * 4);

	//size_t image_pitch = image_width * 4;
	//for (size_t y = 0; y < image_height; y++)
	//{
	//	for (size_t x = 0; x < image_pitch; x += 4)
	//	{
	//		pBuffer[y * image_pitch + x + 0] = 255;
	//		pBuffer[y * image_pitch + x + 1] = 0;
	//		pBuffer[y * image_pitch + x + 2] = 0;
	//		pBuffer[y * image_pitch + x + 3] = 255;
	//	}
	//}

	// create pixel buffer object
	glGenBuffers(1, pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, *pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, size, NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

#ifdef USE_CUDA_SDK
	// register this pbo with CUDA

	cuda_pbo_resource = nullptr;

	if (g_useCuda)
	{
		cudaError cuErr;

		cuErr = cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, *pbo, cudaGraphicsRegisterFlagsNone); __vrcu
		//cuErr = cudaGraphicsResourceSetMapFlags(cuda_pbo_resource, cudaGraphicsMapFlagsWriteDiscard); __vrcu
	}
#endif

	OGL_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete PBO
////////////////////////////////////////////////////////////////////////////////
void deletePBO(GLuint *pbo)
{
	if (*pbo)
		glDeleteBuffers(1, pbo);
	*pbo = 0;
}

void PBO_to_Texture2D(unsigned char* pData, size_t size)
{
	void* pboMemory = nullptr;

	// map PBO and copy data to PBO
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);

	pboMemory = glMapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY);
	if (!pboMemory) return;
	memcpy(pboMemory, pData, size);

	glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	
	// load texture from PBO
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBindTexture(GL_TEXTURE_2D, tex_result);
	//glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	//glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height, g_format, g_type, NULL);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	
	OGL_CHECK_ERROR_GL();
}

#ifdef USE_OPENCL_SDK
int initOpenCLContext()
{
	cl_int error = CL_SUCCESS;
	size_t deviceSize = 0;

	cl_uint numPlatforms = 0;
	cl_platform_id firstPlatformId;

	error = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms); __rcl

	if (numPlatforms == 0)
		return -1;

	cl_platform_id platform = firstPlatformId;

	// Load extension function call
	clGetGLContextInfoKHR_fn clGetGLContextInfoKHR = NULL;

	//clGetGLContextInfoKHR_fn clGetGLContextInfoKHR = (clGetGLContextInfoKHR_fn)clGetExtensionFunctionAddress("clGetGLContextInfoKHR");

	clGetGLContextInfoKHR = (clGetGLContextInfoKHR_fn)clGetExtensionFunctionAddressForPlatform(platform, "clGetGLContextInfoKHR");

	if (!clGetGLContextInfoKHR)
		return -1;

	//Creating the context
#if defined(__APPLE__) || defined(__MACOSX)
	// Apple (untested)
	CGLContextObj kCGLContext = CGLGetCurrentContext();
	CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);
	cl_context_properties contextProperties[] = {
		CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, (cl_context_properties)kCGLShareGroup,
		0 };
#else
#ifdef _WIN32
	// Windows
	cl_context_properties contextProperties[] = {
		CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform),
		CL_GL_CONTEXT_KHR, reinterpret_cast<cl_context_properties>(wglGetCurrentContext()),
		CL_WGL_HDC_KHR, reinterpret_cast<cl_context_properties>(wglGetCurrentDC()),
		0 };
#else
	// Linux
	cl_context_properties contextProperties[] = {
		CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
		CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
		CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
		0 };
#endif
#endif

	// Ask for the CL device associated with the GL context
	error = clGetGLContextInfoKHR(contextProperties, CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR, sizeof(cl_device_id), &device, &deviceSize); __rcl

	// Create the context and the queue
	context = clCreateContext(contextProperties, 1, &device, NULL, NULL, &error); __rcl

	//queue = clCreateCommandQueue(context, device, 0, &error);

	//cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
	//queue = clCreateCommandQueueWithProperties(context, device, props, &error);
	queue = clCreateCommandQueueWithProperties(context, device, NULL, &error); __rcl

	imageCL = clCreateFromGLTexture(context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, tex_result, &error); __rcl
	
	if (error != CL_SUCCESS)
		return -1;

	return 0;
}
#endif

void gpu_initGLBuffers()
{
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

	if (decodeD2->GetImageFormat() == IMAGE_FORMAT_BGRA8BIT || 
		decodeD2->GetImageFormat() == IMAGE_FORMAT_BGRA16BIT || 
		decodeD2->GetImageFormat() == IMAGE_FORMAT_RGBA16BIT)
		g_format = GL_BGRA_EXT;

	// Create texture
	glGenTextures(1, &tex_result);
	glBindTexture(GL_TEXTURE_2D, tex_result);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glTexImage2D(GL_TEXTURE_2D, 0, g_internalFormat, image_width, image_height, 0, g_format, g_type, NULL);

//#if defined(__WIN32__)
	if (g_useCuda || g_useOpenCL)
	{
		if (decodeD2->GetImageFormat() == IMAGE_FORMAT_BGRA8BIT || decodeD2->GetImageFormat() == IMAGE_FORMAT_BGRA16BIT)
		{
			if (program)
			{
				g_bSwapRB = true;
			}
			else // for correct render in this mode version OpenGL must be 3.3 or later
			{
				GLint swizzleMask[] = { GL_BLUE, GL_GREEN, GL_RED, GL_ALPHA };
				glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzleMask);
			}
		}
	}
//#endif

	glBindTexture(GL_TEXTURE_2D, 0);

	OGL_CHECK_ERROR_GL();

	image_size = image_width * image_height * sizeof(GLubyte) * 4;
	
	if (decodeD2->GetImageFormat() == IMAGE_FORMAT_BGRA16BIT ||
		decodeD2->GetImageFormat() == IMAGE_FORMAT_RGBA16BIT)
		image_size *= 2;

	// Create PBO
	createPBO(&pbo, image_size);

	OGL_CHECK_ERROR_GL();

	//OpenGL >= 3.3 core requires a vertex array object containing multiple attribute
	if (program)
	{
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &quadvbo);
		glBindBuffer(GL_ARRAY_BUFFER, quadvbo);
		glBufferData(GL_ARRAY_BUFFER, 6 * 4 * sizeof(float), &quad[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);

		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glGenBuffers(1, &texbo);
		glBindBuffer(GL_ARRAY_BUFFER, texbo);
		glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(float), &texcoord[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);

		glBindBuffer(GL_ARRAY_BUFFER, 0);

		OGL_CHECK_ERROR_GL();

		//slider_pixels.clear();
		//slider_pixels.push_back(0.0f);
		//slider_pixels.push_back(0.5f);

		glGenBuffers(1, &Slider_VBObject);
		glBindBuffer(GL_ARRAY_BUFFER, Slider_VBObject);
		//glBufferData(GL_ARRAY_BUFFER, slider_pixels.size() * sizeof(GLfloat), &slider_pixels[0], GL_DYNAMIC_DRAW);
		glBufferData(GL_ARRAY_BUFFER, 2 * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, 0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		GLint range[2];
		glGetIntegerv(GL_ALIASED_LINE_WIDTH_RANGE, range);
		glGetIntegerv(GL_SMOOTH_LINE_WIDTH_RANGE, range);

		glBindVertexArray(0);

		OGL_CHECK_ERROR_GL();
	}

#ifdef USE_CUDA_SDK
	// register this textures with CUDA

	cuda_tex_result_resource = nullptr;

	if (g_useCuda)
	{
		cudaError cuErr;

		cuErr = cudaGraphicsGLRegisterImage(&cuda_tex_result_resource, tex_result, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore); __vrcu
	}
#endif

	
#ifdef USE_OPENCL_SDK

	context = NULL;
	device = NULL;
	queue = NULL;
	imageCL = NULL;

	if (g_useOpenCL && initOpenCLContext() != 0)
		g_useOpenCL = false;
#endif

	OGL_CHECK_ERROR_GL();

	bytePerPixel = (decodeD2->GetImageFormat() == IMAGE_FORMAT_RGBA8BIT || decodeD2->GetImageFormat() == IMAGE_FORMAT_BGRA8BIT) ? 4 : 8; // RGBA8 or RGBA16
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

#if defined(__CUDAConvertLib__)
	ConvertMatrixCoeff iMatrixCoeff_YUYtoRGBA = (ConvertMatrixCoeff)(pBlock->iMatrixCoeff_YUYtoRGBA);

	IMAGE_FORMAT output_format = decodeD2->GetImageFormat();
	BUFFER_FORMAT buffer_format = decodeD2->GetBufferFormat();

	#define PARAMS pBlock->DataGPUPtr(), texture_ptr, (int)pBlock->Width(), (int)pBlock->Height(), (int)pBlock->Pitch(), NULL, iMatrixCoeff_YUYtoRGBA

	if (buffer_format == BUFFER_FORMAT_RGBA32 || buffer_format == BUFFER_FORMAT_RGBA64)
	{
		cudaMemcpy2DToArray(texture_ptr, 0, 0, pBlock->DataGPUPtr(), pBlock->Pitch(), (pBlock->Width() * bytePerPixel), pBlock->Height(), cudaMemcpyDeviceToDevice); __vrcu
	}
	else if (buffer_format == BUFFER_FORMAT_YUY2)
	{
		if (output_format == IMAGE_FORMAT_RGBA8BIT)
		{
			h_convert_YUY2_to_RGBA32_BtT(PARAMS); __vrcu
		}
		else if (output_format == IMAGE_FORMAT_BGRA8BIT)
		{
			h_convert_YUY2_to_BGRA32_BtT(PARAMS); __vrcu
		}
	}
	else if (buffer_format == BUFFER_FORMAT_Y216)
	{
		if (output_format == IMAGE_FORMAT_RGBA16BIT)
		{
			h_convert_Y216_to_RGBA64_BtT(PARAMS); __vrcu
		}
		else if (output_format == IMAGE_FORMAT_BGRA16BIT)
		{
			h_convert_Y216_to_BGRA64_BtT(PARAMS); __vrcu
		}
		else if (output_format == IMAGE_FORMAT_RGBA8BIT)
		{
			h_convert_Y216_to_RGBA32_BtT(PARAMS); __vrcu
		}
		else if (output_format == IMAGE_FORMAT_BGRA8BIT)
		{
			h_convert_Y216_to_BGRA32_BtT(PARAMS); __vrcu
		}
	}
	else if (buffer_format == BUFFER_FORMAT_NV12)
	{
		if (output_format == IMAGE_FORMAT_RGBA8BIT)
		{
			h_convert_NV12_to_RGBA32_BtT(PARAMS); __vrcu
		}
		else if (output_format == IMAGE_FORMAT_BGRA8BIT)
		{
			h_convert_NV12_to_BGRA32_BtT(PARAMS); __vrcu
		}
	}
	else if (buffer_format == BUFFER_FORMAT_P016)
	{
		if (output_format == IMAGE_FORMAT_RGBA16BIT)
		{
			h_convert_P016_to_RGBA64_BtT(PARAMS); __vrcu
		}
		else if (output_format == IMAGE_FORMAT_BGRA16BIT)
		{
			h_convert_P016_to_BGRA64_BtT(PARAMS); __vrcu
		}
	}
#else
	cudaMemcpy2DToArray(texture_ptr, 0, 0, cuda_dest_resource, pBlock->Pitch(), (pBlock->Width() * bytePerPixel), pBlock->Height(), cudaMemcpyDeviceToDevice); __vrcu
#endif

	cudaGraphicsUnmapResources(1, &cuda_tex_result_resource, 0); __vrcu

	//unsigned char *memPtr;
	//size_t size = 0;
	//cudaGraphicsMapResources(1, &cuda_pbo_resource, 0); __vrcu
	//cudaGraphicsResourceGetMappedPointer((void **)&memPtr, &size, cuda_pbo_resource); __vrcu
	//cudaMemcpy(memPtr, cuda_dest_resource, pBlock->Size(), cudaMemcpyDeviceToDevice); __vrcu
	//cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0); __vrcu

	//glEnable(GL_TEXTURE_2D);
	//glBindTexture(GL_TEXTURE_2D, tex_result);
	//glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	//glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height, g_format, g_type, NULL);
	//glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	//glDisable(GL_TEXTURE_2D);

	//OGL_CHECK_ERROR_GL();

	return 0;
}
#endif

#ifdef USE_OPENCL_SDK
int gpu_generateOpenCLImage(C_Block* pBlock)
{
	unsigned char* pFrameData = pBlock->DataPtr();

	if (g_useCuda)
		pBlock->CopyToCPU();

	IMAGE_FORMAT output_format = decodeD2->GetImageFormat();
	BUFFER_FORMAT buffer_format = decodeD2->GetBufferFormat();

	if (imageCL && pFrameData && g_bCopyToTexture)
	{
		cl_int error = CL_SUCCESS;
		bool is_blockCopy = true;

		// Enqueue Write Image
		size_t origin[] = { 0, 0, 0 };
		size_t region[] = { image_width, image_height, 1 };

		error = clEnqueueAcquireGLObjects(queue, 1, &imageCL, 0, NULL, NULL);
		
		if (buffer_format == BUFFER_FORMAT_RGBA32 || buffer_format == BUFFER_FORMAT_RGBA64)
		{
			error = clEnqueueWriteImage(queue, imageCL, is_blockCopy, origin, region, 0, 0, pFrameData, 0, 0, 0);
			//error = clFlush(queue); // if is_blockCopy == false
			//printf("OCL Image (RGBA)\n");
		}
		else if (buffer_format == BUFFER_FORMAT_YUY2 || buffer_format == BUFFER_FORMAT_Y216)
		{
			region[1] = image_height / 2;
			error = clEnqueueWriteImage(queue, imageCL, is_blockCopy, origin, region, 0, 0, pFrameData, 0, 0, 0);
			//printf("OCL Image (YUY2)\n");
		}

		error = clEnqueueReleaseGLObjects(queue, 1, &imageCL, 0, NULL, NULL);
	}

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
#ifdef USE_OPENCL_SDK
		if (g_useOpenCL)
		{
			if (g_bCopyToTexture)
			{
				gpu_generateOpenCLImage(pBlock);
			}
		}
		else
#endif
	{
		if (g_bCopyToTexture)
		{
			gpu_copyImage(pBlock->DataPtr(), pBlock->Size());
		}
	}

	bRotateFrame = pBlock->GetRotate() ? !bRotateFrame : bRotateFrame; // Rotate frame

	g_bLastRotate = pBlock->GetRotate(); // Save frame rotation value

	iCurPlayFrameNumber = pBlock->iFrameNumber; // Save currect frame number

	decodeD2->UnmapFrame(pBlock); // Add free pointer to queue

	return 0;
}

#if defined(__LINUX__)
int copy_to_framebuffer(unsigned char* pOutput, size_t iSize)
{
    C_AutoLock lock(&g_mutex);

	if (g_bPause && g_fb.size() > 0 && !g_bCopyToFB)
	{
		size_t size_copy = min(g_fb.size(), iSize);
		memcpy(pOutput, g_fb.data(), size_copy);
		return 0;
	}

	if (!decodeD2->isProcess() || decodeD2->isPause()) // check for pause or process
		return 1;

	C_Block *pBlock = decodeD2->MapFrame(); // Get poiter to picture after decoding

	if (!pBlock)
		return -1;

	size_t size_copy = min(pBlock->Size(), iSize);

#ifdef USE_CUDA_SDK
	if (g_useCuda)
	{
		static C_Block devRGBA;
		static long res = 1;
		if (res == 1) res = devRGBA.Init(pBlock->Width(), pBlock->Height(), pBlock->Width() * 4, 0, true);

		if (g_bCopyToTexture && res == 0)
		{
			IMAGE_FORMAT output_format = decodeD2->GetImageFormat();
			BUFFER_FORMAT buffer_format = decodeD2->GetBufferFormat();

#if defined(__CUDAConvertLib__)
			ConvertMatrixCoeff iMatrixCoeff_YUYtoRGBA = (ConvertMatrixCoeff)(pBlock->iMatrixCoeff_YUYtoRGBA);

			#define PARAMS_BtB (void*)pBlock->DataGPUPtr(), (void*)devRGBA.DataGPUPtr(), (int)pBlock->Width(), (int)pBlock->Height(), (int)pBlock->Pitch(), (int)devRGBA.Pitch(), NULL

			if (buffer_format == BUFFER_FORMAT_RGBA32)
			{
				cudaMemcpy2D(pOutput, pBlock->Width() * 4, (void*)pBlock->DataGPUPtr(), pBlock->Pitch(), (pBlock->Width() * 4), pBlock->Height(), cudaMemcpyDeviceToHost); __vrcu
			}
			else
			{
				if (buffer_format == BUFFER_FORMAT_RGBA64)
				{
					h_convert_RGBA64_to_RGBA32_BtB(PARAMS_BtB); __vrcu
				}
				else if (buffer_format == BUFFER_FORMAT_YUY2)
				{
					h_convert_YUY2_to_BGRA32_BtB(PARAMS_BtB, iMatrixCoeff_YUYtoRGBA); __vrcu
				}
				else if (buffer_format == BUFFER_FORMAT_Y216)
				{
					h_convert_Y216_to_BGRA32_BtB(PARAMS_BtB, iMatrixCoeff_YUYtoRGBA); __vrcu
				}

				size_copy = min(devRGBA.Size(), iSize);

				cudaMemcpy(pOutput, devRGBA.DataGPUPtr(), size_copy, cudaMemcpyDeviceToHost); __vrcu
			}
#else
			cudaMemcpy(pOutput, pBlock->DataGPUPtr(), size_copy, cudaMemcpyDeviceToHost); __vrcu
#endif
		}
		//dib_draw::PrintStringToDIB_font8x16<DWORD>((DWORD*)pOutput, 5, 5, (int)pBlock->Width(), g_FpsText, 0xFFE0E0E0, 0XFF102030, 0, 2);
	}
	else
#endif
	{
		if (g_bCopyToTexture)
		{
			//dib_draw::PrintStringToDIB_font8x16<DWORD>((DWORD*)pBlock->DataPtr(), 5, 5, (int)pBlock->Width(), g_FpsText, 0xFFE0E0E0, 0XFF102030, 0, 2);
			memcpy(pOutput, pBlock->DataPtr(), size_copy);
		}
		else
		{
			//dib_draw::PrintStringToDIB_font8x16<DWORD>((DWORD*)pOutput, 5, 5, (int)pBlock->Width(), g_FpsText, 0xFFE0E0E0, 0XFF102030, 0, 2);
		}
	}

	iCurPlayFrameNumber = pBlock->iFrameNumber; // Save currect frame number

	decodeD2->UnmapFrame(pBlock); // Add free pointer to queue

	if ((g_bPause && g_fb.size() == 0) || g_bCopyToFB)
	{
        gpu_copyImage(pOutput, size_copy);
        g_bCopyToFB = false;
	}

	return 0;
}
#endif

void RenderWindow()
{
	C_AutoLock lock(&g_mutex);

	if (g_queuecoord.size() != 0)
	{
		if (g_queuecoord.size() >= 3) g_queuecoord.erase(g_queuecoord.begin(), g_queuecoord.end() - 1);

		mousecoord coord = g_queuecoord.back();	g_queuecoord.pop_back();
		SeekToFrame(coord.x, coord.y, false);
	}

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

	if (program)
	{
		glClearColor(0.0, 0.0, 0.0, 1.0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear the screen

		// viewport
		glViewport(0, 0, (GLsizei)nCoordR, (GLsizei)fBottom);

		//// projection
		//glMatrixMode(GL_PROJECTION);
		//glLoadIdentity();
		//gluPerspective(60.0, (GLfloat)window_width / (GLfloat)window_height, 0.1f, 10.0f);

		// Enable blending
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}
	else
	{
		// Update GL settings
		gpu_UpdateGLSettings();
	}

	if (g_bShowTexture)
	{
		if (program)
		{
			glUseProgramObjectARB(program);
			glUniform1iARB(myTextureSampler, 0);
			glUniform1i(DrawElement, 0);

			if (g_bSwapRB) glUniform1i(SwapRB, 1); else glUniform1i(SwapRB, 0);
			if (bRotate) glUniform1i(RotateID, 1); else glUniform1i(RotateID, 0);
			
			//select geometry to render
			glBindVertexArray(vao);
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, tex_result);

			//draw
			glDrawArrays(GL_TRIANGLES, 0, 6);

			//unbind
			glBindVertexArray(0);

			glUseProgramObjectARB(0);
		}
		else
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
	}

	if (decodeAudio && decodeAudio->IsInitialize())
		decodeAudio->PlayFrame(iCurPlayFrameNumber); // play audio

	if (g_bShowSlider) // draw slider
	{
		GLint w = glutGet(GLUT_WINDOW_WIDTH); // Width in pixels of the current window
		GLint h = glutGet(GLUT_WINDOW_HEIGHT); // Height in pixels of the current window

		sizeSquare2 = (float)w / 100;
        sizeSquare = sizeSquare2 * 2;
		edgeLineY = sizeSquare2 * 4;
		edgeLineX = sizeSquare2 * 2;

		float xCoord = edgeLineX + ((((float)w - (2.f * edgeLineX)) / (float)(iAllFrames - 1)) * (float)iCurPlayFrameNumber);
		float yCoord = (float)h - edgeLineY;

		if (program)
		{
			glUseProgramObjectARB(program);

			/////////////////

			glUniform1i(SwapRB, 0);
			glUniform1i(RotateID, 0);
			glUniform1i(DrawElement, 2);
			glBindVertexArray(vao);

			float x_norm, y_norm;

			y_norm = ((h - yCoord) / (float)h) * 2.0f;

			slider_pixels.clear();

			float heightBox_norm = y_norm * 2 - 1.0f;

			slider_pixels.push_back(-1.0f);
			slider_pixels.push_back(-1.0f);
			slider_pixels.push_back(-1.0f);
			slider_pixels.push_back(heightBox_norm);
			slider_pixels.push_back(1.0f);
			slider_pixels.push_back(-1.0f);

			slider_pixels.push_back(1.0f);
			slider_pixels.push_back(-1.0f);
			slider_pixels.push_back(1.0f);
			slider_pixels.push_back(heightBox_norm);
			slider_pixels.push_back(-1.0f);
			slider_pixels.push_back(heightBox_norm);

			glBindBuffer(GL_ARRAY_BUFFER, Slider_VBObject);
			glBufferData(GL_ARRAY_BUFFER, slider_pixels.size() * sizeof(GLfloat), &slider_pixels[0], GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			glUniform1i(DrawElement, 1);
			glDrawArrays(GL_TRIANGLES, 0, 6);

			slider_pixels.clear();

			if ((xCoord - sizeSquare2) > (0 + edgeLineX))
			{
				x_norm = (edgeLineX / (float)w) * 2.0f;
				slider_pixels.push_back(x_norm - 1.0f);
				slider_pixels.push_back(y_norm - 1.0f);
				x_norm = ((xCoord - sizeSquare2) / (float)w) * 2.0f;
				slider_pixels.push_back(x_norm - 1.0f);
				slider_pixels.push_back(y_norm - 1.0f);
			}

			float x1 = ((xCoord - sizeSquare2) / (float)w) * 2.0f;
			float x2 = ((xCoord + sizeSquare2) / (float)w) * 2.0f;

			float sizeSquare2_norm = (sizeSquare2 / (float)h) * 2.0f;

			float y1 = y_norm - sizeSquare2_norm;
			float y2 = y_norm + sizeSquare2_norm;

			slider_pixels.push_back(x1 - 1.0f);
			slider_pixels.push_back(y1 - 1.0f);
			slider_pixels.push_back(x2 - 1.0f);
			slider_pixels.push_back(y1 - 1.0f);

			slider_pixels.push_back(x2 - 1.0f);
			slider_pixels.push_back(y1 - 1.0f);
			slider_pixels.push_back(x2 - 1.0f);
			slider_pixels.push_back(y2 - 1.0f);

			slider_pixels.push_back(x2 - 1.0f);
			slider_pixels.push_back(y2 - 1.0f);
			slider_pixels.push_back(x1 - 1.0f);
			slider_pixels.push_back(y2 - 1.0f);

			slider_pixels.push_back(x1 - 1.0f);
			slider_pixels.push_back(y2 - 1.0f);
			slider_pixels.push_back(x1 - 1.0f);
			slider_pixels.push_back(y1 - 1.0f);

			if ((xCoord + sizeSquare2) < ((float)w - edgeLineX))
			{
				x_norm = ((xCoord + sizeSquare2) / (float)w) * 2.0f;
				slider_pixels.push_back(x_norm - 1.0f);
				slider_pixels.push_back(y_norm - 1.0f);
				x_norm = (((float)w - edgeLineX) / (float)w) * 2.0f;
				slider_pixels.push_back(x_norm - 1.0f);
				slider_pixels.push_back(y_norm - 1.0f);
			}

			glBindBuffer(GL_ARRAY_BUFFER, Slider_VBObject);
			glBufferData(GL_ARRAY_BUFFER, slider_pixels.size() * sizeof(GLfloat), &slider_pixels[0], GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			//glEnable(GL_LINE_WIDTH);
			//glLineWidth(2); // get error GL_INVALID_VALUE

			glUniform1i(DrawElement, 2);
			glDrawArrays(GL_LINES, 0, (GLsizei)(slider_pixels.size() / 2) - 2);

			glUniform1i(DrawElement, 3);
			glDrawArrays(GL_LINES, (GLsizei)(slider_pixels.size() / 2) - 2, 2);
	
			glBindVertexArray(0);

			/////////////////

			glUseProgramObjectARB(0);
		}
		else
		{
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

		float fCpuload = cpuLoadMeter.GetLoad();

		if (g_bGlutWindow)
		{
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

			if (g_bFullScreen)
				printf("%s: %.0f fps data_rate = %.2f MB/s\n", TITLE_WINDOW_APP, fps, fDataRate);
		}
		else
		{
#if defined(__LINUX__)
			if (g_bFramebuffer)
			{
				//sprintf_s(g_FpsText, "Framebuffer mode: %.0f fps data_rate = %.2f MB/s CPU Load: %.2f%%", fps, fDataRate, fCpuload);

				sprintf_s(textbuf, "%.2f fps %.2f MB/s CPU: %.2f%%", fps, fDataRate, fCpuload);
				dib_draw::PrintStringToDIB_font8x16(textdib, 0, 0, textdib_size, textbuf, -1, 0, 0, 2);
			    text_size_x = strlen(textbuf)*8*2;
                text_size_y = 16*2;

                //sprintf_s(g_FpsText, "%s", textbuf);
            }
			else
#endif
				printf("%s: %.0f fps data_rate = %.2f MB/s\n", TITLE_WINDOW_APP, fps, fDataRate);
		}

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
		if (g_bGlutWindow)
		{
#if defined(__APPLE__)
			exit(0); // On MacOS we have error <Use of undeclared identifier 'glutLeaveMainLoop'> so we call exit(0)
#else
			glutLeaveMainLoop();
#endif
		}
#if defined(__LINUX__)
		else if (g_bFramebuffer)
		{
			g_CLI_bProcess = false;
			exit(0);
		}
#endif
		else
		{
			exit(0);
		}
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

#if defined(__LINUX__)
        if (g_bFramebuffer)
        {
            if (g_bPause)
                g_bCopyToFB = true;
		}
#endif
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

		if (g_bGlutWindow)
		{
			if (g_bFullScreen)
				glutFullScreen(); // if uses freeglut 3.0 and 4K image -> GL error invalid framebuffer operation
			else
			{
				glutPositionWindow(100, 100); // Start position window
				glutReshapeWindow(window_width, window_height); // requests a change to the size of the current window.
			}
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

		if (decodeD2)
			decodeD2->SeekFrame(iCurPlayFrameNumber);

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

	// Delete PBO
	deletePBO(&pbo);

	if (program)
	{
		// Detach and delete the shader objects
		glDetachShader(program, vertexShader);
		glDetachShader(program, fragmentShader);

		glDeleteShader(vertexShader);
		glDeleteShader(fragmentShader);

		glDeleteObjectARB(program);

		glDeleteVertexArrays(1, &vao);
		glDeleteBuffers(1, &quadvbo);
		glDeleteBuffers(1, &texbo);

		glDeleteBuffers(1, &Slider_VBObject);
		slider_pixels.clear();
	}

#ifdef USE_CUDA_SDK
	if (cuda_tex_result_resource)
	{
		cudaGraphicsUnregisterResource(cuda_tex_result_resource); __vrcu
	}
	if (cuda_pbo_resource)
	{
		cudaGraphicsUnregisterResource(cuda_pbo_resource); __vrcu
	}
#endif

#ifdef USE_OPENCL_SDK
	__clerror

	if (imageCL)
	{
		error = clReleaseMemObject(imageCL); __vrcl
	}

	if (queue)
	{
		error = clReleaseCommandQueue(queue); __vrcl
	}

	if (context)
	{
		error = clReleaseContext(context); __vrcl
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
			//SeekToFrame(x, y);
			//RenderWindow(); // update frame to improve performance of scrubbing
			g_queuecoord.push_back(mousecoord(x, y));
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
	if (!g_bGlutWindow)
		return;

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

void SeekToFrame(size_t iFrame, bool bLock)
{
	if (bLock)
	{
		C_AutoLock lock(&g_mutex);
		SeekToFrameImpl(iFrame);
	}
	else
	{
		SeekToFrameImpl(iFrame);
	}
}

void SeekToFrameImpl(size_t iFrame)
{
	//C_AutoLock lock(&g_mutex);

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
					gpu_copyImage(pBlock->DataPtr(), pBlock->Size());
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

void SeekToFrame(int x, int y, bool bLock)
{
    int w, h;

    if (g_bGlutWindow)
    {
        w = glutGet(GLUT_WINDOW_WIDTH); // Width in pixels of the current window
        h = glutGet(GLUT_WINDOW_HEIGHT); // Height in pixels of the current window
    }
#if defined(__LINUX__)
    else if (g_bFramebuffer)
    {
        w = g_var_info.xres;
        h = g_var_info.yres;
    }
#endif
    else return;

	sizeSquare2 = (float)w / 100;
	sizeSquare = sizeSquare2 * 2;
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

		SeekToFrame(iFrame, bLock); // Seek to frame number <iFrame>
	}
}

