#include "stdafx.h"
#include "GPURenderGL.h"

typedef BOOL(WINAPI *PFNWGLSWAPINTERVALEXTPROC)(int interval);
PFNWGLSWAPINTERVALEXTPROC wglSwapInterval;

#define GL_CLAMP_TO_EDGE 0x812F
#define GL_UNSIGNED_INT_2_10_10_10_REV 0x8368
#define GL_UNSIGNED_INT_10_10_10_2 0x8036

#define GL_TEXTURE_SWIZZLE_RGBA 0x8E46

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

GPURenderGL::GPURenderGL()
{
	start_ortho_w = 0;
	start_ortho_h = 0;

	stop_ortho_w = (float)window_width;
	stop_ortho_h = (float)window_height;

	nViewportWidth = window_width;
	nViewportHeight = window_height;

	internalFormat = GL_RGBA;
	format = GL_BGRA_EXT;
	type = GL_UNSIGNED_BYTE;

	m_windowCaption = L"TestApp (Decode Daniel2) GL"; // Set window caption

	gpu_render_type = GPU_RENDER_OPENGL;

	pbo = 0;
}

GPURenderGL::~GPURenderGL()
{
}

int GPURenderGL::GenerateImage(bool & bRotateFrame)
{
	if (!m_decodeD2->isProcess() || m_decodeD2->isPause()) // check for pause or process
		return 1;

	C_Block *pBlock = m_decodeD2->MapFrame(); // Get poiter to picture after decoding

	if (!pBlock)
		return -1;

	if (m_bCopyToTexture)
	{
		gpu_CopyImage(pBlock->DataPtr(), pBlock->Size());
	}

	bRotateFrame = pBlock->GetRotate() ? !bRotateFrame : bRotateFrame; // Rotate frame

	m_bLastRotate = pBlock->GetRotate(); // Save frame rotation value

	iCurPlayFrameNumber = pBlock->iFrameNumber; // Save currect frame number

	m_decodeD2->UnmapFrame(pBlock); // Add free pointer to queue

	return 0;
}

int GPURenderGL::CopyBufferToTexture(C_Block *pBlock)
{
#ifdef USE_CUDA_SDK
	if (m_bUseGPU)
	{
		if (m_bCopyToTexture)
		{
			CopyCUDAImage(pBlock);
		}
	}
	else
#endif
	{
		if (m_bCopyToTexture)
		{
			gpu_CopyImage(pBlock->DataPtr(), pBlock->Size());
		}
	}

	return 0;
}

int GPURenderGL::gpu_CopyImage(unsigned char* pImage, size_t iSize)
{
	if (pbo)
	{
		void* pboMemory = nullptr;

		// map PBO and copy data to PBO
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);

		pboMemory = glMapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY);
		if (!pboMemory) return -1;
		memcpy(pboMemory, pImage, iSize);

		glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

		// load texture from PBO
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
		glBindTexture(GL_TEXTURE_2D, tex_result);
		//glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		//glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height, format, type, NULL);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

		OGL_CHECK_ERROR_GL();
	}
	else if (pImage)
	{
		glBindTexture(GL_TEXTURE_2D, tex_result);
		glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, image_width, image_height, 0, format, type, pImage); // copying decoded frame into the GL texture
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	return 0;
}

int GPURenderGL::RenderWindow()
{
	C_AutoLock lock(&m_mutex);

	//////////////////////////////////////

	if (!m_bVSync && !m_bMaxFPS && m_bVSyncHand)
	{
		double timestep = 1000.0 / ValueFPS;

		double ms_elapsed = timerqFPSMode.GetElapsedTime();

		if (ms_elapsed < timestep)
		{
			return 0;
		}
		else
		{
			timerqFPSMode.StartTimer();
		}
	}

	bool bRotate = m_bRotate;

	int res = 1;

	if (!m_bPause)
	{
		if (m_bUseGPU)
			res = GenerateCUDAImage(bRotate); // Copy data from device to device(array)
		else
			res = GenerateImage(bRotate); // Copy data from host to device

		if (res < 0)
			printf("Load texture from decoder failed!\n");
	}
	else
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(100)); // for unload CPU when set pause
	}

	if (res != 0)
	{
		bRotate = m_bLastRotate ? !bRotate : bRotate; // Rotate frame
	}

	if (m_decodeAudio && m_decodeAudio->IsInitialize())
		m_decodeAudio->PlayFrame(iCurPlayFrameNumber); // play audio

	//////////////////////////////////////

	float nCoordL = 0.0f;
	float nCoordR = (float)window_width;
	float fBottom = (float)window_height;
	float fTop = 0.0f;

	// Update GL settings
	gpu_UpdateGLSettings();

	if (m_bShowTexture)
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
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	if (m_bShowSlider) // draw slider
	{
		size_t w = window_width; // Width in pixels of the current window
		size_t h = window_height; // Height in pixels of the current window

		sizeSquare2 = (float)w / 100;
		edgeLineY = sizeSquare2 * 4;
		edgeLineX = sizeSquare2 * 2;

		float xCoord = edgeLineX; // edgeLineX + ((((float)w - (2.f * edgeLineX)) / (float)(iAllFrames - 1)) * (float)iCurPlayFrameNumber);
		float yCoord = (float)h - edgeLineY;

		if (iAllFrames > 1)
			xCoord += ((((float)w - (2.f * edgeLineX)) / (float)(iAllFrames - 1)) * (float)iCurPlayFrameNumber);

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

	SwapBuffers(m_hDC);

	OGL_CHECK_ERROR_GL();

	ComputeFPS(); // Calculate fps

	return 0;
}

int GPURenderGL::InitRender()
{
	if (CreateGL() != TRUE)
		return -1;

	ShowWindow(m_hWnd, SW_SHOWDEFAULT);

	return 0;
}

int GPURenderGL::DestroyRender()
{
	gpu_DestroyGLBuffers();

	if (glContext)
	{
		wglMakeCurrent(NULL, NULL);
		wglDeleteContext(glContext);
	}

	return 0;
}

int GPURenderGL::SetVerticalSync(bool bVerticalSync)
{
	if (!wglSwapInterval)
	{
		wglSwapInterval = (PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT");

		if (!wglSwapInterval)
			wglSwapInterval = (PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapInterval");
	}

	if (wglSwapInterval)
		wglSwapInterval(bVerticalSync);

	return 0;
}

double GPURenderGL::getVersionGL()
{
	char *versionGL = nullptr;
	versionGL = (char *)(glGetString(GL_VERSION));
	if (!versionGL) { printf("Error: canot get OpenGL version\n");  return -1;	}

	std::string strVersion = versionGL;
	strVersion = strVersion.substr(0, strVersion.find(" "));
	double number = std::atof(strVersion.c_str()) + 0.005;

	printf("OpenGL version: %s\n", versionGL);
	printf("-------------------------------------\n");

	return number;
}

BOOL GPURenderGL::CreateGL()
{
	HDC hdc = m_hDC;

	//////////////////////////////////////

	PIXELFORMATDESCRIPTOR PixFormatDesc;

	memset(&PixFormatDesc, 0, sizeof(PixFormatDesc));
	PixFormatDesc.nSize = sizeof(PixFormatDesc);
	PixFormatDesc.nVersion = 1;
	PixFormatDesc.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER | PFD_DEPTH_DONTCARE;

	PixFormatDesc.iPixelType = PFD_TYPE_RGBA;
	PixFormatDesc.cColorBits = 32;

	PixFormatDesc.cRedBits = 8;
	PixFormatDesc.cGreenBits = 8;
	PixFormatDesc.cBlueBits = 8;
	PixFormatDesc.cAlphaBits = 8;

	//////////////////////////////////////

	int PixFormat = ChoosePixelFormat(hdc, &PixFormatDesc);

	if (PixFormat)
	{
		if (!SetPixelFormat(hdc, PixFormat, &PixFormatDesc))
		{
			_assert(0);
			DWORD le = GetLastError();
			return -1;
		}
	}

	//////////////////////////////////////

	HGLRC pglC = wglCreateContext(hdc);

	if (!pglC)
	{
		_assert(0);
		return GetLastError();
	}

	glContext = pglC;

	//////////////////////////////////////

	if (!wglMakeCurrent(hdc, pglC))
	{
		_assert(0);
		return GetLastError();
	}

	SetVerticalSync(m_bVSync);

	//////////////////////////////////////

	if (gpu_InitGLBuffers() != 0)
		return FALSE;

	m_bInitRender = true;

	return TRUE;
}

void GPURenderGL::createPBO(GLuint *pbo, int size)
{
	// create pixel buffer object
	glGenBuffers(1, pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, *pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, size, NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	OGL_CHECK_ERROR_GL();
}

void GPURenderGL::deletePBO(GLuint *pbo)
{
	if (*pbo)
		glDeleteBuffers(1, pbo);
	*pbo = 0;
}

int GPURenderGL::gpu_InitGLBuffers()
{
	// Create texture that will receive the result of CUDA

	glGenTextures(1, &tex_result);
	glBindTexture(GL_TEXTURE_2D, tex_result);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	if (m_decodeD2->GetImageFormat() == IMAGE_FORMAT_RGBA8BIT || m_decodeD2->GetImageFormat() == IMAGE_FORMAT_BGRA8BIT) // RGBA 8 bit
	{
		internalFormat = GL_RGBA;
		type = GL_UNSIGNED_BYTE;
		//g_format = GL_RGBA;      // this one is 2x faster
		//if (m_decodeD2->GetImageFormat() == IMAGE_FORMAT_BGRA8BIT) format = GL_RGBA;
	}
	else if (m_decodeD2->GetImageFormat() == IMAGE_FORMAT_RGB30) // R10G10B10A2 fromat
	{
		internalFormat = GL_RGB10;
		format = GL_RGBA;
		//g_type = GL_UNSIGNED_INT_10_10_10_2;
		type = GL_UNSIGNED_INT_2_10_10_10_REV;	// this one is 2x faster
	}
	else if (m_decodeD2->GetImageFormat() == IMAGE_FORMAT_RGBA16BIT || m_decodeD2->GetImageFormat() == IMAGE_FORMAT_BGRA16BIT) // RGBA 16 bit
	{
		internalFormat = GL_RGBA16;
		type = GL_UNSIGNED_SHORT;
		//if (m_decodeD2->GetImageFormat() == IMAGE_FORMAT_BGRA16BIT) format = GL_RGBA;
	}
	else
	{
		printf("Image format is invalid!\n");
	}

	format = GL_RGBA;

	if (m_decodeD2->GetImageFormat() == IMAGE_FORMAT_BGRA8BIT ||
		m_decodeD2->GetImageFormat() == IMAGE_FORMAT_BGRA16BIT ||
		m_decodeD2->GetImageFormat() == IMAGE_FORMAT_RGBA16BIT)
		format = GL_BGRA_EXT;

	glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, image_width, image_height, 0, format, type, NULL);

	double versionGL = getVersionGL();

	if (m_bUseGPU)
	{
		if (m_decodeD2->GetImageFormat() == IMAGE_FORMAT_BGRA8BIT || m_decodeD2->GetImageFormat() == IMAGE_FORMAT_BGRA16BIT)
		{
			if (versionGL < 3.3)
			{
				printf("Error: for correct render in this mode version OpenGL must be 3.3 or later)\n");
				return -1;
			}

			GLint swizzleMask[] = { GL_BLUE, GL_GREEN, GL_RED, GL_ALPHA };
			glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzleMask);
		}
	}

	glBindTexture(GL_TEXTURE_2D, 0);

	OGL_CHECK_ERROR_GL();

	image_size = image_width * image_height * sizeof(GLubyte) * 4;

	if (m_decodeD2->GetImageFormat() == IMAGE_FORMAT_BGRA16BIT ||
		m_decodeD2->GetImageFormat() == IMAGE_FORMAT_RGBA16BIT)
		image_size *= 2;

	glewExperimental = GL_TRUE;
	GLenum GlewInitResult = glewInit();

	if (GlewInitResult != GLEW_OK) {
		// Problem: glewInit failed, something is seriously wrong.
		printf("glewInit failed: %s\n", glewGetErrorString(GlewInitResult));
		exit(1);
	}

	OGL_CHECK_ERROR_GL();

	// Create PBO
	createPBO(&pbo, image_size); // need use GLEW Library => glewInit()

	OGL_CHECK_ERROR_GL();

	if (m_bUseGPU)
	{
		// Register this texture with CUDA
		cudaGraphicsGLRegisterImage(&cuda_tex_result_resource, tex_result, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore); __vrcu
	}

	OGL_CHECK_ERROR_GL();

	bytePerPixel = (m_decodeD2->GetImageFormat() == IMAGE_FORMAT_RGBA8BIT || m_decodeD2->GetImageFormat() == IMAGE_FORMAT_BGRA8BIT) ? 4 : 8; // RGBA8 or RGBA16
	size_tex_data = sizeof(GLubyte) * image_width * image_height * bytePerPixel;

	return 0;
}

int GPURenderGL::gpu_DestroyGLBuffers()
{
	if (m_bUseGPU)
	{
		// Unregister resource with CUDA
		if (cuda_tex_result_resource)
		{
			cudaGraphicsUnregisterResource(cuda_tex_result_resource); __vrcu
		}
	}

	// Delete GL texture
	glDeleteTextures(1, &tex_result);

	OGL_CHECK_ERROR_GL();

	// Delete PBO
	deletePBO(&pbo);

	OGL_CHECK_ERROR_GL();

	return 0;
}

int GPURenderGL::gpu_UpdateGLSettings()
{
	stop_ortho_w = static_cast<float>(window_width);
	stop_ortho_h = static_cast<float>(window_height);

	nViewportWidth = window_width;
	nViewportHeight = window_height;

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f); // Clear buffer
	glEnable(GL_TEXTURE_2D); // Allow texture mapping

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(start_ortho_w, stop_ortho_w, stop_ortho_h, start_ortho_h, -1, 1);

	glMatrixMode(GL_MODELVIEW);
	glViewport(0, 0, nViewportWidth, nViewportHeight);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	glClear(GL_COLOR_BUFFER_BIT);

	// initialization for transparency
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glClearColor(0.0, 0.0, 0.0, 0.0);

	glClearColor(1.0f, 1.0f, 0.0f, 1.0f); // Clear buffer

	return 0;
}
