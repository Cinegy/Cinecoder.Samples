#pragma once

#include "BaseGPURender.h"

class GPURenderGL : public BaseGPURender
{
private:
	HGLRC glContext;

	GLuint tex_result;  // texture

	GLuint pbo; // OpenGL pixel buffer object

	float start_ortho_w;
	float start_ortho_h;

	float stop_ortho_w;
	float stop_ortho_h;

	int nViewportWidth;
	int nViewportHeight;

	GLint internalFormat;
	GLenum format;
	GLenum type;

public:
	GPURenderGL();
	virtual ~GPURenderGL();

private:
	virtual int RenderWindow();
	virtual int InitRender();
	virtual int DestroyRender();
	virtual int GenerateImage(bool & bRotateFrame);
	virtual int CopyBufferToTexture(C_Block *pBlock);
	virtual int SetVerticalSync(bool bVerticalSync);

private:
	BOOL CreateGL();
	
	double getVersionGL();

	void createPBO(GLuint *pbo, int size);
	void deletePBO(GLuint *pbo);

	int gpu_InitGLBuffers();
	int gpu_DestroyGLBuffers();
	int gpu_UpdateGLSettings();
	int gpu_CopyImage(unsigned char* pImage, size_t iSize);
};

