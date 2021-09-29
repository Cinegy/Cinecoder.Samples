#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef cl_uint     cl_gl_object_type;
typedef cl_uint     cl_gl_texture_info;
typedef cl_uint     cl_gl_platform_info;
typedef struct __GLsync *cl_GLsync;

/* cl_gl_object_type = 0x2000 - 0x200F enum values are currently taken           */
#define CL_GL_OBJECT_BUFFER                     0x2000
#define CL_GL_OBJECT_TEXTURE2D                  0x2001
#define CL_GL_OBJECT_TEXTURE3D                  0x2002
#define CL_GL_OBJECT_RENDERBUFFER               0x2003
#ifdef CL_VERSION_1_2
#define CL_GL_OBJECT_TEXTURE2D_ARRAY            0x200E
#define CL_GL_OBJECT_TEXTURE1D                  0x200F
#define CL_GL_OBJECT_TEXTURE1D_ARRAY            0x2010
#define CL_GL_OBJECT_TEXTURE_BUFFER             0x2011
#endif

/* cl_gl_texture_info           */
#define CL_GL_TEXTURE_TARGET                    0x2004
#define CL_GL_MIPMAP_LEVEL                      0x2005
#ifdef CL_VERSION_1_2
#define CL_GL_NUM_SAMPLES                       0x2012
#endif


typedef CL_API_ENTRY cl_mem (CL_API_CALL *FTclCreateFromGLBuffer)(
	cl_context     context,
	cl_mem_flags   flags,
	cl_GLuint      bufobj,
	cl_int *       errcode_ret) CL_API_SUFFIX__VERSION_1_0;

#ifdef CL_VERSION_1_2

typedef CL_API_ENTRY cl_mem (CL_API_CALL *FTclCreateFromGLTexture)(
	cl_context      context,
	cl_mem_flags    flags,
	cl_GLenum       target,
	cl_GLint        miplevel,
	cl_GLuint       texture,
	cl_int *        errcode_ret) CL_API_SUFFIX__VERSION_1_2;

#endif

typedef CL_API_ENTRY cl_mem (CL_API_CALL *FTclCreateFromGLRenderbuffer)(
	cl_context   context,
	cl_mem_flags flags,
	cl_GLuint    renderbuffer,
	cl_int *     errcode_ret) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *FTclGetGLObjectInfo)(
	cl_mem                memobj,
	cl_gl_object_type *   gl_object_type,
	cl_GLuint *           gl_object_name) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *FTclGetGLTextureInfo)(
	cl_mem               memobj,
	cl_gl_texture_info   param_name,
	size_t               param_value_size,
	void *               param_value,
	size_t *             param_value_size_ret) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *FTclEnqueueAcquireGLObjects)(
	cl_command_queue      command_queue,
	cl_uint               num_objects,
	const cl_mem *        mem_objects,
	cl_uint               num_events_in_wait_list,
	const cl_event *      event_wait_list,
	cl_event *            event) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *FTclEnqueueReleaseGLObjects)(
	cl_command_queue      command_queue,
	cl_uint               num_objects,
	const cl_mem *        mem_objects,
	cl_uint               num_events_in_wait_list,
	const cl_event *      event_wait_list,
	cl_event *            event) CL_API_SUFFIX__VERSION_1_0;


/* Deprecated OpenCL 1.1 APIs */
typedef CL_API_ENTRY CL_API_PREFIX__VERSION_1_1_DEPRECATED cl_mem (CL_API_CALL *FTclCreateFromGLTexture2D)(
	cl_context      context,
	cl_mem_flags    flags,
	cl_GLenum       target,
	cl_GLint        miplevel,
	cl_GLuint       texture,
	cl_int *        errcode_ret) CL_API_SUFFIX__VERSION_1_1_DEPRECATED;

typedef CL_API_ENTRY CL_API_PREFIX__VERSION_1_1_DEPRECATED cl_mem (CL_API_CALL *FTclCreateFromGLTexture3D)(
	cl_context      context,
    cl_mem_flags    flags,
    cl_GLenum       target,
    cl_GLint        miplevel,
    cl_GLuint       texture,
    cl_int *        errcode_ret) CL_API_SUFFIX__VERSION_1_1_DEPRECATED;

/* cl_khr_gl_sharing extension  */

#define cl_khr_gl_sharing 1

typedef cl_uint     cl_gl_context_info;

/* Additional Error Codes  */
#define CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR  -1000

/* cl_gl_context_info  */
#define CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR    0x2006
#define CL_DEVICES_FOR_GL_CONTEXT_KHR           0x2007

/* Additional cl_context_properties  */
#define CL_GL_CONTEXT_KHR                       0x2008
#define CL_EGL_DISPLAY_KHR                      0x2009
#define CL_GLX_DISPLAY_KHR                      0x200A
#define CL_WGL_HDC_KHR                          0x200B
#define CL_CGL_SHAREGROUP_KHR                   0x200C

typedef CL_API_ENTRY cl_int (CL_API_CALL *FTclGetGLContextInfoKHR)(
	const cl_context_properties * properties,
    cl_gl_context_info            param_name,
    size_t                        param_value_size,
    void *                        param_value,
    size_t *                      param_value_size_ret) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int (CL_API_CALL *clGetGLContextInfoKHR_fn)(
    const cl_context_properties * properties,
    cl_gl_context_info            param_name,
    size_t                        param_value_size,
    void *                        param_value,
    size_t *                      param_value_size_ret);

/* 
 *  cl_khr_gl_event extension
 */
#define CL_COMMAND_GL_FENCE_SYNC_OBJECT_KHR     0x200D

typedef CL_API_ENTRY cl_event (CL_API_CALL *FTclCreateEventFromGLsyncKHR)(
	cl_context context,
    cl_GLsync  sync,
    cl_int *   errcode_ret) CL_API_SUFFIX__VERSION_1_1;

#ifdef __cplusplus
}
#endif
