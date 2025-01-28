// Memory types used in benchmark
enum MemType { MEM_SYSTEM, MEM_PINNED, MEM_GPU };
MemType g_mem_type = MEM_SYSTEM;

struct memobj_t
{
	void*	Ptr;
	void*	OrgPtr;
	size_t	Size;
	MemType	Type;

	operator bool()  const { return Ptr != NULL; }
    operator PBYTE() const { return PBYTE(Ptr);  }
};

memobj_t MK_MEMOBJ(MemType type, size_t size, void* ptr, void *org_ptr = nullptr)
{
	memobj_t obj = { ptr, org_ptr, size, type };
	return obj;
}

memobj_t mem_alloc(MemType type, size_t size)
{
  if(type == MEM_SYSTEM)
  {
#ifdef _WIN32
    BYTE *ptr = (BYTE*)VirtualAlloc(NULL, size + 2*4096, MEM_COMMIT, PAGE_READWRITE);
    ptr += 4096 - (size & 4095);
    DWORD oldf;
    VirtualProtect(ptr + size, 4096, PAGE_NOACCESS, &oldf);
    return MK_MEMOBJ(type, size, ptr);
#elif defined(__APPLE__)
    return MK_MEMOBJ(type, size, malloc(size));
#elif defined(__ANDROID__)
	void *ptr = nullptr;
	posix_memalign(&ptr, 4096, size);
    return MK_MEMOBJ(type, size, ptr);
#else
    return MK_MEMOBJ(type, size, aligned_alloc(4096, size));
#endif		
  }

  if(type == MEM_PINNED)
  {
    if(g_cudaContext)
    {
      void *ptr = nullptr;

	  if(auto err = cuMemAllocHost(&ptr, size))
        return MK_MEMOBJ(type, size, (fprintf(stderr, "cuMemAllocHost(%zd) error %d (%s)\n", size, err, GetCudaDrvApiErrorText(err)), nullptr));

      return MK_MEMOBJ(type, size, ptr);
    }

    else if(g_clContext)
    {
      cl_int err;
	  
	  auto clbuf = clCreateBuffer(g_clContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size, NULL, &err);
	  if(err != CL_SUCCESS)
        return MK_MEMOBJ(type, size, (fprintf(stderr, "clCreateBuffer(%zd) error %d (%s)\n", size, err, GetOpenClErrorText(err)), nullptr));

      auto ptr = clEnqueueMapBuffer(g_clMemAllocQueue, clbuf, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, size, 0, NULL, NULL, &err);
      if(err != CL_SUCCESS)
        return MK_MEMOBJ(type, size, (fprintf(stderr, "clEnqueueMapBuffer(%zd) error %d (%s)\n", size, err, GetOpenClErrorText(err)), nullptr));

      if(auto err = clFinish(g_clMemAllocQueue))
        return MK_MEMOBJ(type, size, (fprintf(stderr, "clFinish() error %d (%s)\n", err, GetOpenClErrorText(err)), nullptr));

      return MK_MEMOBJ(type, size, ptr, clbuf);
    }

#ifdef __APPLE__
    else if(g_metalDevice)
    {
      auto buf = ((MTL::Device*)g_metalDevice)->newBuffer(size, MTL::ResourceStorageModeShared);

	  if(!buf)
        return MK_MEMOBJ(type, size, (fprintf(stderr, "newBuffer(%zd) allocation error\n", size), nullptr));

      return MK_MEMOBJ(type, size, buf->contents(), buf);
    }
#endif
    return MK_MEMOBJ(type, size, (fprintf(stderr, "GPU context is not set\n"), nullptr));
  }

  if(type == MEM_GPU)
  {
    if(g_cudaContext)
    {
      void *ptr = nullptr;

	  if(auto err = cuMemAlloc((CUdeviceptr*)&ptr, size))
        return MK_MEMOBJ(type, size, (fprintf(stderr, "cuMemAlloc() error %d (%s)\n", err, GetCudaDrvApiErrorText(err)), nullptr));

      return MK_MEMOBJ(type, size, ptr);
    }

    else if(g_clContext)
    {
      cl_int err;
	  
	  auto clbuf = clCreateBuffer(g_clContext, CL_MEM_READ_WRITE, size, NULL, &err);
	  if(err != CL_SUCCESS)
        return MK_MEMOBJ(type, size, (fprintf(stderr, "clCreateBuffer(%zd) error %d (%s)\n", size, err, GetOpenClErrorText(err)), nullptr));

      return MK_MEMOBJ(type, size, clbuf);
    }

#ifdef __APPLE__
    else if(g_metalDevice)
    {
      auto buf  = ((MTL::Device*)g_metalDevice)->newBuffer(size, MTL::ResourceStorageModeShared);

	  if(!buf)
        return MK_MEMOBJ(type, size, (fprintf(stderr, "newBuffer(%zd) allocation error\n", size), nullptr));

      return MK_MEMOBJ(type, size, buf);
    }
#endif

    return MK_MEMOBJ(type, size, (fprintf(stderr, "GPU context is not set\n"), nullptr));
  }

  return MK_MEMOBJ(MEM_SYSTEM, 0, nullptr);
}

void mem_release(memobj_t &obj)
{
  if(!obj)
    return;

  if(obj.Type == MEM_SYSTEM)
  {
#ifdef _WIN32
	VirtualFree(obj.Ptr, 0, MEM_RELEASE);
#else
	free(obj.Ptr);
#endif		
  }

  if(obj.Type == MEM_PINNED)
  {
    if(g_cudaContext)
    {
	  if(auto err = cuMemFreeHost(obj.Ptr))
      {
        fprintf(stderr, "cuMemFreeHost() error %d (%s)\n", err, GetCudaDrvApiErrorText(err));
        return;
      }
    }
    else if(g_clContext)
    {
      if (auto err = clEnqueueUnmapMemObject(g_clMemAllocQueue, (cl_mem)obj.OrgPtr, obj.Ptr, 0, 0, 0))
      {
        fprintf(stderr, "clEnqueueUnmapMemObject() error %d (%s)\n", err, GetOpenClErrorText(err));
        return;
      }

      if (auto err = clFinish(g_clMemAllocQueue))
      {
        fprintf(stderr, "clFinish() error %d (%s)\n", err, GetOpenClErrorText(err));
        return;
      }

	  if(auto err = clReleaseMemObject((cl_mem)obj.OrgPtr))
      {
        fprintf(stderr, "clReleaseMemObject() error %d (%s)\n", err, GetOpenClErrorText(err));
        return;
      }
	}
#ifdef __APPLE__
    else if(g_metalDevice)
    {
      ((MTL::Buffer*)obj.OrgPtr)->release();
    }
#endif
  }

  if(obj.Type == MEM_GPU)
  {
    if(g_cudaContext)
    {
	  if(auto err = cuMemFree((CUdeviceptr)obj.Ptr))
        fprintf(stderr, "cuMemFree() error %d (%s)\n", err, GetCudaDrvApiErrorText(err));

      return;
    }

    else if(g_clContext)
    {
	  if(auto err = clReleaseMemObject((cl_mem)obj.Ptr))
      {
        fprintf(stderr, "clReleaseMemObject() error %d (%s)\n", err, GetOpenClErrorText(err));
        return;
      }
    }

#ifdef __APPLE__
    else if(g_metalDevice)
    {
      ((MTL::Buffer*)obj.Ptr)->release();
    }
#endif
  }

  memset(&obj, 0, sizeof(obj));

  return;
}

int mem_copy(memobj_t &dst, const void *src_ptr, size_t size)
{
  if(dst.Type == MEM_SYSTEM || dst.Type == MEM_PINNED)
  {
    memcpy(dst.Ptr, src_ptr, size);
    return 0;
  }

  if(g_cudaContext)
  {
	if(auto err = cuMemcpyHtoD((CUdeviceptr)dst.Ptr, src_ptr, size))
	  return fprintf(stderr, "cuMemcpyHtoD(%zd) error %d (%s)\n", size, err, GetCudaDrvApiErrorText(err)), err;

    return 0;
  }	

  if(g_clContext)
  {
    if(auto err = clEnqueueWriteBuffer(g_clMemAllocQueue, (cl_mem)dst.Ptr, CL_TRUE, 0, size, src_ptr, 0, 0, 0))
      return fprintf(stderr, "clEnqueueWriteBuffer(%zd) failed with code %d(%s)", size, err, GetOpenClErrorText(err)), err;

    if(auto err = clFinish(g_clMemAllocQueue))
      return fprintf(stderr, "clFinish() failed with code %d(%s)", err, GetOpenClErrorText(err)), err;

    return 0;
  }

#ifdef __APPLE__
  if(g_metalDevice)
  {
    memcpy(((MTL::Buffer*)dst.Ptr)->contents(), src_ptr, size);
    return 0;
  }
#endif

  return fprintf(stderr, "GPU context is not set\n"), E_UNEXPECTED;
}

int mem_copy(void *dst_ptr, memobj_t &src, size_t size)
{
  if(src.Type == MEM_SYSTEM || src.Type == MEM_PINNED)
  {
    memcpy(dst_ptr, src.Ptr, size);
    return 0;
  }

  if(g_cudaContext)
  {
	if(auto err = cuMemcpyDtoH(dst_ptr, (CUdeviceptr)src.Ptr, size))
	  return fprintf(stderr, "cuMemcpyDtoH(%zd) error %d (%s)\n", size, err, GetCudaDrvApiErrorText(err)), err;

    return 0;
  }	

  if(g_clContext)
  {
    if(auto err = clEnqueueReadBuffer(g_clMemAllocQueue, (cl_mem)src.Ptr, CL_TRUE, 0, size, dst_ptr, 0, 0, 0))
      return fprintf(stderr, "clEnqueueReadBuffer(%zd) failed with code %d(%s)", size, err, GetOpenClErrorText(err)), err;

    if(auto err = clFinish(g_clMemAllocQueue))
      return fprintf(stderr, "clFinish() failed with code %d(%s)", err, GetOpenClErrorText(err)), err;

    return 0;
  }

#ifdef __APPLE__
  if(g_metalDevice)
  {
    memcpy(dst_ptr, ((MTL::Buffer*)src.Ptr)->contents(), size);
    return 0;
  }
#endif

  return fprintf(stderr, "GPU context is not set\n"), E_UNEXPECTED;
}
