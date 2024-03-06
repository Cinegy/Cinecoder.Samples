// Memory types used in benchmark
enum MemType { MEM_SYSTEM, MEM_PINNED, MEM_GPU };
MemType g_mem_type = MEM_SYSTEM;

struct memobj_t
{
	void*	Ptr;
	size_t	Size;
	MemType	Type;

	operator bool()  const { return Ptr != NULL; }
    operator PBYTE() const { return PBYTE(Ptr);  }
};

memobj_t MK_MEMOBJ(MemType type, size_t size, void* ptr)
{
	memobj_t obj = { ptr, size, type };
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
    void *ptr = nullptr;

    if(auto err = cuMemAllocHost(&ptr, size))
      fprintf(stderr, "cuMemAllocHost(%zd) error %d (%s)\n", size, err, GetCudaDrvApiErrorText(err));

    return MK_MEMOBJ(type, size, ptr);
  }

  if(type == MEM_GPU)
  {
    void *ptr = nullptr;

    if(auto err = cuMemAlloc((CUdeviceptr*)&ptr, size))
      fprintf(stderr, "cuMemAlloc() error %d (%s)\n", err, GetCudaDrvApiErrorText(err));

    return MK_MEMOBJ(type, size, ptr);
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
	if(auto err = cuMemFreeHost(obj.Ptr))
      fprintf(stderr, "cuMemFreeHost() error %d (%s)\n", err, GetCudaDrvApiErrorText(err));
  }

  if(obj.Type == MEM_GPU)
  {
	if(auto err = cuMemFree((CUdeviceptr)obj.Ptr))
      fprintf(stderr, "cuMemFree() error %d (%s)\n", err, GetCudaDrvApiErrorText(err));
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

  if(auto err = cuMemcpyHtoD((CUdeviceptr)dst.Ptr, src_ptr, size))
    return fprintf(stderr, "cuMemcpyHtoD(%zd) error %d (%s)\n", size, err, GetCudaDrvApiErrorText(err)), err;

  return 0;
}

int mem_copy(void *dst_ptr, memobj_t &src, size_t size)
{
  if(src.Type == MEM_SYSTEM || src.Type == MEM_PINNED)
  {
    memcpy(dst_ptr, src.Ptr, size);
    return 0;
  }

  if(auto err = cuMemcpyDtoH(dst_ptr, (CUdeviceptr)src.Ptr, size))
    return fprintf(stderr, "cuMemcpyDtoH(%zd) error %d (%s)\n", size, err, GetCudaDrvApiErrorText(err)), err;

  return 0;
}
