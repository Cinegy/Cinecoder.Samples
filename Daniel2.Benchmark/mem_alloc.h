// Memory types used in benchmark
enum MemType { MEM_SYSTEM, MEM_PINNED, MEM_GPU };
MemType g_mem_type = MEM_SYSTEM;

void* mem_alloc(MemType type, size_t size)
{
  if(type == MEM_SYSTEM)
  {
#ifdef _WIN32
    BYTE *ptr = (BYTE*)VirtualAlloc(NULL, size + 2*4096, MEM_COMMIT, PAGE_READWRITE);
    ptr += 4096 - (size & 4095);
    DWORD oldf;
    VirtualProtect(ptr + size, 4096, PAGE_NOACCESS, &oldf);
    return ptr;
#elif defined(__APPLE__)
	return (LPBYTE)malloc(size);
#elif defined(__ANDROID__)
	void *ptr = nullptr;
	posix_memalign(&ptr, 4096, size);
	return ptr;
#else
	return (LPBYTE)aligned_alloc(4096, size);
#endif		
  }

  if(!g_CudaEnabled)
    return fprintf(stderr, "CUDA is disabled\n"), nullptr;

  if(type == MEM_PINNED)
  {
    void *ptr = nullptr;

	if(auto err = cuMemAllocHost(&ptr, size))
      return fprintf(stderr, "cuMemAllocHost() error %d (%s)\n", err, GetCudaDrvApiErrorText(err)), nullptr;
    
    return ptr;
  }

  if(type == MEM_GPU)
  {
    //printf("Allocating CUDA GPU memory: %zd byte(s)\n", size, device);

    void *ptr = nullptr;

	if(auto err = cuMemAlloc((CUdeviceptr*)&ptr, size))
      return fprintf(stderr, "cuMemAlloc() error %d (%s)\n", err, GetCudaDrvApiErrorText(err)), nullptr;

    return ptr;
  }

  return nullptr;
}

void mem_release(MemType type, void* ptr)
{
  if(type == MEM_SYSTEM)
  {
#ifdef _WIN32
	VirtualFree(ptr, 0, MEM_RELEASE);
#else
	free(ptr);
#endif		
  }

  if (!g_CudaEnabled)
  {
	fprintf(stderr, "CUDA is disabled\n");
	return;
  }

  if(type == MEM_PINNED)
  {
	cuMemFreeHost(ptr);
	return;
  }

  if(type == MEM_GPU)
  {
	cuMemFree((CUdeviceptr)ptr);
	return;
  }
}

