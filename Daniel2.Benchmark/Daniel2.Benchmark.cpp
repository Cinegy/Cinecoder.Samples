#define _CRT_SECURE_NO_WARNINGS

#ifdef _WIN32
#include <windows.h>
#include <atlbase.h>
#endif

#include <stdio.h>
#include <stdlib.h>

#include <thread>
#include <chrono>
using namespace std::chrono;
using namespace std::chrono_literals;

#include "cpu_load_meter.h"

#include "../common/cuda_dyn/cuda_dyn_load.h"

#include <Cinecoder_h.h>
#include <Cinecoder_i.c>

#ifdef _WIN32
#include "Cinecoder.Plugin.GpuCodecs.h"
#include "Cinecoder.Plugin.GpuCodecs_i.c"
#endif

#include "cinecoder_errors.h"

#include "../common/cinecoder_license_string.h"
#include "../common/cinecoder_error_handler.h"

#include "../common/com_ptr.h"
#include "../common/c_unknown.h"
#include "../common/conio.h"

LONG g_target_bitrate = 0;
bool g_CudaEnabled = false;

// variables used for encoder/decoder latency calculation
static decltype(system_clock::now()) g_EncoderTimeFirstFrameIn, g_EncoderTimeFirstFrameOut;
static decltype(system_clock::now()) g_DecoderTimeFirstFrameIn, g_DecoderTimeFirstFrameOut;

// Memory types used in benchmark
enum MemType { MEM_SYSTEM, MEM_PINNED, MEM_GPU };
MemType g_mem_type = MEM_SYSTEM;

void* mem_alloc(MemType type, size_t size, int device = 0)
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
    auto err = cudaMallocHost(&ptr, size);
    if(err) fprintf(stderr, "CUDA error %d\n", err);
    return ptr;
  }

  if(type == MEM_GPU)
  {
    printf("Using CUDA GPU memory: %zd byte(s) on Device %d\n", size, device);

    int old_device;
    auto err = cudaGetDevice(&old_device);
    if(err)
      return fprintf(stderr, "cudaGetDevice() error %d\n", err), nullptr;

    if(device != old_device && (err = cudaSetDevice(device)) != 0)
      return fprintf(stderr, "cudaSetDevice(%d) error %d\n", device, err), nullptr;

    void *ptr = nullptr;
  	err = cudaMalloc(&ptr, size);
    if(err)
      return fprintf(stderr, "CUDA error %d\n", err), nullptr;

    cudaSetDevice(old_device);

    return ptr;
  }

  return nullptr;
}

#include "file_writer.h"
#include "dummy_consumer.h"

//-----------------------------------------------------------------------------
CC_COLOR_FMT ParseColorFmt(const char *s)
//-----------------------------------------------------------------------------
{
  if(0 == strcmp(s, "YUY2")) return CCF_YUY2;
  if(0 == strcmp(s, "V210")) return CCF_V210;
  if(0 == strcmp(s, "Y216")) return CCF_Y216;
  if(0 == strcmp(s, "RGBA")) return CCF_RGBA;
  if(0 == strcmp(s, "RGBX")) return CCF_RGBX;
  if(0 == strcmp(s, "NV12")) return CCF_NV12;
  if(0 == strcmp(s, "NULL")) return CCF_UNKNOWN;
  return (CC_COLOR_FMT)-1;
}

//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
//-----------------------------------------------------------------------------
{
  g_CudaEnabled = __InitCUDA() == 0;

  if(argc < 5)
  {
    puts("Usage: intra_encoder <codec> <profile.xml> <rawtype> <input_file.raw> [/outfile=<output_file.bin>] [/outfmt=<rawtype>] [/outscale=#] [/fps=#] [/device=#]");
    puts("Where the <codec> is one of the following:");
    puts("\t'D2'           -- Daniel2 CPU codec test");
	if(g_CudaEnabled)
	{
    puts("\t'D2CUDA'       -- Daniel2 CUDA codec test, data is copying from GPU into CPU pinned memory");
    puts("\t'D2CUDAGPU'    -- Daniel2 CUDA codec test, data is copying from GPU into GPU global memory");
    puts("\t'D2CUDANP'     -- Daniel2 CUDA codec test, data is copying from GPU into CPU NOT-pinned memory (bad case test)");
    }
#ifndef __aarch64__
    puts("\t'AVCI'         -- AVC-Intra CPU codec test");
#endif
#ifdef _WIN32
    puts("\t'H264_NV'      -- H264 NVidia GPU codec test (requires GPU codec plugin)");
    puts("\t'HEVC_NV'      -- HEVC NVidia GPU codec test (requires GPU codec plugin)");
    puts("\t'H264_IMDK'    -- H264 Intel QuickSync codec test (requires GPU codec plugin)");
    puts("\t'HEVC_IMDK'    -- HEVC Intel QuickSync codec test (requires GPU codec plugin)");
    puts("\t'H264_IMDK_SW' -- H264 Intel QuickSync codec test (requires GPU codec plugin)");
    puts("\t'HEVC_IMDK_SW' -- HEVC Intel QuickSync codec test (requires GPU codec plugin)");
    puts("\t'MPEG'         -- MPEG s/w encoder");
    puts("\t'H264'         -- H264 s/w encoder");
#endif
    puts("\n      <rawtype> can be 'YUY2','V210','V216','RGBA' or 'NULL'");
    return 1;
  }

  CC_VERSION_INFO version = Cinecoder_GetVersion();
  printf("Cinecoder version %d.%02d.%02d\n", version.VersionHi, version.VersionLo, version.EditionNo);

  Cinecoder_SetErrorHandler(new C_CinecoderErrorHandler());

  CLSID clsidEnc = {}, clsidDec = {}; const char *strEncName = 0;
  bool bForceGetFrameOnDecode = false;
  bool bLoadGpuCodecsPlugin = false;
  if(0 == strcmp(argv[1], "AVCI"))
  { 
    clsidEnc = CLSID_CC_AVCIntraEncoder; 
    clsidDec = CLSID_CC_AVCIntraDecoder2;
    strEncName = "AVC-Intra"; 
  }
  if(0 == strcmp(argv[1], "D2"))
  { 
    clsidEnc = CLSID_CC_DanielVideoEncoder;
    clsidDec = CLSID_CC_DanielVideoDecoder; 
    strEncName = "Daniel2"; 
    bForceGetFrameOnDecode = true;
  }

  if(g_CudaEnabled && 0 == strcmp(argv[1], "D2CUDA"))
  {
    clsidEnc = CLSID_CC_DanielVideoEncoder_CUDA; 
    clsidDec = CLSID_CC_DanielVideoDecoder_CUDA; 
    strEncName = "Daniel2_CUDA";
    g_mem_type = MEM_PINNED;
    bForceGetFrameOnDecode = true;
  }
  if(g_CudaEnabled && 0 == strcmp(argv[1], "D2CUDANP"))
  {
    clsidEnc = CLSID_CC_DanielVideoEncoder_CUDA; 
    clsidDec = CLSID_CC_DanielVideoDecoder_CUDA; 
    strEncName = "Daniel2_CUDA (NOT PINNED MEMORY!!)";
    //g_mem_type = MEM_PINNED;
    bForceGetFrameOnDecode = true;
  }
  if(g_CudaEnabled && 0 == strcmp(argv[1], "D2CUDAGPU"))
  {
    clsidEnc = CLSID_CC_DanielVideoEncoder_CUDA; 
    clsidDec = CLSID_CC_DanielVideoDecoder_CUDA; 
    strEncName = "Daniel2_CUDA (GPU-GPU mode)";
    g_mem_type = MEM_GPU;
  }

#ifdef _WIN32
  if(0 == strcmp(argv[1], "H264_NV"))
  { 
    clsidEnc = CLSID_CC_H264VideoEncoder_NV; 
    clsidDec = CLSID_CC_H264VideoDecoder_NV; 
    strEncName = "NVidia H264"; 
    bLoadGpuCodecsPlugin = true;
  }
  if(0 == strcmp(argv[1], "H264_NV_GPU"))
  { 
    clsidEnc = CLSID_CC_H264VideoEncoder_NV; 
    clsidDec = CLSID_CC_H264VideoDecoder_NV; 
    strEncName = "NVidia H264"; 
    bLoadGpuCodecsPlugin = true;
    g_mem_type = MEM_GPU;
  }
  if(0 == strcmp(argv[1], "HEVC_NV"))
  { 
    clsidEnc = CLSID_CC_HEVCVideoEncoder_NV; 
    clsidDec = CLSID_CC_HEVCVideoDecoder_NV; 
    strEncName = "NVidia HEVC"; 
    bLoadGpuCodecsPlugin = true;
  }
  if(0 == strcmp(argv[1], "HEVC_NV_GPU"))
  { 
    clsidEnc = CLSID_CC_HEVCVideoEncoder_NV; 
    clsidDec = CLSID_CC_HEVCVideoDecoder_NV; 
    strEncName = "NVidia HEVC"; 
    g_mem_type = MEM_GPU;
    bLoadGpuCodecsPlugin = true;
  }
  if(0 == strcmp(argv[1], "H264_IMDK"))
  { 
    clsidEnc = CLSID_CC_H264VideoEncoder_IMDK; 
    clsidDec = CLSID_CC_H264VideoDecoder_IMDK; 
    strEncName = "Intel QuickSync H264"; 
    bLoadGpuCodecsPlugin = true;
  }
  if(0 == strcmp(argv[1], "HEVC_IMDK"))
  { 
    clsidEnc = CLSID_CC_HEVCVideoEncoder_IMDK; 
    clsidDec = CLSID_CC_HEVCVideoDecoder_IMDK; 
    strEncName = "Intel QuickSync HEVC"; 
    bLoadGpuCodecsPlugin = true;
  }
  if(0 == strcmp(argv[1], "H264_IMDK_SW"))
  { 
    clsidEnc = CLSID_CC_H264VideoEncoder_IMDK_SW;
    clsidDec = CLSID_CC_H264VideoDecoder_IMDK_SW;
    strEncName = "Intel QuickSync H264 (SOFTWARE)"; 
    bLoadGpuCodecsPlugin = true;
  }
  if(0 == strcmp(argv[1], "HEVC_IMDK_SW"))
  { 
    clsidEnc = CLSID_CC_HEVCVideoEncoder_IMDK_SW;
    clsidDec = CLSID_CC_HEVCVideoDecoder_IMDK_SW;
    strEncName = "Intel QuickSync HEVC (SOFTWARE)";
    bLoadGpuCodecsPlugin = true;
  }
  if(0 == strcmp(argv[1], "H264"))
  { 
    clsidEnc = CLSID_CC_H264VideoEncoder; 
    clsidDec = CLSID_CC_H264VideoDecoder; 
    strEncName = "H264"; 
  }
  if(0 == strcmp(argv[1], "MPEG"))
  { 
    clsidEnc = CLSID_CC_MpegVideoEncoder; 
    clsidDec = CLSID_CC_MpegVideoDecoder; 
    strEncName = "MPEG"; 
  }
#endif

  if(!strEncName)
    return fprintf(stderr, "Unknown encoder type '%s'\n", argv[1]), -1;

  FILE *profile = fopen(argv[2], "rt");
  if(profile == NULL)
    return fprintf(stderr, "Can't open the profile %s\n", argv[2]), -2;

  const char *strInputFormat = argv[3], *strOutputFormat = argv[3];
  CC_COLOR_FMT cFormat = ParseColorFmt(strInputFormat);
  if(cFormat == (CC_COLOR_FMT)-1)
    return fprintf(stderr, "Unknown raw data type '%s'\n", argv[3]), -3;

  FILE *inpf = fopen(argv[4], "rb");
  if(inpf == NULL)
    return fprintf(stderr, "Can't open the file %s", argv[4]), -4;

  FILE *outf = NULL;
  CC_COLOR_FMT cOutputFormat = cFormat;
  int DecoderScale = 0;
  double TargetFps = 0;
  int EncDeviceID = -2, DecDeviceID = -2;
  int NumThreads = 0;

  for(int i = 5; i < argc; i++)
  {
    if(0 == strncmp(argv[i], "/outfile=", 9))
    {
      outf = fopen(argv[i] + 9, "wb");

      if(outf == NULL)
        return fprintf(stderr, "Can't create the file %s", argv[i] + 9), -i;
    }

    else if(0 == strncmp(argv[i], "/outfmt=", 8))
    {
      cOutputFormat = ParseColorFmt(strOutputFormat = argv[i] + 8);

      if(cOutputFormat == (CC_COLOR_FMT)-1)
        return fprintf(stderr, "Unknown output raw data type '%s'\n", argv[i]), -i;
    }

    else if(0 == strncmp(argv[i], "/outscale=", 10))
    {
 	  DecoderScale = atoi(argv[i] + 10);
    }

    else if(0 == strncmp(argv[i], "/fps=", 5))
    {
 	  TargetFps = atof(argv[i] + 5);
    }

    else if(0 == strncmp(argv[i], "/device=", 8))
    {
 	  EncDeviceID = atoi(argv[i] + 8);
    }

    else if(0 == strncmp(argv[i], "/device2=", 9))
    {
 	  DecDeviceID = atoi(argv[i] + 9);
    }

    else if(0 == strncmp(argv[i], "/numthreads=", 12))
    {
 	  NumThreads = atoi(argv[i] + 12);
    }

    else
      return fprintf(stderr, "Unknown switch '%s'\n", argv[i]), -i;
  }

//  cudaSetDeviceFlags(cudaDeviceMapHost);

  HRESULT hr = S_OK;

  com_ptr<ICC_ClassFactory> pFactory;
  hr = Cinecoder_CreateClassFactory(&pFactory);
  if(FAILED(hr)) return hr;

  hr = pFactory->AssignLicense(COMPANYNAME,LICENSEKEY);
  if(FAILED(hr))
    return fprintf(stderr, "Incorrect license"), hr;

#ifdef _WIN32
  const char *gpu_plugin_name = "Cinecoder.Plugin.GpuCodecs.dll";
  if(bLoadGpuCodecsPlugin && FAILED(hr = pFactory->LoadPlugin(CComBSTR(gpu_plugin_name))))
    return fprintf(stderr, "Error loading '%s'", gpu_plugin_name), hr;
#endif
  
  char profile_text[4096] = { 0 };
  if (fread(profile_text, 1, sizeof(profile_text), profile) < 0)
	return fprintf(stderr, "Profile reading error"), -1;

#ifdef _WIN32
  CComBSTR pProfile = profile_text;
#else
  auto pProfile = profile_text;
#endif

  com_ptr<ICC_VideoEncoder> pEncoder;

  _fseeki64(inpf, 0, SEEK_SET);

  hr = pFactory->CreateInstance(clsidEnc, IID_ICC_VideoEncoder, (IUnknown**)&pEncoder);
  if(FAILED(hr)) return hr;

  if(NumThreads > 0)
  {
    fprintf(stderr, "Setting up specified number of threads = %d for the encoder: ", NumThreads);

    com_ptr<ICC_ThreadsCountProp> pTCP;

    if(FAILED(hr = pEncoder->QueryInterface(IID_ICC_ThreadsCountProp, (void**)&pTCP)))
      fprintf(stderr, "NAK. No ICC_ThreadsCountProp interface found\n");

    else if(FAILED(hr = pTCP->put_ThreadsCount(NumThreads)))
      return fprintf(stderr, "FAILED\n"), hr;

    fprintf(stderr, "OK\n");
  }

  com_ptr<ICC_DeviceIDProp> pDevId;
  pEncoder->QueryInterface(IID_ICC_DeviceIDProp, (void**)&pDevId);

  if(EncDeviceID >= -1)
  {
    if(pDevId)
    {
      printf("Encoder has ICC_DeviceIDProp interface.\n");

      if(EncDeviceID >= -1)
      {
        if(FAILED(hr = pDevId->put_DeviceID(EncDeviceID)))
          return fprintf(stderr, "Failed to assign DeviceId=%d to the encoder", EncDeviceID), hr;
      }
    }
    else
    {
      printf("Encoder has no ICC_DeviceIDProp interface. Using default device (unknown)\n");
    }
  }

  hr = pEncoder->InitByXml(pProfile);
  if(FAILED(hr)) return hr;

  if(EncDeviceID < -1)
  {
    if(pDevId)
    {
      if(FAILED(hr = pDevId->get_DeviceID(&EncDeviceID)))
        return fprintf(stderr, "Failed to get DeviceId from the encoder"), hr;
    }
    else
      EncDeviceID = 0;
  }

  if(EncDeviceID >= -1)
    printf("Encoder device id = %d\n", EncDeviceID);

  CC_AMOUNT concur_level = 0;
  com_ptr<ICC_ConcurrencyLevelProp> pConcur;
  if(S_OK == pEncoder->QueryInterface(IID_ICC_ConcurrencyLevelProp, (void**)&pConcur))
  {
    printf("Encoder has ICC_ConcurrencyLevelProp interface.\n");
    
    if(FAILED(hr = pConcur->get_ConcurrencyLevel(&concur_level)))
      return fprintf(stderr, "Failed to get ConcurrencyLevel from the encoder"), hr;
    
    printf("Encoder concurrency level = %d\n", concur_level);
  }

  CC_VIDEO_FRAME_DESCR vpar = { cFormat };

  printf("Encoder: %s\n", strEncName);
  printf("Footage: type=%s filename=%s\n", argv[3], argv[4]);
  printf("Profile: %s\n%s\n", argv[2], profile_text);

  com_ptr<ICC_VideoStreamInfo> pVideoInfo;
  if(FAILED(hr = pEncoder->GetVideoStreamInfo(&pVideoInfo)))
    return fprintf(stderr, "Failed to get video stream info the encoder: code=%08x", hr), hr;

  CC_SIZE frame_size = {};
  pVideoInfo->get_FrameSize(&frame_size);

  DWORD frame_pitch = 0, dec_frame_pitch = 0;
  if(FAILED(hr = pEncoder->GetStride(cFormat, &frame_pitch)))
    return fprintf(stderr, "Failed to get frame pitch from the encoder: code=%08x", hr), hr;
  if(cOutputFormat == CCF_UNKNOWN)
    dec_frame_pitch = frame_pitch;
  else if(FAILED(hr = pEncoder->GetStride(cOutputFormat, &dec_frame_pitch)))
    return fprintf(stderr, "Failed to get frame pitch for the decoder: code=%08x", hr), hr;

  //__declspec(align(32)) static BYTE buffer[];
  size_t uncompressed_frame_size = size_t(frame_pitch) * frame_size.cy;
  printf("Frame size: %dx%d, pitch=%d, bytes=%zd\n", frame_size.cx, frame_size.cy, frame_pitch, uncompressed_frame_size);

  BYTE *read_buffer = (BYTE*)mem_alloc(MEM_SYSTEM, uncompressed_frame_size);
  if(!read_buffer)
    return fprintf(stderr, "buffer allocation error for %zd byte(s)", uncompressed_frame_size), E_OUTOFMEMORY;
  else
    printf("Compressed buffer address  : 0x%p\n", read_buffer);

  std::vector<BYTE*> source_frames;
  int max_num_frames_in_loop = 32;

  for(int i = 0; i < max_num_frames_in_loop; i++)
  {
    size_t read_size = fread(read_buffer, 1, uncompressed_frame_size, inpf);

    if(read_size < uncompressed_frame_size)
      break;

    BYTE *buf = (BYTE*)mem_alloc(g_mem_type, uncompressed_frame_size, EncDeviceID);
    if(!buf)
      return fprintf(stderr, "buffer allocation error for %zd byte(s)", uncompressed_frame_size), E_OUTOFMEMORY;
    else
      printf("Uncompressed buffer address: 0x%p, format: %s, size: %zd byte(s)\n", buf, strInputFormat, uncompressed_frame_size);

    if(g_mem_type == MEM_GPU)
  	  cudaMemcpy(buf, read_buffer, uncompressed_frame_size, cudaMemcpyHostToDevice);
  	else
  	  memcpy(buf, read_buffer, uncompressed_frame_size);

  	source_frames.push_back(buf);
  }

  C_FileWriter *pFileWriter = new C_FileWriter(outf, true, source_frames.size());
  hr = pEncoder->put_OutputCallback(static_cast<ICC_ByteStreamCallback*>(pFileWriter));
  if(FAILED(hr)) return hr;

  com_ptr<ICC_VideoConsumerExtAsync> pEncAsync = 0;
  pEncoder->QueryInterface(IID_ICC_VideoConsumerExtAsync, (void**)&pEncAsync);

  CpuLoadMeter cpuLoadMeter;
  
  auto t00 = system_clock::now();
  int frame_count = 0, total_frame_count = 0;
  auto t0 = t00;

  auto coded_size0 = pFileWriter->GetTotalBytesWritten();

  g_EncoderTimeFirstFrameIn = t00;

  printf("Performing encoding loop, press ESC to break\n");

  int max_frames = 0x7fffffff;
  int update_mask = 0x07;
  
  for(int frame_no = 0; frame_no < max_frames; frame_no++)
  {
    size_t idx = frame_no % (source_frames.size()*2-1);
    if(idx >= source_frames.size())
      idx = source_frames.size()*2 - idx - 1;

    if(pEncAsync)
      hr = pEncAsync->AddScaleFrameAsync(source_frames[idx], (DWORD)uncompressed_frame_size, &vpar, pEncAsync);

    else
      hr = pEncoder->AddFrame(vpar.cFormat, source_frames[idx], (DWORD)uncompressed_frame_size);

    if(FAILED(hr))
    {
      pEncoder = NULL;
      return hr;
    }

    if(TargetFps > 0)
    {
 	  auto t1 = system_clock::now();

	  auto Treal = duration_cast<milliseconds>(t1 - t0).count();
	  auto Tideal = (int)(frame_count * 1000 / TargetFps);

	  if (Tideal > Treal + 1)
		  std::this_thread::sleep_for(milliseconds{ Tideal - Treal });
	}

    if((frame_count & update_mask) == update_mask)
    {
 	  auto t1 = system_clock::now();
      auto dT = duration<double>(t1 - t0).count();
	  auto coded_size = pFileWriter->GetTotalBytesWritten();

      fprintf(stderr, " %d, %.3f fps, in %.3f GB/s, out %.3f Mbps, CPU load: %.1f%%    \r",
      	frame_no, frame_count / dT, 
      	uncompressed_frame_size / 1E9 * frame_count / dT,
      	(coded_size - coded_size0) * 8 / 1E6 / dT,
      	cpuLoadMeter.GetLoad());
      
      t0 = t1;
      frame_count = 0;
      coded_size0 = coded_size;

      if(dT < 0.5)
      {
        update_mask = (update_mask<<1) | 1;
      }
      else while(dT > 2 && update_mask > 1)
      {
        update_mask = (update_mask>>1) | 1;
        dT /= 2;
      }
    }

    frame_count++;
    total_frame_count++;

    if(_kbhit() && _getch() == 27)
      break;
  }

  hr = pEncoder->Done(CC_TRUE);
  if(FAILED(hr)) return hr;

  auto t1 = system_clock::now();

  pEncoder = NULL;

  puts("\nDone.\n");

  auto dT = duration<double>(t1 - t00).count();
  printf("Average performance = %.3f fps (%.1f ms/f), avg data rate = %.3f GB/s\n", 
          total_frame_count / dT, dT * 1000 / total_frame_count,
          uncompressed_frame_size / 1E9 * total_frame_count / dT);

  auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(g_EncoderTimeFirstFrameOut - g_EncoderTimeFirstFrameIn);
  printf("Encoder latency = %d ms\n", (int)time_ms.count());

  fclose(inpf);
  if(outf) fclose(outf);

  // decoder test ==================================================================
  fprintf(stderr, "\n------------------------------------------------------------\nEntering decoder test loop...\n");
  com_ptr<ICC_VideoDecoder> pDecoder;
  hr = pFactory->CreateInstance(clsidDec, IID_ICC_VideoDecoder, (IUnknown**)&pDecoder);
  if(FAILED(hr)) return hr;

  if(NumThreads > 0)
  {
    fprintf(stderr, "Setting up specified number of threads = %d for the decoder: ", NumThreads);

    com_ptr<ICC_ThreadsCountProp> pTCP;

    if(FAILED(hr = pEncoder->QueryInterface(IID_ICC_ThreadsCountProp, (void**)&pTCP)))
      fprintf(stderr, "NAK. No ICC_ThreadsCountProp interface found\n");

    else if(FAILED(hr = pTCP->put_ThreadsCount(NumThreads)))
      return fprintf(stderr, "FAILED\n"), hr;

    fprintf(stderr, "OK\n");
  }

  if(DecDeviceID < -1 && EncDeviceID >= -1)
    DecDeviceID = EncDeviceID;

  if(DecDeviceID >= -1 && S_OK == pDecoder->QueryInterface(IID_ICC_DeviceIDProp, (void**)&pDevId))
  {
    printf("Decoder has ICC_DeviceIDProp interface.\n");
    
    if(FAILED(hr = pDevId->put_DeviceID(DecDeviceID)))
      return fprintf(stderr, "Failed to assign DeviceId %d to the decoder", DecDeviceID), hr;

    printf("Decoder device id = %d\n", DecDeviceID);
  }

  if(concur_level != 0 && S_OK == pDecoder->QueryInterface(IID_ICC_ConcurrencyLevelProp, (void**)&pConcur))
  {
    printf("Decoder has ICC_ConcurrencyLevelProp interface.\n");
    
    if(FAILED(hr = pConcur->put_ConcurrencyLevel(concur_level)))
      return fprintf(stderr, "Failed to assign ConcurrencyLevel %d to the decoder", concur_level), hr;
  }

  hr = pDecoder->Init();
  if(FAILED(hr)) return hr;

  if(pConcur)
  {
    if(FAILED(hr = pConcur->get_ConcurrencyLevel(&concur_level)))
      return fprintf(stderr, "Failed to get ConcurrencyLevel from the decoder"), hr;
    
    printf("Decoder concurrency level = %d\n", concur_level);
  }

  if(!bForceGetFrameOnDecode)
  	cFormat = CCF_UNKNOWN;

  uncompressed_frame_size = size_t(dec_frame_pitch) * frame_size.cy;

  BYTE *dec_buf = (BYTE*)mem_alloc(g_mem_type, uncompressed_frame_size, DecDeviceID);
  if(!dec_buf)
    return fprintf(stderr, "buffer allocation error for %zd byte(s)", uncompressed_frame_size), E_OUTOFMEMORY;
  else
    printf("Uncompressed buffer address: 0x%p, format: %s, size: %zd byte(s)\n", dec_buf, strOutputFormat, uncompressed_frame_size);

  com_ptr<ICC_VideoQualityMeter> pPsnrCalc;
  if(cOutputFormat == cFormat && g_mem_type != MEM_GPU)
  {
    if(FAILED(hr = pFactory->CreateInstance(CLSID_CC_VideoQualityMeter, IID_ICC_VideoQualityMeter, (IUnknown**)&pPsnrCalc)))
      fprintf(stdout, "Can't create VideoQualityMeter, error=%xh, PSNR calculation is disabled\n", hr);
  }
  else
  {
    fprintf(stdout, "PSNR calculation is disabled due to %s\n", cOutputFormat != cFormat ? "color format mismatch" : "GPU memory");
  }

  hr = pDecoder->put_OutputCallback(new C_DummyWriter(cOutputFormat, dec_buf, (int)uncompressed_frame_size, pPsnrCalc, source_frames[0]));
  if(FAILED(hr)) return hr;

  com_ptr<ICC_ProcessDataPolicyProp> pPDP;
  if(SUCCEEDED(pDecoder->QueryInterface(IID_ICC_ProcessDataPolicyProp, (void**)&pPDP)))
    pPDP->put_ProcessDataPolicy(CC_PDP_PARSED_DATA);

  com_ptr<ICC_DanielVideoDecoder_CUDA> pCudaDec;
  if(SUCCEEDED(pDecoder->QueryInterface(IID_ICC_DanielVideoDecoder_CUDA, (void**)&pCudaDec)))
    pCudaDec->put_TargetColorFormat(cOutputFormat);

  t00 = t0 = system_clock::now();

  g_DecoderTimeFirstFrameIn = t00;

  frame_count = total_frame_count = 0;

  printf("Performing decoding loop, press ESC to break\n");
  printf("coded sequence length = %zd\n", pFileWriter->GetCodedSequenceLength());
  for(int i = 0; i < pFileWriter->GetCodedSequenceLength(); i++)
    printf(" %zd", pFileWriter->GetCodedFrame(i).second);
  puts("");

  update_mask = 0x07;

  int key_pressed = 0;

  int warm_up_frames = 4;

  coded_size0 = 0; long long coded_size = 0;

  if(int num_coded_frames = (int)pFileWriter->GetCodedSequenceLength())
  for(int frame_no = 0; frame_no < max_frames; frame_no++)
  {
    auto codedFrame = pFileWriter->GetCodedFrame(frame_no % num_coded_frames);
	hr = pDecoder->ProcessData(codedFrame.first, (DWORD)codedFrame.second);

    if(FAILED(hr))
    {
      pDecoder = NULL;
      return hr;
    }

    coded_size += codedFrame.second;

    if(TargetFps > 0)
    {
 	  auto t1 = system_clock::now();

 	  auto Treal  = duration_cast<milliseconds>(t1 - t0).count();
      auto Tideal = (int)(frame_count * 1000 / TargetFps);

      if(Tideal > Treal + 1)
        std::this_thread::sleep_for(milliseconds{Tideal - Treal});
    }

    if(warm_up_frames > 0)
    {
      if(--warm_up_frames > 0)
        continue;

      t00 = t0 = system_clock::now();
    }

    if((frame_count & update_mask) == update_mask)
    {
	  auto t1 = system_clock::now();
	  auto dT = duration<double>(t1 - t0).count();

      if(dT < 0.5)
        update_mask = (update_mask<<1) | 1;
      else if(dT > 2)
        update_mask = (update_mask>>1) | 1;

      fprintf(stderr, " %d, %.3f fps, in %.3f Mbps, out %.3f GB/s, CPU load: %.1f%%    \r", 
      	frame_no, frame_count / dT, 
      	(coded_size - coded_size0) * 8 / 1E6 / dT,
      	uncompressed_frame_size / 1E9 * frame_count / dT, cpuLoadMeter.GetLoad());
      
      t0 = t1;
      coded_size0 = coded_size;

      frame_count = 0;
    }

    frame_count++;
    total_frame_count++;

    if(_kbhit() && _getch() == 27)
      break;
  }

  hr = pDecoder->Done(CC_TRUE);
  if(FAILED(hr)) return hr;

  t1 = system_clock::now();

  pDecoder = NULL;

  puts("\nDone.\n");

  dT = duration<double>(t1 - t00).count();
  printf("Average performance = %.3f fps (%.1f ms/f), avg data rate = %.3f GB/s\n", 
          total_frame_count / dT, dT * 1000 / total_frame_count,
          uncompressed_frame_size / 1E9 * total_frame_count / dT);

  time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(g_DecoderTimeFirstFrameOut - g_DecoderTimeFirstFrameIn);
  printf("Decoder latency = %d ms\n", (int)time_ms.count());
  
  return 0;
}
