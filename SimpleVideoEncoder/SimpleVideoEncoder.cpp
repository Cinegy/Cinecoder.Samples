// SimpleVideoEncoder.cpp : Defines the entry point for the console application.
//

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <tchar.h>

#include <conio.h>
#include <malloc.h>

//#include <windows.h>

#include "Cinecoder_h.h"
#include "Cinecoder_i.c"

#include "../cinecoder_license_string.h"
#include "../cinecoder_error_handler.h"

#include "../common/com_ptr.h"
#include "../common/c_unknown.h"

#include "file_writer.h"

#include <chrono>
using namespace std::chrono;

//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
//-----------------------------------------------------------------------------
{
  puts("The Cinegy(r) Video Encoder sample application\n");

  if(argc < 5)
  {
    puts("Usage: video_encoder <profile.xml> <input_file.yuv> <color_format> <output_file>");
    puts("Where <color_format> can be YUY2, UYVY, YUY2_10, UYVY_10");
    return 1;
  }

  // Opening the profile file -------------------------
  FILE *xmlf = fopen(argv[1], "rt");
  if(xmlf == NULL)
    return fprintf(stderr, "Can't open the profile XML file %s", argv[1]), -1;

  fseek(xmlf, 0, SEEK_END);
  long xml_length = ftell(xmlf);
  fseek(xmlf, 0, SEEK_SET);

  char *profile = (char*)calloc(xml_length+1, 1);
  fread(profile, 1, xml_length, xmlf);
  fclose(xmlf);

  // Simple XML parsing
  // We assuming the profile is in form of
  // <EncoderName>
  // ....
  // </EncoderName>
  char *p = profile;
  while(isspace(*p)) p++;
  if(p[0] != '<' || !isalpha(p[1]))
    return fprintf(stderr, "Profile parsing error"), -10;

  CLSID clsidVideoEncoder;
  if(0 == strncmp(p+1, "Mpeg", 4))
    clsidVideoEncoder = CLSID_CC_MpegVideoEncoder;
  else if(0 == strncmp(p+1, "H264", 4))
    clsidVideoEncoder = CLSID_CC_H264VideoEncoder;
  else if(0 == strncmp(p+1, "AVCI", 4))
    clsidVideoEncoder = CLSID_CC_AVCIntraEncoder;
  else if(0 == strncmp(p+1, "Daniel", 6))
    clsidVideoEncoder = CLSID_CC_DanielVideoEncoder;
  else
    return fprintf(stderr, "Can't determine the encoder type from the profile\n"), -11;

  DWORD encoder_fcc = *(DWORD*)(p+1); // dirty hack

  // Determining the color format ----------------------------------
  CC_COLOR_FMT color_fmt = CCF_UNKNOWN;
  if     (0 == stricmp(argv[3], "YUY2"))    color_fmt = CCF_YUY2;       
  else if(0 == stricmp(argv[3], "UYVY"))	color_fmt = CCF_UYVY;      
  else if(0 == stricmp(argv[3], "YUY2_10")) color_fmt = CCF_YUY2_10BIT;
  else if(0 == stricmp(argv[3], "UYVY_10")) color_fmt = CCF_UYVY_10BIT; 
  else return fprintf(stderr, "Unknown color format '%s'.\n", argv[3]), -3;

  // Initializing the Cinecoder ------------------------------------
  printf("Initializing Cinecoder: ");

  CC_VERSION_INFO version = Cinecoder_GetVersion();
  printf("Cinecoder.dll version %d.%02d.%02d\n\n", version.VersionHi, version.VersionLo, version.EditionNo);

  HRESULT hr = S_OK;
  Cinecoder_SetErrorHandler(&g_ErrorHandler);

  com_ptr<ICC_ClassFactory> pFactory;
  if(FAILED(hr = Cinecoder_CreateClassFactory(&pFactory)))
    return fprintf(stderr, "Failed to create the factory"), hr;

  if(FAILED(hr = pFactory->AssignLicense(COMPANYNAME, LICENSEKEY)))
    return fprintf(stderr, "Incorrect license"), hr;

  com_ptr<ICC_VideoEncoder> pVideoEncoder;
  if(FAILED(hr = pFactory->CreateInstance(clsidVideoEncoder, IID_ICC_VideoEncoder, (IUnknown**)&pVideoEncoder)))
    return hr;

#ifdef _WIN32
  OLECHAR *cc_profile = (OLECHAR*)_alloca((xml_length + 1) * sizeof(OLECHAR));
  mbstowcs(cc_profile, profile, xml_length);
#else
#define cc_profile profile
#endif
  if(FAILED(hr = pVideoEncoder->InitByXml(cc_profile)))
    return hr;

  free(profile);

  com_ptr<ICC_VideoStreamInfo> pVideoDescr;
  if(FAILED(hr = pVideoEncoder->GetVideoStreamInfo(&pVideoDescr)))
    return hr;

  CC_SIZE frame_size;
  pVideoDescr->get_FrameSize(&frame_size);

  CC_RATIONAL frame_rate;
  pVideoDescr->get_FrameRate(&frame_rate);

  printf("Encoder reports: type = %4.4s\n", (char*)&encoder_fcc);
  printf("Target frame size = %dx%d\n", frame_size.cx, frame_size.cy);
  printf("Target frame rate = %g\n", (double)frame_rate.num / frame_rate.denom);

  // we assume the full (non-anamorph) frame sizes at input
  if(frame_size.cx ==  960) frame_size.cx = 1280;
  if(frame_size.cx == 1440) frame_size.cx = 1920;

  int bpp = 2 << int(color_fmt == CCF_YUY2_10BIT || color_fmt == CCF_UYVY_10BIT);
  int pitch = frame_size.cx * bpp;
  int src_frame_size = pitch * frame_size.cy;

  BYTE *yuv_buffer = (BYTE*)malloc(src_frame_size);
  if(!yuv_buffer)
    return fprintf(stderr, "Memory allocation error for %d byte(s)", src_frame_size), E_OUTOFMEMORY;

  puts("");
  printf("Source format is:\n");
  printf("Source frame size = %dx%d\n", frame_size.cx, frame_size.cy);
  printf("Pixel format = %s, bytes per pixel = %d\n", argv[3], bpp);

  CC_ADD_VIDEO_FRAME_PARAMS vpar = { color_fmt, frame_size, pitch };

  // Opening source YUV file --------------------------
  FILE *inpf = fopen(argv[2], "rb");
  if(inpf == NULL)
    return fprintf(stderr, "Can't open the source YUV file %s", argv[2]), -2;

  // Creating target file for compressed video --------
  FILE *outf = fopen(argv[4], "wb");
  if(outf == NULL)
    return fprintf(stderr, "Can't create the target file %s", argv[4]), -4;

  __int64 total_size = 0;
  hr = pVideoEncoder->put_OutputCallback(static_cast<ICC_ByteStreamCallback*>(new C_FileWriter(outf, &total_size)));
  if(FAILED(hr)) return hr;

  // The main loop ------------------------------------
  int frame = 0;
  auto t0 = system_clock::now();

  for(;;)
  {
    size_t ret_size = fread(yuv_buffer, 1, src_frame_size, inpf);
    
    if(ret_size != src_frame_size) 
      break;

    fprintf(stderr, "\rframe # %d", frame++);

    //if(FAILED(hr = pVideoEncoder->AddFrame(color_fmt, yuv_buffer, src_frame_size, 0, NULL)))
    if(FAILED(hr = pVideoEncoder->AddScaleFrame(yuv_buffer, src_frame_size, &vpar)))
      break;

    if(_kbhit() && _getch() == 27)
    {
      puts("\nCancelled.");
      break;
    }
  }

  free(yuv_buffer);

  if(SUCCEEDED(hr)) hr = pVideoEncoder->Done(CC_TRUE);

  if(FAILED(hr))
  {
	  printf("Error %08x\n", hr);
	  return hr;
  }

  auto t1 = system_clock::now();
  auto dT = duration_cast<milliseconds>(t1 - t0).count();

  printf("\n%.3f Mbps, %g fps\n", total_size * 8.0 * frame_rate.num / frame_rate.denom / frame / 1E6, double(frame) * 1000 / dT);

  fclose(inpf);
  fclose(outf);

  puts("Press any key to exit . . .");
  getchar();

  return 0;
}
