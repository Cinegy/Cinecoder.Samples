// SimpleVideoDecoder.cpp : Defines the entry point for the console application.
//
//#include <windows.h>

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <tchar.h>

#include <memory>
#include <assert.h>

#include "Cinecoder_h.h"	
#include "Cinecoder_i.c"

#include "cinecoder_errors.h"

#include "../common/cinecoder_license_string.h"
#include "../common/cinecoder_error_handler.h"

#include "../common/com_ptr.h"
#include "../common/c_unknown.h"

#include "ppm_writer.h"

#ifdef _MSC_VER
#define stricmp _stricmp
#endif

//------------------------------------------------------------------
int main(int argc, char* argv[])
//------------------------------------------------------------------
{
	puts("\nThis sample application decodes an elementary video stream and stores the decoded frames into PPM sequence\n");

	if(argc < 3)
	{
      puts("Usage: video_decoder.exe <input_file> <codec_name>");
      puts("Where codec_name can be MPEG, H264 or DN2");
      return 1;
	}

    HRESULT hr = S_OK;

    // Ppening the input file --------------------------------------
    fprintf(stderr, "Opening input file '%s': ", argv[1]);

    FILE *hSrcFile = fopen(argv[1], "rb");
    if(hSrcFile == NULL)
      return fprintf(stderr, "Can't open source file %s\n", argv[1]), -1;

    fprintf(stderr, "ok\n");

    // Determining the codec clsid ----------------------------------
    CLSID CODEC_CLSID;
    
    if(0 == stricmp(argv[2], "MPEG"))
      CODEC_CLSID = CLSID_CC_MpegVideoDecoder;
    else if(0 == stricmp(argv[2], "H264"))
      CODEC_CLSID = CLSID_CC_H264VideoDecoder;
    else if(0 == stricmp(argv[2], "DN2"))
      CODEC_CLSID = CLSID_CC_DanielVideoDecoder;
    else
      return fprintf(stderr, "Unknown codec_name '%s' specified\n", argv[2]), -2;

    // Initializing the Cinecoder ------------------------------------
    printf("Initializing Cinecoder: ");

    CC_VERSION_INFO version = Cinecoder_GetVersion();
    printf("Cinecoder.dll version %d.%02d.%02d.%d\n\n", version.VersionHi, version.VersionLo, version.EditionNo, version.RevisionNo);

    Cinecoder_SetErrorHandler(&g_ErrorHandler);

    com_ptr<ICC_ClassFactory> spFactory;
    if(FAILED(hr = Cinecoder_CreateClassFactory(&spFactory)))
      return hr;

    spFactory->AssignLicense(COMPANYNAME, LICENSEKEY);

    com_ptr<ICC_VideoDecoder> spVideoDec;
    if(FAILED(hr = spFactory->CreateInstance(CODEC_CLSID, IID_ICC_VideoDecoder, (IUnknown**)&spVideoDec)))
      return hr;

    if(FAILED(hr = spVideoDec->put_OutputCallback(static_cast<ICC_DataReadyCallback*>(new C_PPMWriter("frame%05d.ppm")))))
      return hr;

    // 4. Main cycle ------------------------------------------------
    for(;;)
    {
      static unsigned char buffer[65536];

      size_t dwBytesRead = fread(buffer, 1, sizeof(buffer), hSrcFile);

      if(dwBytesRead > 0)
      {
        CC_UINT dwBytesProcessed = 0;

        if(FAILED(hr = spVideoDec->ProcessData(buffer, (CC_UINT)dwBytesRead, 0, -1, &dwBytesProcessed)))
          return hr;
      }

      if(dwBytesRead != sizeof(buffer))
        break;
    }

	if (hSrcFile)
		fclose(hSrcFile);

	if(FAILED(hr = spVideoDec->Done(CC_TRUE)))
	  return hr;

	return 0;
}
