#pragma once

#if defined(__LINUX__)

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <fcntl.h>
#include <linux/fb.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <string.h>

class CFrameBuffer
{
private:
	long int screensize;
	char *fbp;
	int fbfd;

public:
	CFrameBuffer()
	{
		screensize = 0;
		fbp = nullptr;
		fbfd = 0;
	}
	~CFrameBuffer()
	{
		Destroy();
	}
	int Init()
	{
		struct fb_var_screeninfo vinfo;
		struct fb_fix_screeninfo finfo;
		int x = 0, y = 0;
		long int location = 0;

		// Open the file for reading and writing
		fbfd = open("/dev/fb0", O_RDWR);
		if (fbfd == -1) {
			perror("Error: cannot open framebuffer device");
			//exit(1);
			return -1;
		}
		printf("The framebuffer device was opened successfully.\n");

		// Get fixed screen information
		if (ioctl(fbfd, FBIOGET_FSCREENINFO, &finfo) == -1) {
			perror("Error reading fixed information");
			//exit(2);
			return -2;
			}

		// Get variable screen information
		if (ioctl(fbfd, FBIOGET_VSCREENINFO, &vinfo) == -1) {
			perror("Error reading variable information");
			//exit(3);
			return -2;
		}

		printf("%dx%d, %dbpp\n", vinfo.xres, vinfo.yres, vinfo.bits_per_pixel);

		// Figure out the size of the screen in bytes
		screensize = vinfo.xres * vinfo.yres * vinfo.bits_per_pixel / 8;

		// Map the device to memory
		fbp = (char *)mmap(0, screensize, PROT_READ | PROT_WRITE, MAP_SHARED, fbfd, 0);
		if (fbp == nullptr) {
			perror("Error: failed to map framebuffer device to memory");
			//exit(4);
			return -4;
		}
		printf("The framebuffer device was mapped to memory successfully.\n");

		return 0;
	}
	void Destroy()
	{
		if (fbp)
		{
			memset(fbp, 0x00, screensize);
			munmap(fbp, screensize);
		}
		if (fbfd)
			close(fbfd);

		fbp = nullptr;
		screensize = 0;
		fbfd = 0;
	}

	unsigned char* GetPtr() { return (unsigned char*)(fbp); }
	size_t SizeBuffer() { return (size_t)(screensize); }
};

#endif
