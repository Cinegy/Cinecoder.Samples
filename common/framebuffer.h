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
	long int pagesize;
	char *fbp;
	int fbfd;

	struct fb_var_screeninfo vinfo;
	struct fb_fix_screeninfo finfo;

	struct fb_var_screeninfo orig_vinfo;

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
			return -3;
		}

		// Store for reset (copy vinfo to vinfo_orig)
		memcpy(&orig_vinfo, &vinfo, sizeof(struct fb_var_screeninfo));

		printf("%dx%d, %dbpp\n", vinfo.xres, vinfo.yres, vinfo.bits_per_pixel);

		// Figure out the size of the screen in bytes
		//screensize = vinfo.xres * vinfo.yres * vinfo.bits_per_pixel / 8;
		screensize = finfo.smem_len;

		pagesize = finfo.line_length * vinfo.yres;

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

	void PrintInfo()
	{
		printf("vinfo.xres           = %u\n", vinfo.xres);
		printf("vinfo.yres           = %u\n", vinfo.yres);
		printf("vinfo.xres_virtual   = %u\n", vinfo.xres_virtual);
		printf("vinfo.yres_virtual   = %u\n", vinfo.yres_virtual);
		printf("vinfo.xoffset        = %u\n", vinfo.xoffset);
		printf("vinfo.yoffset        = %u\n", vinfo.yoffset);
		printf("vinfo.bits_per_pixel = %u\n", vinfo.bits_per_pixel);
		printf("vinfo.grayscale      = %x\n", vinfo.grayscale);
		printf("vinfo.nonstd         = %u\n", vinfo.nonstd);
		printf("vinfo.activate       = %u\n", vinfo.activate);
		printf("vinfo.height         = %x\n", vinfo.height);
		printf("vinfo.width          = %x\n", vinfo.width);
		printf("vinfo.accel_flags    = %x\n", vinfo.accel_flags);
		printf("vinfo.pixclock       = %u\n", vinfo.pixclock);
		printf("vinfo.left_margin    = %u\n", vinfo.left_margin);
		printf("vinfo.right_margin   = %u\n", vinfo.right_margin);
		printf("vinfo.upper_margin   = %u\n", vinfo.upper_margin);
		printf("vinfo.lower_margin   = %u\n", vinfo.lower_margin);
		printf("vinfo.hsync_len      = %u\n", vinfo.hsync_len);
		printf("vinfo.vsync_len      = %u\n", vinfo.vsync_len);
		printf("vinfo.sync           = %u\n", vinfo.sync);
		printf("vinfo.vmode          = %u\n", vinfo.vmode);
		printf("vinfo.rotate         = %x\n", vinfo.rotate);
		printf("vinfo.colorspace     = %x\n", vinfo.colorspace);

		printf("finfo.id             = %-16.16s\n", finfo.id);
		printf("finfo.smem_start     = 0x%lx\n", finfo.smem_start);
		printf("finfo.smem_len       = %u\n", finfo.smem_len);
		printf("finfo.type           = %u\n", finfo.type);
		printf("finfo.type_aux       = %u\n", finfo.type_aux);
		printf("finfo.visual         = %u\n", finfo.visual);
		printf("finfo.xpanstep       = %u\n", finfo.xpanstep);
		printf("finfo.ypanstep       = %u\n", finfo.ypanstep);
		printf("finfo.ywrapstep      = %u\n", finfo.ywrapstep);
		printf("finfo.line_length    = %u\n", finfo.line_length);
		printf("finfo.mmio_start     = 0x%lx\n", finfo.mmio_start);
		printf("finfo.mmio_len       = %u\n", finfo.mmio_len);
		printf("finfo.accel          = %u\n", finfo.accel);
		printf("finfo.capabilities   = %u\n", finfo.capabilities);
	}

	void Destroy()
	{
		// reset the display mode
		if (ioctl(fbfd, FBIOPUT_VSCREENINFO, &orig_vinfo)) {
			printf("Error re-setting variable information.\n");
		}

		if (fbp)
		{
			memset(fbp, 0x00, screensize);
			munmap(fbp, screensize);
		}
		if (fbfd)
			close(fbfd);

		fbp = nullptr;
		screensize = 0;
		pagesize = 0;
		fbfd = 0;
	}

	unsigned char* GetPtr() { return (unsigned char*)(fbp); }
	size_t SizeBuffer() { return (size_t)(pagesize); }
	fb_var_screeninfo GetVInfo() { return vinfo; }

	int DisplayBuffer(size_t y_offset)
	{
		vinfo.yoffset = y_offset;

		/* Swap the working buffer for the displayed buffer */
		if (ioctl(fbfd, FBIOPAN_DISPLAY, &vinfo) == -1) {
			printf("Failed FBIOPAN_DISPLAY (%s)\n", strerror(errno));
			return -1;
		}

		return 0;
	}
};

#endif
