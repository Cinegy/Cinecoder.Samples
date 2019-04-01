#pragma once

#ifdef _WIN32
#include <windows.h>
typedef HANDLE file_handle_t;
#define INVALID_FILE_HANDLE INVALID_HANDLE_VALUE
#else
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
typedef int file_handle_t;
#define INVALID_FILE_HANDLE (-1)
#endif

class C_ReadFile
{
private:
	file_handle_t hFile;

public:
	C_ReadFile() { hFile = INVALID_FILE_HANDLE; }

	int OpenFile(const char* filename, bool unbuffered)
	{
		CloseFile();

#ifdef _WIN32
		hFile = CreateFileA(filename, GENERIC_READ, FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, unbuffered ? FILE_FLAG_NO_BUFFERING : 0, NULL);
#else
#ifdef __APPLE__
#define O_DIRECT 0
#endif
		hFile = open(filename, O_BINARY | O_RDONLY | (unbuffered ? (O_DIRECT | O_SYNC) : 0), S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);
#endif
		if (!hFile)
			return -1;

		return 0;
	}

	void CloseFile()
	{
		if (hFile)
#ifdef _WIN32
			CloseHandle(hFile);
#else
			close(hFile);
#endif
		hFile = INVALID_FILE_HANDLE;
	}

	bool isValid() { return hFile != INVALID_FILE_HANDLE; }

	bool ReadFile(void *buf, DWORD count, DWORD *actual)
	{
#ifdef _WIN32
		return !!::ReadFile(hFile, buf, count, actual, NULL);
#else
		*actual = 0;

		int r = read(hFile, buf, count);

		if (r < 0)
			return false;

		*actual = r;

		return true;
#endif
	}

	bool SetFilePos(uintptr_t offset)
	{
#ifdef _WIN32
		return !!SetFilePointer(hFile, (LONG)offset, ((LONG*)&offset) + 1, FILE_BEGIN);
#else
		return _lseeki64(hFile, offset, SEEK_SET);
#endif
	}
};
