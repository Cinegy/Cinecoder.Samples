#ifdef _WIN32
typedef HANDLE file_handle_t;
#define INVALID_FILE_HANDLE INVALID_HANDLE_VALUE

file_handle_t open_file(LPCTSTR filename, bool unbuffered)
{
	return CreateFile(filename, GENERIC_READ, FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, unbuffered ? FILE_FLAG_NO_BUFFERING : 0, NULL);
}

void close_file(file_handle_t &h)
{
	CloseHandle(h);
	h = INVALID_FILE_HANDLE;
}

bool is_valid(const file_handle_t h)
{
	return h != INVALID_FILE_HANDLE;
}

bool read_file(file_handle_t h, void *buf, DWORD count, DWORD *actual)
{
	return !!ReadFile(h, buf, count, actual, NULL);
}

bool set_file_pos(file_handle_t h, uintptr_t offset)
{
	return !!SetFilePointer(h, (LONG)offset, ((LONG*)&offset)+1, FILE_BEGIN);
}

#else

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

typedef int file_handle_t;

#define INVALID_FILE_HANDLE (-1)

file_handle_t open_file(LPCTSTR filename, bool unbuffered)
{
    return open(filename, O_EXCL | (unbuffered ? O_DIRECT : 0), S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH |  S_IWOTH);
}

void close_file(file_handle_t &h)
{
	close(h);
	h = INVALID_FILE_HANDLE;
}

bool is_valid(const file_handle_t h)
{
	return h != INVALID_FILE_HANDLE;
}

bool read_file(file_handle_t h, void *buf, DWORD count, DWORD *actual)
{
	*actual = 0;
	
	int r = read(h, buf, count);

	if(r < 0)
	  return false;

    *actual = r;

    return true;
}

bool set_file_pos(file_handle_t h, uintptr_t offset)
{
	return _lseeki64(h, offset, SEEK_SET);
}

#endif
