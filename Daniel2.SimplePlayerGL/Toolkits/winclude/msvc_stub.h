#ifndef __MSVC_STUB_H__
#define __MSVC_STUB_H__

#include <stdint.h>

#define __forceinline __inline__ __attribute__((always_inline))
#define __fastcall               __attribute__((fastcall))
#define __stdcall                __attribute__((stdcall))

static __inline__ unsigned _byteswap_ulong(unsigned x)
{
  __asm__ __volatile__
  (
    "bswapl	%0\n\t"
    : "=&r"(x) : "0"(x)
  );
  return x;
}

static __inline__ void __stosw(unsigned short *where, unsigned short val, unsigned int count)
{
#if 0
  __asm__ __volatile__
  (
    "rep\n\t"
    "stosw\n\t"
    : /* no output */
    : "D"(where), "a"(val), "c"(count)
    : "%edx", "%ecx"
  );
#else
  for(int i = 0; i < count; i++)
    where[i] = val;
#endif
}

static __inline__ long long Int32x32To64(int a, int b)
{
  return ((long long)(a)) * b;
}

static __inline__ unsigned long long UInt32x32To64(unsigned a, unsigned b)
{
  return ((unsigned long long)(a)) * b;
}

/* _countof helper */
#if !defined(_countof)
#if !defined(__cplusplus)
#define _countof(_Array) (sizeof(_Array) / sizeof(_Array[0]))
#else
extern "C++"
{
template <typename _CountofType, int _SizeOfArray>
char (*__countof_helper(_CountofType (&_Array)[_SizeOfArray]))[_SizeOfArray];
#define _countof(_Array) sizeof(*__countof_helper(_Array))
}
#endif
#endif

#define NOMINMAX

#ifndef NOMINMAX

#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

#endif

#ifndef __max
#define __max(a,b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef __min
#define __min(a,b) (((a) < (b)) ? (a) : (b))
#endif

#if !defined(__INTEL_COMPILER)

  #if !defined(__int64)
    #define __int64 long long
  #endif

  #if !defined(__int32)
    #define __int32 int
  #endif

#endif

#ifndef _I64_MAX
#define _I64_MAX  9223372036854775807LL
#endif

#ifndef _I64_MIN
#define _I64_MIN -9223372036854775808LL
#endif

#define wcstombs_s(pretval,mbstr,sizeinbytes,wcstr,count) assert(0)

#define strcpy_s    strcpy
#define sprintf_s   sprintf
#define vsnprintf_s vsnprintf

#define _strdup     strdup
#define  stricmp    strcasecmp
#define _stricmp    strcasecmp
#define  strnicmp   strncasecmp
#define _strnicmp   strncasecmp

#define _tmain      main

#define _tcslen     strlen
#define _tcscpy     strcpy
#define _tcscat     strcat
#define _tcschr     strchr
#define _tcsrchr    strrchr
#define _tcsstr     strstr

#define _tstoi      atoi
#define _tstoi64    atoll
#define _tstof      atof

#define _putts      puts

#define _tprintf    printf
#define _stprintf   sprintf
#define _ftprintf   fprintf
#define _vstprintf  vsprintf
#define _tscanf     scanf
#define _stscanf    sscanf
#define _ftscanf    fscanf

#define _tfopen     fopen

#define _access     access
#define _taccess    access
#define _tchmod     chmod
#define _tunlink    unlink
#define _open       open
#define _topen      open
#define _tsopen     open
#define _read       read
#define _write      write
#define _close      close
#define _commit     fsync
#define _fileno(f)  fileno(f)
#define _tsplitpath _splitpath

#ifdef __APPLE__

#define _lseeki64   lseek
#define _fseeki64   fseek
#define _ftelli64   ftell
#define O_LARGEFILE 0

#else

#define _lseeki64   lseek64
#define _fseeki64   fseek     /* think! */
#define _ftelli64   ftell     /* think! */

#endif

#define O_BINARY O_LARGEFILE

#define _telli64(f) _lseeki64(f,0,SEEK_CUR)

#include <stdio.h>
#include <unistd.h>

#if __APPLE__
#include <stdlib.h>
#else
#include <malloc.h>
#endif

inline long long _filelengthi64(int f)
{
  long long old = _lseeki64(f, 0, SEEK_CUR);
  if(old < 0) return old;
  long long len = _lseeki64(f, 0, SEEK_END);
  _lseeki64(f, old, SEEK_SET);
  return len;
}

inline int eof(int f)
{
  long long old = _lseeki64(f, 0, SEEK_CUR);
  if(old < 0) return old;
  long long len = _lseeki64(f, 0, SEEK_END);
  _lseeki64(f, old, SEEK_SET);
  return int(old >= len);
}

void *_aligned_malloc(size_t size, size_t align_size);
void _aligned_free(void *aligned_ptr);

void _splitpath(const char *path, char *drive, char *dir, char *fname, char *ext);

#endif
