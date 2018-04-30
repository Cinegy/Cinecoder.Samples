#ifndef __TCHAR_H__
#define __TCHAR_H__

#if 1

#define _MBCS

typedef char _TCHAR;
typedef _TCHAR TCHAR;
#define _T(x) x

#define _tcscmp   strcmp
#define _tcsncmp  strncmp
#define _tcsicmp  strcasecmp
#define _tcsnicmp strncasecmp

#define _tcsncpy  strncpy

#define _tstoi    atoi
#define _ttoi     atoi

#define _tfopen   fopen
#define _fgetts   fgets

#define _tcsupr(x)

#else

typedef wchar_t TCHAR;
#define _T(x) L ## x

#endif

#endif //__TCHAR_H__
