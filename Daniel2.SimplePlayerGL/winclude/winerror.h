#ifndef __WINSTUB_WINERROR_H__
#define __WINSTUB_WINERROR_H__

typedef int HRESULT;

#ifdef RC_INVOKED
#define _HRESULT_TYPEDEF_(_sc) _sc
#else // RC_INVOKED
#define _HRESULT_TYPEDEF_(_sc) ((HRESULT)_sc)
#endif // RC_INVOKED

#define S_OK                             0
#define S_FALSE                          1
#define STILL_ACTIVE                     256

#define NOERROR                          0
#define ERROR_FILE_NOT_FOUND             2L
#define ERROR_PATH_NOT_FOUND             3L
#define ERROR_ACCESS_DENIED              5L
#define ERROR_INVALID_HANDLE             6L
#define ERROR_INVALID_PARAMETER          87L
#define ERROR_OUTOFMEMORY                14L
#define ERROR_INSUFFICIENT_BUFFER        122L
#define ERROR_MOD_NOT_FOUND              126L
#define ERROR_PROC_NOT_FOUND             127L

#define ERROR_INVALID_INDEX              1413L
#define ERROR_INVALID_THREAD_ID          1444L
#define ERROR_TIMEOUT                    1460L
#define ERROR_INCORRECT_SIZE             1462L
#define ERROR_XML_PARSE_ERROR            1465L
#define ERROR_CANT_ACCESS_FILE           1920L

#define CLASS_E_CLASSNOTAVAILABLE        _HRESULT_TYPEDEF_(0x80040111L)

//  Catastrophic failure
#define E_UNEXPECTED                     _HRESULT_TYPEDEF_(0x8000FFFFL)
//  Not implemented
#define E_NOTIMPL                        _HRESULT_TYPEDEF_(0x80004001L)
//  Ran out of memory
#define E_OUTOFMEMORY                    _HRESULT_TYPEDEF_(0x8007000EL)
//  One or more arguments are invalid
#define E_INVALIDARG                     _HRESULT_TYPEDEF_(0x80070057L)
//  No such interface supported
#define E_NOINTERFACE                    _HRESULT_TYPEDEF_(0x80004002L)
//  Invalid pointer
#define E_POINTER                        _HRESULT_TYPEDEF_(0x80004003L)
//  Invalid handle
#define E_HANDLE                         _HRESULT_TYPEDEF_(0x80070006L)
//  Operation aborted
#define E_ABORT                          _HRESULT_TYPEDEF_(0x80004004L)
//  Unspecified error
#define E_FAIL                           _HRESULT_TYPEDEF_(0x80004005L)
//  General access denied error
#define E_ACCESSDENIED                   _HRESULT_TYPEDEF_(0x80070005L)
//  The data necessary to complete this operation is not yet available.
#define E_PENDING                        _HRESULT_TYPEDEF_(0x8000000AL)

#define SEVERITY_SUCCESS    0
#define SEVERITY_ERROR      1

#define FACILITY_NT_BIT     0x10000000
#define FACILITY_ITF        4
#define FACILITY_WIN32      7

#define SUCCEEDED(hr) (((HRESULT)(hr)) >= 0)
#define FAILED(hr) (((HRESULT)(hr)) < 0)

#define IS_ERROR(Status) (((unsigned int)(Status)) >> 31 == SEVERITY_ERROR)

#define HRESULT_CODE(hr)    ((hr) & 0xFFFF)
#define SCODE_CODE(sc)      ((sc) & 0xFFFF)

#define HRESULT_FACILITY(hr)  (((hr) >> 16) & 0x1fff)
#define SCODE_FACILITY(sc)    (((sc) >> 16) & 0x1fff)

#define HRESULT_SEVERITY(hr)  (((hr) >> 31) & 0x1)
#define SCODE_SEVERITY(sc)    (((sc) >> 31) & 0x1)

#define MAKE_HRESULT(sev,fac,code) \
    ((HRESULT) (((unsigned int)(sev)<<31) | ((unsigned int)(fac)<<16) | ((unsigned int)(code))) )
#define MAKE_SCODE(sev,fac,code) \
    ((SCODE) (((unsigned int)(sev)<<31) | ((unsigned int)(fac)<<16) | ((unsigned int)(code))) )

#define __HRESULT_FROM_WIN32(x) ((HRESULT)(x) <= 0 ? ((HRESULT)(x)) : ((HRESULT) (((x) & 0x0000FFFF) | (FACILITY_WIN32 << 16) | 0x80000000)))

#ifndef __midl
inline HRESULT HRESULT_FROM_WIN32(unsigned x) { return (HRESULT)(x) <= 0 ? (HRESULT)(x) : (HRESULT) (((x) & 0x0000FFFF) | (FACILITY_WIN32 << 16) | 0x80000000);}
#else
#define HRESULT_FROM_WIN32(x) __HRESULT_FROM_WIN32(x)
#endif

#endif // __WINSTUB_WINERROR_H__
