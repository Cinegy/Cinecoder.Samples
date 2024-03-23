//---------------------------------------------------------------------
const char *GetProcessorName()
//---------------------------------------------------------------------
{
  static char processor_name[8 * 1024] = "<no info>";
#ifdef _WIN32
  DWORD reg_size = sizeof(processor_name);
  RegGetValueA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", "ProcessorNameString", RRF_RT_REG_SZ, NULL, processor_name, &reg_size);
#else
  if(FILE *fversion = fopen("/proc/cpuinfo", "rt"))
  {
    for(int i = 0; i < 8; i++)
    {
      fgets(processor_name, sizeof(processor_name), fversion);
      if(0 == strnicmp(processor_name, "model name", 10))
        break;
    }
    
    if(auto lastcr = strrchr(processor_name, '\n'))
      *lastcr = 0;

    if(auto colon = strchr(processor_name, ':'))
      strcpy(processor_name, colon+2);

    fclose(fversion);
  }
#endif

  return processor_name;
}

//---------------------------------------------------------------------
const char *GetPlatformName()
//---------------------------------------------------------------------
{
  static char platform_descr[8 * 1024] = "<no info>";
#ifdef _WIN32
  static char prod_name[1024] = {};
  DWORD reg_size = sizeof(prod_name);
  RegGetValueA(HKEY_LOCAL_MACHINE, "Software\\Microsoft\\Windows NT\\CurrentVersion", "ProductName", RRF_RT_REG_SZ, NULL, prod_name, &reg_size);

  static char edition[1024] = {};
  reg_size = sizeof(edition);
  RegGetValueA(HKEY_LOCAL_MACHINE, "Software\\Microsoft\\Windows NT\\CurrentVersion", "EditionID", RRF_RT_REG_SZ, NULL, edition, &reg_size);

  static char composition[1024] = {};
  reg_size = sizeof(composition);
  RegGetValueA(HKEY_LOCAL_MACHINE, "Software\\Microsoft\\Windows NT\\CurrentVersion", "CompositionEditionID", RRF_RT_REG_SZ, NULL, composition, &reg_size);

  static char disp_version[1024] = {};
  reg_size = sizeof(disp_version);
  RegGetValueA(HKEY_LOCAL_MACHINE, "Software\\Microsoft\\Windows NT\\CurrentVersion", "DisplayVersion", RRF_RT_REG_SZ, NULL, disp_version, &reg_size);

  static char build_lab_ex[1024] = {};
  reg_size = sizeof(build_lab_ex);
  RegGetValueA(HKEY_LOCAL_MACHINE, "Software\\Microsoft\\Windows NT\\CurrentVersion", "BuildLabEx", RRF_RT_REG_SZ, NULL, build_lab_ex, &reg_size);

  if(*prod_name)
  {
	sprintf(platform_descr, "Microsoft %s", prod_name);
    if(*edition)
      sprintf(platform_descr + strlen(platform_descr), " %s", edition);
    if(*composition)
      sprintf(platform_descr + strlen(platform_descr), " %s", composition);
    if(*disp_version)
      sprintf(platform_descr + strlen(platform_descr), " %s", disp_version);
    if(*build_lab_ex)
      sprintf(platform_descr + strlen(platform_descr), " (%s)", build_lab_ex);
  }
#else
  if(FILE *fversion = fopen("/proc/version", "rt"))
  {
    fgets(platform_descr, sizeof(platform_descr), fversion);
    if(auto lastcr = strrchr(platform_descr, '\n'))
      *lastcr = 0;
    fclose(fversion);
  }
#endif

  return platform_descr;
}

//---------------------------------------------------------------------
template<class time_point_t>
const char* GetTimeStr(const time_point_t &tp)
//---------------------------------------------------------------------
{
  static char buf[128];

  auto t = std::chrono::system_clock::to_time_t(tp);
  struct tm timeinfo = *localtime(&t);
  sprintf(buf, "%04d-%02d-%02d %02d:%02d:%02d.%03d",
	  	timeinfo.tm_year+1900, timeinfo.tm_mon+1, timeinfo.tm_mday, timeinfo.tm_hour, timeinfo.tm_min, timeinfo.tm_sec,
	    (int)duration_cast<milliseconds>(tp.time_since_epoch()).count() % 1000);

  return buf;
}

//---------------------------------------------------------------------
const char *GetGuidStr(const GUID &guid)
//---------------------------------------------------------------------
{
  static char buf[128];

  sprintf(buf, "{%08X-%04X-%04X-%02X%02X-%02X%02X%02X%02X%02X%02X}",
      guid.Data1, guid.Data2, guid.Data3,
      guid.Data4[0], guid.Data4[1], guid.Data4[2], guid.Data4[3],
      guid.Data4[4], guid.Data4[5], guid.Data4[6], guid.Data4[7]);

  return buf;
};

//---------------------------------------------------------------------
GUID GetClassGUID(IUnknown *pUnk)
//---------------------------------------------------------------------
{
  GUID guid = {};

  com_ptr<ICC_Object> pObj;
  pUnk->QueryInterface(IID_ICC_Object, (void**)&pObj);

  if(pObj)
    pObj->get_CLSID(&guid);

  return guid;
}

//---------------------------------------------------------------------
const char *GetClassNameA(IUnknown *pUnk)
//---------------------------------------------------------------------
{
  static char buf[128];

  com_ptr<ICC_Object> pObj;
  pUnk->QueryInterface(IID_ICC_Object, (void**)&pObj);

  if(pObj)
  {
    CC_STRING className;
    pObj->get_ClassName(&className);
#ifdef _WIN32
    wcstombs(buf, className, sizeof(buf)-1);
	SysFreeString(className);
#else
	strncpy(buf, className, sizeof(buf)-1);
	free(className);
#endif
  }
  else
  {
    strcpy(buf, "<no info>");
  }
  
  buf[sizeof(buf)-1] = 0;

  return buf;
}

//---------------------------------------------------------------------
const char *GetNormStr(const char *str)
//---------------------------------------------------------------------
{
  static char buf[8 * 1024];

  const char *s = str;
  char *d = buf;

  for(;;)
  {
    char c = *s++;
    
    *d++ = c;

    if(c == '\\')
      *d++ = '\\';

    if(!c) break;

    if(d - buf >= sizeof(buf))
    {
      d[sizeof(buf)-1] = 0;
      break;
    }
  }

  return buf;
}
