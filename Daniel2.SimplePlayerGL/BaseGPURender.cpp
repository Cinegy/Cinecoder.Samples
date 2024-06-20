#include "stdafx.h"
#include "BaseGPURender.h"

BaseGPURender::BaseGPURender() :
	m_bPause(false),
	m_bVSync(false),
	m_bRotate(false),
	m_bFullScreen(false),
	m_bFullCurScreen(false),
	m_bUseGPU(false),
	m_windowCaption(L"TestApp (Decode Daniel2)")
{
	// Init window size
	window_width = 512;
	window_height = 512;

	// Init image size
	image_width = 512;
	image_height = 512;
	image_size = image_width * image_height * 4;

	fpsCount = 0;
	ValueFPS = 60.0;

	m_bInitRender = false;

	iAllFrames = 1;
	iCurPlayFrameNumber = 0;

	m_bLastRotate = m_bRotate;

	m_decodeD2 = nullptr;
	m_decodeAudio = nullptr;

	cuda_tex_result_resource = nullptr;
	cuda_tex_result_resource_buff = nullptr;

	m_bShowSlider = false;
	edgeLineX = 20.f;
	edgeLineY = 60.f;
	sizeSquare = 20.f;
	sizeSquare2 = sizeSquare / 2;

	m_bCopyToTexture = true;
	m_bShowTexture = true;
	m_bDecoder = true;

	m_bMaxFPS = false;
	m_bVSyncHand = true;

	gpu_render_type = GPU_RENDER_UNKNOWN;
}

BaseGPURender::~BaseGPURender()
{
	StopPipe();
}

int BaseGPURender::SetParameters(bool bVSync, bool bRotate, bool bMaxFPS)
{
	m_bVSync = bVSync;
	m_bRotate = bRotate;
	m_bLastRotate = m_bRotate;
	m_bMaxFPS = bMaxFPS;

	SetVerticalSync(m_bVSync);

	return 0;
}

int BaseGPURender::Init(std::string filename, ST_VIDEO_DECODER_PARAMS dec_params, size_t gpuDevice)
{
	m_bUseGPU = dec_params.type == VD_TYPE_CUDA ? true : false;

	m_decodeD2 = std::make_shared<DecodeDaniel2>(); // Create decoder for decoding DN2 files

	if (!m_decodeD2)
	{
		printf("Cannot create create decoder!\n");
		return -1;
	}

	if (gpu_render_type == GPU_RENDER_D3DX11)
	{
		HRESULT hr = S_OK;

		IDXGIFactory* pDXGIFactory;
		CreateDXGIFactory(IID_IDXGIFactory, (void**)&pDXGIFactory);

		m_pCapableAdapter = nullptr;

		for (UINT adapter = 0; pDXGIFactory; adapter++)
		{
			// Get a candidate DXGI adapter
			com_ptr<IDXGIAdapter> pAdapter = nullptr;
			hr = pDXGIFactory->EnumAdapters(adapter, (IDXGIAdapter**)&pAdapter);

			if (!pAdapter)
				break;

			DXGI_ADAPTER_DESC adapterDesc;
			pAdapter->GetDesc(&adapterDesc);

			std::wstring dev_name(adapterDesc.Description);

			if ( (gpuDevice == 0) || 
				 (gpuDevice == 1 && dev_name.find(L"NVIDIA") != std::string::npos) || 
				 (gpuDevice == 2 && dev_name.find(L"Intel") != std::string::npos)
				)
			{
				pAdapter->QueryInterface(IID_IDXGIAdapter1, (void**)&m_pCapableAdapter);
				break;
			}
		}

		if (dec_params.use_cinecoder_d3d11)
		{
			m_decodeD2->InitD3DX11Render((GPURenderDX*)this);
			m_decodeD2->InitD3DXAdapter(m_pCapableAdapter);
		}
	}

	int res = m_decodeD2->OpenFile(filename.c_str(), dec_params);

	if (res != 0)
	{
		printf("Cannot open input file <%s> or create decoder!\n", filename.c_str());
		return res;
	}

	image_width = (unsigned int)m_decodeD2->GetImageWidth();
	image_height = (unsigned int)m_decodeD2->GetImageHeight();

	CC_RATIONAL AspectRatio = m_decodeD2->GetAspectRatio();

	unsigned int image_width_ = image_width;
	unsigned int image_height_ = (unsigned int)((float)image_width_ * (float)AspectRatio.denom / (float)AspectRatio.num);

	int iWinW = GetSystemMetrics(SM_CXSCREEN);
	int iWinH = GetSystemMetrics(SM_CYSCREEN);

	float fKoeffDiv = (float)image_width_ / ((float)iWinW / 3.f);
	// Correction start of window size using global height of monitor
	while (window_height >= iWinH)
		fKoeffDiv *= 1.5f;

	window_width = static_cast<unsigned int>((float)(image_width_) / fKoeffDiv);
	window_height = static_cast<unsigned int>((float)(image_height_) / fKoeffDiv);

	//window_width = 1920; window_height = 1080;

	iAllFrames = m_decodeD2->GetCountFrames(); // Get count of frames

	ValueFPS = m_decodeD2->GetFrameRate(); // get frame rate

	InitAudioTrack(filename, m_decodeD2->GetFrameRateValue());
		
	return 0;
}

int BaseGPURender::InitAudioTrack(std::string filename, CC_FRAME_RATE frameRate)
{
	int res = 0;

	m_decodeAudio = std::make_shared<AudioSource>(); // Create decoder for audio track

	if (!m_decodeAudio)
	{
		printf("Cannot create create audio decoder!\n");
		return 0;
	}

	res = m_decodeAudio->Init(frameRate); // Init audio decoder

	if (res != 0)
		return res;

	res = m_decodeAudio->OpenFile(filename.c_str()); // Open audio stream

	if (res == 0)
		printf("Audio track: Yes\n");
	else
		printf("Audio track: No (error = 0x%x)\n", res);

	printf("-------------------------------------\n");

	return res;
}

int BaseGPURender::StartPipe()
{
	ThreadProc();

	return 0;
}

int BaseGPURender::StopPipe()
{
	m_evExit.Set();

	return 0;
}

LRESULT CALLBACK BaseGPURender::WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	BaseGPURender *_this = (BaseGPURender*)GetProp(hWnd, L"This");

	if (!_this)
		return DefWindowProc(hWnd, msg, wParam, lParam);

	return _this->ProcessWndProc(hWnd, msg, wParam, lParam);
}

LRESULT BaseGPURender::ProcessWndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	WORD x, y;

	x = LOWORD(lParam);
	y = HIWORD(lParam);

	RECT rc;
	GetClientRect(m_hWnd, &rc);
	UINT width = rc.right - rc.left;    // получаем ширину
	UINT height = rc.bottom - rc.top;   // и высоту окна

	switch (msg)
	{
	case WM_KEYUP:
	{
		if (wParam == 27) // "ESC"
		{
			StopPipe();
		}
		else if (wParam == 107 || wParam == 109) // "+" "-"
		{
			if (m_decodeAudio && m_decodeAudio->IsInitialize())
			{
				float volume = m_decodeAudio->GetVolume() + (wParam == 107 ? 0.1f : -0.1f);
				if (volume > 1.f) volume = 1.f;
				else if (volume < 0.f) volume = 0;
				m_decodeAudio->SetVolume(volume);
				printf("audio volume = %.0f %%\n", m_decodeAudio->GetVolume() * 100.f);
			}
			break;
		}
		else if (wParam == 80 || wParam == VK_SPACE) // "p"
		{
			if (m_bPause)
				SeekToFrame(iCurPlayFrameNumber);

			SetPause(!m_bPause);
			break;
		}
		else if (wParam == 74) // "j"
		{
			int iSpeed = m_decodeD2->GetSpeed();

			if (iSpeed < 0)
				m_decodeD2->SetSpeed(iSpeed * 2);
			else
				m_decodeD2->SetSpeed(-1);

			if (m_bPause)
			{
				SeekToFrame(iCurPlayFrameNumber);
				SetPause(!m_bPause);
			}

			if (m_decodeAudio && m_decodeAudio->IsInitialize())
				m_decodeAudio->SetSpeed(m_decodeD2->GetSpeed());

			printf("press J (speed: %dx)\n", m_decodeD2->GetSpeed());
			break;
		}
		else if (wParam == 75) // "k"
		{
			int iSpeed = m_decodeD2->GetSpeed();

			if (iSpeed > 0)
				m_decodeD2->SetSpeed(1);
			else
				m_decodeD2->SetSpeed(-1);

			if (m_bPause)
				SeekToFrame(iCurPlayFrameNumber);

			SetPause(!m_bPause);

			if (m_decodeAudio && m_decodeAudio->IsInitialize())
				m_decodeAudio->SetSpeed(m_decodeD2->GetSpeed());

			printf("press K (speed: %dx)\n", m_decodeD2->GetSpeed());
			break;
		}
		else if (wParam == 76) // "l"
		{
			int iSpeed = m_decodeD2->GetSpeed();

			if (iSpeed > 0)
				m_decodeD2->SetSpeed(iSpeed * 2);
			else
				m_decodeD2->SetSpeed(1);

			if (m_bPause)
			{
				SeekToFrame(iCurPlayFrameNumber);
				SetPause(!m_bPause);
			}

			if (m_decodeAudio && m_decodeAudio->IsInitialize())
				m_decodeAudio->SetSpeed(m_decodeD2->GetSpeed());

			printf("press L (speed: %dx)\n", m_decodeD2->GetSpeed());
			break;
		}
		else if (wParam == 86) // "v"
		{
			m_bVSync = !m_bVSync;
			SetVerticalSync(m_bVSync);

			if (m_bVSync)
				printf("vertical synchronisation: on\n");
			else
				printf("vertical synchronisation: off\n");

			break;
		}
		else if (wParam == 77) // "m"
		{
			m_bMaxFPS = !m_bMaxFPS;

			if (m_bMaxFPS)
				printf("maximum playing fps: on\n");
			else
				printf("maximum playing fps: off\n");

			break;
		}
		else if (wParam == 82) // "r"
		{
			m_bRotate = !m_bRotate;

			if (m_bRotate)
				printf("rotate image: on\n");
			else
				printf("rotate image: off\n");

			break;
		}
		else if (wParam == 70) // "f"
		{
			m_bFullScreen = !m_bFullScreen;
			UpdateWindow();

			if (m_bFullScreen)
				printf("fullscreen mode: on\n");
			else
				printf("fullscreen mode: off\n");

			break;
		}
		else if (wParam == 67) // "c"
		{
			if (!m_bFullScreen)
			{
				m_bFullCurScreen = !m_bFullCurScreen;
				UpdateWindow();
			}

			if (m_bFullScreen)
				printf("current screen mode: on\n");
			else
				printf("current screen mode: off\n");

			break;
		}
		else if (wParam == 84) // "t"
		{
			m_bCopyToTexture = !m_bCopyToTexture;

			if (m_bCopyToTexture)
				printf("copy result to texture: on\n");
			else
				printf("copy result to texture: off\n");

			break;
		}
		else if (wParam == 79) // "o"
		{
			m_bShowTexture = !m_bShowTexture;

			if (m_bShowTexture)
				printf("show texture: on\n");
			else
				printf("show texture: off\n");

			break;
		}
		else if (wParam == 68) // "d"
		{
			m_bDecoder = !m_bDecoder;
			m_decodeD2->SetDecode(m_bDecoder);
			
			if (m_bDecoder)	
				m_decodeD2->SeekFrame(iCurPlayFrameNumber);

			if (m_bDecoder)
				printf("decoder: on\n");
			else
				printf("decoder: off\n");

			break;
		}
		else if (wParam == 78) // "n"
		{
			bool bReadFile = m_decodeD2->GetReadFile();
			m_decodeD2->SetReadFile(!bReadFile);

			bReadFile = m_decodeD2->GetReadFile();

			if (bReadFile)
				printf("read file: on\n");
			else
				printf("read file: off\n");

			break;
		}
		else if (wParam == VK_HOME) // "VK_HOME"
		{
			SeekToFrame(0);
			break;
		}
		else if (wParam == VK_END) // "VK_END"
		{
			SeekToFrame(iAllFrames - 1);
			break;
		}
		else if (wParam == VK_RIGHT) // "VK_RIGHT"
		{
			if (iCurPlayFrameNumber <= iAllFrames - 2)
				SeekToFrame(iCurPlayFrameNumber + 1);
			break;
		}
		else if (wParam == VK_LEFT) // "VK_LEFT"
		{
			if (iCurPlayFrameNumber >= 1)
				SeekToFrame(iCurPlayFrameNumber - 1);
			break;
		}

		break;
	}

	case WM_MOUSEMOVE:
	{
		float fStartY = (float)height - (edgeLineY * 2.f);
		float fStopY = (float)height;

		if (((float)y >= fStartY) && ((float)y <= fStopY))
		{
			m_bShowSlider = true;
		}
		else
		{
			m_bShowSlider = false;
		}

		//Check the mouse left button is pressed or not
		if ((GetKeyState(VK_LBUTTON) & 0x80) != 0)
		{
			if (m_bShowSlider)
			{
				SeekToFrame(x, y);
				RenderWindow(); // update frame to improve performance of scrubbing
			}
		}

		break;
	}

	case WM_LBUTTONDOWN:
	{
		if (m_bShowSlider)
		{
			SeekToFrame(x, y);
			RenderWindow(); // update frame to improve performance of scrubbing
		}
		else
		{
			if (m_bPause)
				SeekToFrame(iCurPlayFrameNumber);

			SetPause(!m_bPause);
		}

		break;
	}

	case WM_SIZE:
	{
		RECT r;
		GetClientRect(m_hWnd, &r);
		window_width = (r.right - r.left);
		window_height = (r.bottom - r.top);
		break;
	}

	case WM_CLOSE:
	{
		StopPipe();
		break;
	}

	default:
		return DefWindowProc(hWnd, msg, wParam, lParam);

	}

	return 0;
}

int BaseGPURender::UpdateWindow()
{
	int wndWidth, wndHeight, wndX, wndY;

	if (m_bFullScreen || m_bFullCurScreen)
	{
		wndWidth = GetSystemMetrics(SM_CXSCREEN);
		wndHeight = GetSystemMetrics(SM_CYSCREEN);
		wndX = 0;
		wndY = 0;

		if (m_bFullScreen)
		{
			wndWidth = GetSystemMetrics(SM_CXVIRTUALSCREEN);
			wndHeight = GetSystemMetrics(SM_CYVIRTUALSCREEN);
			wndX = GetSystemMetrics(SM_XVIRTUALSCREEN);
			wndY = GetSystemMetrics(SM_YVIRTUALSCREEN);
		}

		window_width = wndWidth;
		window_height = wndHeight;

		RECT recMove;
		GetWindowRect(m_hWnd, &recMove);
		m_posWindow.left = recMove.left;
		m_posWindow.top = recMove.top;
		m_posWindow.right = recMove.right - recMove.left;
		m_posWindow.bottom = recMove.bottom - recMove.top;

		SetWindowLong(m_hWnd, GWL_STYLE, WS_VISIBLE | WS_SYSMENU);
		SetWindowLong(m_hWnd, GWL_EXSTYLE, WS_EX_APPWINDOW | WS_EX_ACCEPTFILES);
		SetWindowPos(m_hWnd, NULL, wndX, wndY, wndWidth, wndHeight, SWP_NOACTIVATE);
	}
	else
	{
		window_width = m_posWindow.right;
		window_height = m_posWindow.bottom;

		wndX = m_posWindow.left;
		wndY = m_posWindow.top;
		wndWidth = m_posWindow.right;
		wndHeight = m_posWindow.bottom;

		SetWindowLong(m_hWnd, GWL_STYLE, WS_CAPTION | WS_SYSMENU | WS_VISIBLE | WS_SIZEBOX | WS_MAXIMIZEBOX | WS_MINIMIZEBOX);
		SetWindowLong(m_hWnd, GWL_EXSTYLE, WS_EX_APPWINDOW | WS_EX_ACCEPTFILES);
		SetWindowPos(m_hWnd, NULL, wndX, wndY, wndWidth, wndHeight, SWP_NOACTIVATE);
	}

	return 0;
}

int BaseGPURender::CreateWnd()
{
	const wchar_t * const strWindowName = m_windowCaption.c_str();
	const wchar_t * MyClassApp = L"Cinegy TestApp window class";

	HINSTANCE hInstance = GetModuleHandle(0);

	WNDCLASSEX wndclass = { sizeof(WNDCLASSEX), CS_DBLCLKS, WndProc,
		0, 0, hInstance, LoadIcon(hInstance, NULL),
		LoadCursor(0, IDC_ARROW), HBRUSH(COLOR_WINDOW + 1),
		0, MyClassApp, LoadIcon(hInstance, NULL)
	};

	m_wndClass = wndclass;
	m_Instance = hInstance;

	if (RegisterClassEx(&wndclass))
	{
		DWORD wndStyle;

		m_posWindow.left = m_posWindow.top = 100;
		m_posWindow.right = window_width;
		m_posWindow.bottom = window_height;

		wndStyle = WS_CAPTION | WS_SYSMENU | WS_VISIBLE | WS_SIZEBOX | WS_MAXIMIZEBOX | WS_MINIMIZEBOX;
		m_hWnd = CreateWindowExW(WS_EX_APPWINDOW, MyClassApp, strWindowName, wndStyle, m_posWindow.left, m_posWindow.top, m_posWindow.right, m_posWindow.bottom, 0, 0, NULL, NULL);
	}

	if (!m_hWnd)
	{
		return -1;
	}
	else
	{
		SetProp(m_hWnd, L"This", (HANDLE)this);
		m_hDC = GetDC(m_hWnd);
	}

	RECT r;
	GetClientRect(m_hWnd, &r);
	window_width = (r.right - r.left);
	window_height = (r.bottom - r.top);

	return 0;
}

int BaseGPURender::DestroyWindow()
{
	if (m_hWnd)
	{
		RemoveProp(m_hWnd, L"This");
		::DestroyWindow(m_hWnd);
		UnregisterClass(m_wndClass.lpszClassName, m_Instance);
	}

	return 0;
}

int BaseGPURender::SetVerticalSync(bool bVerticalSync)
{
	return 0;
}

int BaseGPURender::GenerateImage(bool & bRotateFrame)
{
	return 0;
}

int BaseGPURender::GenerateCUDAImage(bool & bRotateFrame)
{
	if (!m_decodeD2->isProcess() || m_decodeD2->isPause()) // check for pause or process
		return 1;

	C_Block *pBlock = m_decodeD2->MapFrame(); // Get poiter to picture after decoding

	if (!pBlock)
		return -1;

	CopyCUDAImage(pBlock);

	bRotateFrame = pBlock->GetRotate() ? !bRotateFrame : bRotateFrame; // Rotate frame

	m_bLastRotate = pBlock->GetRotate(); // Save frame rotation value

	iCurPlayFrameNumber = pBlock->iFrameNumber; // Save currect frame number
	
	m_decodeD2->UnmapFrame(pBlock); // Add free pointer to queue

	return 0;
}

int BaseGPURender::CopyCUDAImage(C_Block *pBlock)
{
	if (!m_bCopyToTexture)
		return 0;
	
	MultithreadSyncBegin();

	// We want to copy image data to the texture
	// map buffer objects to get CUDA device pointers
	cudaArray *texture_ptr;
	cudaGraphicsMapResources(1, &cuda_tex_result_resource, 0); __vrcu
	cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_result_resource, 0, 0); __vrcu
#if defined(__CUDAConvertLib__)	
	ConvertMatrixCoeff iMatrixCoeff_YUYtoRGBA = (ConvertMatrixCoeff)(pBlock->iMatrixCoeff_YUYtoRGBA);

	IMAGE_FORMAT output_format = m_decodeD2->GetImageFormat();
	BUFFER_FORMAT buffer_format = m_decodeD2->GetBufferFormat();

	#define PARAMS pBlock->DataGPUPtr(), texture_ptr, (int)pBlock->Width(), (int)pBlock->Height(), (int)pBlock->Pitch(), NULL, iMatrixCoeff_YUYtoRGBA

	if (buffer_format == BUFFER_FORMAT_RGBA32 || buffer_format == BUFFER_FORMAT_RGBA64)
	{
		cudaMemcpy2DToArray(texture_ptr, 0, 0, pBlock->DataGPUPtr(), pBlock->Pitch(), (pBlock->Width() * bytePerPixel), pBlock->Height(), cudaMemcpyDeviceToDevice); __vrcu
	}
	else if (buffer_format == BUFFER_FORMAT_YUY2)
	{
		if (output_format == IMAGE_FORMAT_RGBA8BIT)
		{
			h_convert_YUY2_to_RGBA32_BtT(PARAMS); __vrcu
		}
		else if (output_format == IMAGE_FORMAT_BGRA8BIT)
		{
			h_convert_YUY2_to_BGRA32_BtT(PARAMS); __vrcu
		}
	}
	else if (buffer_format == BUFFER_FORMAT_Y216)
	{
		if (output_format == IMAGE_FORMAT_RGBA16BIT)
		{
			h_convert_Y216_to_RGBA64_BtT(PARAMS); __vrcu
		}
		else if (output_format == IMAGE_FORMAT_BGRA16BIT)
		{
			h_convert_Y216_to_BGRA64_BtT(PARAMS); __vrcu
		}
		else if (output_format == IMAGE_FORMAT_RGBA8BIT)
		{
			h_convert_Y216_to_RGBA32_BtT(PARAMS); __vrcu
		}
		else if (output_format == IMAGE_FORMAT_BGRA8BIT)
		{
			h_convert_Y216_to_BGRA32_BtT(PARAMS); __vrcu
		}
	}
	else if (buffer_format == BUFFER_FORMAT_NV12)
	{
		if (output_format == IMAGE_FORMAT_RGBA8BIT)
		{
			h_convert_NV12_to_RGBA32_BtT(PARAMS); __vrcu
		}
		else if (output_format == IMAGE_FORMAT_BGRA8BIT)
		{
			h_convert_NV12_to_BGRA32_BtT(PARAMS); __vrcu
		}
	}
	else if (buffer_format == BUFFER_FORMAT_P016)
	{
		if (output_format == IMAGE_FORMAT_RGBA16BIT)
		{
			h_convert_P016_to_RGBA64_BtT(PARAMS); __vrcu
		}
		else if (output_format == IMAGE_FORMAT_BGRA16BIT)
		{
			h_convert_P016_to_BGRA64_BtT(PARAMS); __vrcu
		}
	}
#else
	cudaMemcpy2DToArray(texture_ptr, 0, 0, pBlock->DataGPUPtr(), pBlock->Pitch(), pBlock->Pitch(), pBlock->Height(), cudaMemcpyDeviceToDevice); __vrcu
#endif
	// Unmap the resources
	cudaGraphicsUnmapResources(1, &cuda_tex_result_resource, 0); __vrcu

	MultithreadSyncEnd();

	return 0;
}

int BaseGPURender::MultithreadSyncBegin()
{
	if (gpu_render_type == GPU_RENDER_D3DX11)
		m_pMulty->Enter();

	return 0;
}

int BaseGPURender::MultithreadSyncEnd()
{
	if (gpu_render_type == GPU_RENDER_D3DX11)
		m_pMulty->Leave();

	return 0;
}

int BaseGPURender::ComputeFPS()
{
	// Update fps counter, fps/title display
	fpsCount++;
	double ms_elapsed = timer.GetElapsedTime();

	if (ms_elapsed > 1000.0f)
	{
		static bool bInit = true;

		double fps = (fpsCount / (ms_elapsed / 1000.0f));

		size_t data_rate = m_decodeD2->GetDataRate(true);
		double fDataRate = bInit ? 0.0, bInit = false : (data_rate * 1000) / ms_elapsed / (1024 * 1024);

		static wchar_t cString[256];
		std::wstring cTitle;
		swprintf_s(cString, L"%s (%d x %d): %.0f fps data_rate = %.2f MB/s", m_windowCaption.c_str(), window_width, window_height, fps, fDataRate);
		
		cTitle = cString;

		IMAGE_FORMAT img_frm = m_decodeD2->GetImageFormat();
		switch (img_frm)
		{
		case IMAGE_FORMAT_RGBA8BIT: cTitle += L" fmt=RGBA32"; break;
		case IMAGE_FORMAT_BGRA8BIT: cTitle += L" fmt=BGRA32"; break;
		case IMAGE_FORMAT_RGBA16BIT: cTitle += L" fmt=RGBA64"; break;
		case IMAGE_FORMAT_BGRA16BIT: cTitle += L" fmt=BGRA64"; break;
		case IMAGE_FORMAT_RGB30: cTitle += L" fmt=RGB30"; break;
		default: break;
		}

		cTitle += L" cur_frm=";
		cTitle += std::to_wstring((long long)iCurPlayFrameNumber); // print current frame number

		SetWindowText(m_hWnd, cTitle.c_str());
		
		fpsCount = 0;
		timer.StartTimer();
	}

	return 0;
}

void BaseGPURender::SetPause(bool bPause)
{
	m_bPause = bPause;

	if (m_bPause)
		printf("pause: on\n");
	else
		printf("pause: off\n");

	if (m_decodeAudio && m_decodeAudio->IsInitialize())
		m_decodeAudio->SetPause(m_bPause);
}

void BaseGPURender::SeekToFrame(size_t iFrame)
{
	C_AutoLock lock(&m_mutex);

	m_decodeD2->SeekFrame(iFrame); // Setting the reading of the input file from the expected frame (from frame number <iFrame>)

	size_t nReadFrame = 0;
	C_Block *pBlock = nullptr;

	C_Timer seek_timer;
	seek_timer.StartTimer();

	while (seek_timer.GetElapsedTime() <= 5000) // max wait 5 sec
	{
		pBlock = m_decodeD2->MapFrame(); // Get poiter to picture after decoding
		nReadFrame = pBlock->iFrameNumber; // Get currect frame number

		if (!pBlock)
			break;

		if (nReadFrame == iFrame) // Search for the expected frame
		{
			CopyBufferToTexture(pBlock);

			iCurPlayFrameNumber = iFrame; // Save currect frame number
			m_decodeD2->UnmapFrame(pBlock); // Add free pointer to queue
			SetPause(true); // Set pause enable
			printf("seek to frame %zu\n", iFrame);
			break;
		}

		if (pBlock)
			m_decodeD2->UnmapFrame(pBlock); // Add free pointer to queue
	}

	if (nReadFrame != iFrame) // The expected frame was not found
	{
		printf("Cannot seek to frame %zu\n", iFrame);
	}
}

void BaseGPURender::SeekToFrame(int x, int y)
{
	int w = window_width; // Width in pixels of the current window
	int h = window_height; // Height in pixels of the current window

	sizeSquare2 = (float)w / 100;
	edgeLineY = sizeSquare2 * 4;
	edgeLineX = sizeSquare2 * 2;

	int iStartX = (int)0;
	int iStopX = (int)(w - (int)(edgeLineX * 2));
	int iStartY = (int)(h - ((int)edgeLineY * 2));
	int iStopY = (int)h;

	x -= (int)edgeLineX;

	if ((x >= iStartX) && (x <= iStopX) &&
		(y >= iStartY) && (y <= iStopY)) // Get the frame number based on the coordinates (x,y)
	{
		size_t iFrame = (size_t)(((float)x * (float)(iAllFrames - 1)) / ((float)w - (2.f * edgeLineX)));

		SeekToFrame(iFrame); // Seek to frame number <iFrame>
	}
}

DWORD BaseGPURender::ThreadProc()
{
	int res = 0;

	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);

	if (CreateWnd() != 0) // Create window
		return 0;

	if (InitRender() != 0) // Init render
		return 0;

	// Start timers
	timer.StartTimer();
	timerqFPSMode.StartTimer();

	if (m_decodeD2->StartDecode() != 0) // Start decoding
		return 0;

	tagMSG m;

	bool MainCircle = true;

	while (MainCircle) // Main circle
	{
		while (PeekMessage(&m, 0, 0, 0, PM_REMOVE))
		{
			TranslateMessage(&m);
			DispatchMessage(&m);
		}

		if (m_evExit.Check()) { MainCircle = false; break; }

		RenderWindow(); // Render
	}

	m_decodeD2->StopDecode(); // Stop decoding

	DestroyRender(); // Destroy render
	DestroyWindow(); // Destroy window

	m_decodeD2 = nullptr; // destroy all object in this thread!
	m_decodeAudio = nullptr; // destroy all object in this thread!

	printf("Window was closed!\n");

	return 0;
}

