#include "stdafx.h"
#include "GPURenderDX.h"

#include "Shaders/VShader.h"
#include "Shaders/PShader.h"

GPURenderDX::GPURenderDX() :
	m_bInitD3D11(false)
{
	m_windowCaption = L"TestApp (Decode Daniel2) DX"; // Set window caption

	InitValues(); // Init values

	gpu_render_type = GPU_RENDER_D3DX11;
}

GPURenderDX::~GPURenderDX()
{
}

void GPURenderDX::InitValues()
{
	cuda_tex_result_resource = nullptr;

	m_pLayout = nullptr;
	m_pVS = nullptr;
	m_pPS = nullptr;
	m_pVBufferRotate = nullptr;
	m_pVBufferNonRotate = nullptr;
	m_pSamplerState = nullptr;
	m_pConstantBuffer = nullptr;

	m_pTexture = nullptr;
	m_pTexture_Srv = nullptr;

	m_d3dBlendState = nullptr;

	m_pRenderTargetView = nullptr;
	m_pSwapChain = nullptr;
	m_pd3dDeviceContext = nullptr;
	m_pd3dDevice = nullptr;
}

int GPURenderDX::GenerateImage(bool & bRotateFrame)
{
	HRESULT hr = S_OK;

	if (!m_decodeD2->isProcess() || m_decodeD2->isPause()) // check for pause or process
		return 1;

	C_Block *pBlock = m_decodeD2->MapFrame(); // Get poiter to picture after decoding

	if (!pBlock)
		return -1;

	unsigned char* pFrameData = pBlock->DataPtr();

	if (m_bCopyToTexture)
	{
		D3D11_MAPPED_SUBRESOURCE ms;
		hr = m_pd3dDeviceContext->Map(m_pTexture, NULL, D3D11_MAP_WRITE_DISCARD, NULL, &ms); __hr(hr)
		if (ms.pData && pFrameData)
		{
			if (ms.RowPitch == pBlock->Pitch())
				memcpy(ms.pData, pFrameData, size_tex_data);
			else
			{
				for (size_t i = 0; i < image_height; i++)
					memcpy((BYTE*)ms.pData + (i * ms.RowPitch), pFrameData + (i * pBlock->Pitch()), pBlock->Pitch());
			}
		}
		m_pd3dDeviceContext->Unmap(m_pTexture, NULL);
	}

	bRotateFrame = pBlock->GetRotate() ? !bRotateFrame : bRotateFrame; // Rotate frame

	m_bLastRotate = pBlock->GetRotate(); // Save frame rotation value

	iCurPlayFrameNumber = pBlock->iFrameNumber; // Save currect frame number

	m_decodeD2->UnmapFrame(pBlock); // Add free pointer to queue

	return 0;
}

int GPURenderDX::CopyBufferToTexture(C_Block *pBlock)
{
#ifdef USE_CUDA_SDK
	if (m_bUseGPU)
	{
		if (m_bCopyToTexture)
		{
			CopyCUDAImage(pBlock);
		}
	}
	else
#endif
	{
		if (m_bCopyToTexture)
		{
			unsigned char* pFrameData = pBlock->DataPtr();

			D3D11_MAPPED_SUBRESOURCE ms;
			HRESULT hr = m_pd3dDeviceContext->Map(m_pTexture, NULL, D3D11_MAP_WRITE_DISCARD, NULL, &ms); __hr(hr)
			if (ms.pData && pFrameData)
			{
				memcpy(ms.pData, pFrameData, size_tex_data);
			}
			m_pd3dDeviceContext->Unmap(m_pTexture, NULL);
		}
	}

	return 0;
}

int GPURenderDX::RenderWindow()
{
	C_AutoLock lock(&m_mutex);

	//////////////////////////////////////

	if (!m_bVSync && !m_bMaxFPS && m_bVSyncHand)
	{
		double timestep = 1000.0 / ValueFPS;

		double ms_elapsed = timerqFPSMode.GetElapsedTime();

		if (ms_elapsed < timestep)
		{
			return 0;
		}
		else
		{
			timerqFPSMode.StartTimer();
		}
	}

	bool bRotate = m_bRotate;

	int res = 1;

	if (!m_bPause)
	{
		if (m_bUseGPU)
			res = GenerateCUDAImage(bRotate); // Copy data from device to device(array)
		else
			res = GenerateImage(bRotate); // Copy data from host to device

		if (res < 0)
			printf("Load texture from decoder failed!\n");
	}
	else
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(100)); // for unload CPU when set pause
	}

	if (res != 0)
	{
		bRotate = m_bLastRotate ? !bRotate : bRotate; // Rotate frame
	}

	if (m_decodeAudio && m_decodeAudio->IsInitialize())
		m_decodeAudio->PlayFrame(iCurPlayFrameNumber); // play audio

	//////////////////////////////////////

	MultithreadSyncBegin();

	HRESULT hr = S_OK;

	D3D11_BUFFER_DESC bd;
	ID3D11Buffer* pVBufferQueue = nullptr;
	D3D11_SUBRESOURCE_DATA vertexData = { 0 };

	CheckChangeSwapChainSize();

	//////////////////////////////////////

	// Set the viewport
	D3D11_VIEWPORT viewport;
	ZeroMemory(&viewport, sizeof(D3D11_VIEWPORT));

	viewport.TopLeftX = 0;
	viewport.TopLeftY = 0;
	viewport.Width = (float)window_width;
	viewport.Height = (float)window_height;

	m_pd3dDeviceContext->RSSetViewports(1, &viewport);

	//////////////////////////////////////

	// Clear the back buffer to a deep blue
	float fColorClear[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
	m_pd3dDeviceContext->ClearRenderTargetView(m_pRenderTargetView, fColorClear);

	m_pd3dDeviceContext->OMSetRenderTargets(1, &m_pRenderTargetView, NULL);

	// Select which vertex buffer to display
	UINT stride = sizeof(VERTEX);
	UINT offset = 0;

	if (bRotate)
	{
		m_pd3dDeviceContext->IASetVertexBuffers(0, 1, &m_pVBufferRotate, &stride, &offset);
	}
	else
	{
		m_pd3dDeviceContext->IASetVertexBuffers(0, 1, &m_pVBufferNonRotate, &stride, &offset);
	}

	m_pd3dDeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

	IMAGE_FORMAT output_format = m_decodeD2->GetImageFormat();

	D3D11_MAPPED_SUBRESOURCE mappedResource;
	ConstantBuffer *pcb;

	hr = m_pd3dDeviceContext->Map(m_pConstantBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	pcb = (ConstantBuffer*)mappedResource.pData;
	pcb->UseCase = output_format == IMAGE_FORMAT_BGRA16BIT ? 1 : 0;
	m_pd3dDeviceContext->Unmap(m_pConstantBuffer, 0);

	// Set shader resources
	m_pd3dDeviceContext->PSSetShaderResources(0, 1, &m_pTexture_Srv);

	if (m_bShowTexture)
	{
		// Draw texture
		m_pd3dDeviceContext->Draw(4, 0);
	}

	if (m_bShowSlider) // draw slider
	{
		size_t w = window_width; // Width in pixels of the current window
		size_t h = window_height; // Height in pixels of the current window

		sizeSquare2 = (float)w / 100;
		edgeLineY = sizeSquare2 * 4;
		edgeLineX = sizeSquare2 * 2;

		float xCoord = edgeLineX; // edgeLineX + ((((float)w - (2.f * edgeLineX)) / (float)(iAllFrames - 1)) * (float)iCurPlayFrameNumber);
		float yCoord = (float)h - edgeLineY;

		if (iAllFrames > 1)
			xCoord += ((((float)w - (2.f * edgeLineX)) / (float)(iAllFrames - 1)) * (float)iCurPlayFrameNumber);

		ZeroMemory(&viewport, sizeof(D3D11_VIEWPORT));
		viewport.TopLeftX = 0;
		viewport.TopLeftY = (float)window_height - (edgeLineY * 2);
		viewport.Width = (float)window_width;
		viewport.Height = (edgeLineY * 2.f);
		m_pd3dDeviceContext->RSSetViewports(1, &viewport);

		m_pd3dDeviceContext->IASetVertexBuffers(0, 1, &m_pVBufferNonRotate, &stride, &offset);

		hr = m_pd3dDeviceContext->Map(m_pConstantBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource); __hr(hr)
		pcb = (ConstantBuffer*)mappedResource.pData;
		pcb->UseCase = 2;
		m_pd3dDeviceContext->Unmap(m_pConstantBuffer, 0);

		m_pd3dDeviceContext->Draw(4, 0);

		/////////////////////

		m_pd3dDeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_LINESTRIP);

		VERTEX OurVertices[] =
		{
			{ -1.0f, 0.0f, 0.0f, D3DXCOLOR(1.0f, 0.0f, 0.0f, 1.0f), 0.0f, 0.0f },
			{ 1.0f,  0.0f, 0.0f, D3DXCOLOR(0.0f, 1.0f, 0.0f, 1.0f), 0.0f, 1.0f },
		};

		ZeroMemory(&bd, sizeof(bd));

		bd.Usage = D3D11_USAGE_IMMUTABLE;
		bd.ByteWidth = sizeof(VERTEX) * 2;
		bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;

		vertexData.pSysMem = OurVertices;
		pVBufferQueue = nullptr;
		hr = m_pd3dDevice->CreateBuffer(&bd, &vertexData, &pVBufferQueue); __hr(hr)
		m_pVBufferQueueList.push_back(pVBufferQueue);

		stride = sizeof(VERTEX); offset = 0;
		m_pd3dDeviceContext->IASetVertexBuffers(0, 1, &pVBufferQueue, &stride, &offset);

		hr = m_pd3dDeviceContext->Map(m_pConstantBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource); __hr(hr)
		pcb = (ConstantBuffer*)mappedResource.pData;
		pcb->UseCase = 3;
		m_pd3dDeviceContext->Unmap(m_pConstantBuffer, 0);

		if ((xCoord - sizeSquare2) > (0 + edgeLineX))
		{
			viewport.TopLeftX = edgeLineX;
			viewport.Width = (xCoord - sizeSquare2) - viewport.TopLeftX;

			m_pd3dDeviceContext->RSSetViewports(1, &viewport);	

			m_pd3dDeviceContext->Draw(2, 0);
		}

		hr = m_pd3dDeviceContext->Map(m_pConstantBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource); __hr(hr)
		pcb = (ConstantBuffer*)mappedResource.pData;
		pcb->UseCase = 4;
		m_pd3dDeviceContext->Unmap(m_pConstantBuffer, 0);

		if ((xCoord + sizeSquare2) < ((float)window_width - edgeLineX))
		{
			viewport.TopLeftX = (xCoord + sizeSquare2);
			viewport.Width = ((float)window_width - edgeLineX) - viewport.TopLeftX;

			m_pd3dDeviceContext->RSSetViewports(1, &viewport);

			m_pd3dDeviceContext->Draw(2, 0);
		}

		/////////////////////

		float fSide = 2.f;
		viewport.TopLeftX = xCoord - sizeSquare2 - fSide;
		viewport.TopLeftY = yCoord - sizeSquare2 - fSide;
		viewport.Width = (sizeSquare2 * 2.0f) + fSide;
		viewport.Height = (sizeSquare2 * 2.0f) + fSide;
		m_pd3dDeviceContext->RSSetViewports(1, &viewport);

		m_pd3dDeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_LINESTRIP);

		float fValue = 0.8f;

		VERTEX Points[] =
		{
			{ -fValue, -fValue, 0.0f, D3DXCOLOR(1.0f, 0.0f, 0.0f, 1.0f), 0.0f, 0.0f },
			{ -fValue, fValue, 0.0f, D3DXCOLOR(0.0f, 1.0f, 0.0f, 1.0f), 0.0f, 1.0f },
			{ fValue, fValue, 0.0f, D3DXCOLOR(0.0f, 1.0f, 0.0f, 1.0f), 0.0f, 1.0f },
			{ fValue, -fValue, 0.0f, D3DXCOLOR(0.0f, 1.0f, 0.0f, 1.0f), 0.0f, 1.0f },
			{ -fValue, -fValue, 0.0f, D3DXCOLOR(0.0f, 1.0f, 0.0f, 1.0f), 0.0f, 1.0f }
		};

		ZeroMemory(&bd, sizeof(bd));

		bd.Usage = D3D11_USAGE_IMMUTABLE;
		bd.ByteWidth = sizeof(VERTEX) * 5;
		bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;

		vertexData.pSysMem = Points;
		pVBufferQueue = nullptr;
		hr = m_pd3dDevice->CreateBuffer(&bd, &vertexData, &pVBufferQueue); __hr(hr)
		m_pVBufferQueueList.push_back(pVBufferQueue);

		stride = sizeof(VERTEX); offset = 0;
		m_pd3dDeviceContext->IASetVertexBuffers(0, 1, &pVBufferQueue, &stride, &offset);

		hr = m_pd3dDeviceContext->Map(m_pConstantBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource); __hr(hr)
		pcb = (ConstantBuffer*)mappedResource.pData;
		pcb->UseCase = 3;
		m_pd3dDeviceContext->Unmap(m_pConstantBuffer, 0);

		m_pd3dDeviceContext->Draw(5, 0);

		/////////////////////
	}

	MultithreadSyncEnd();

	// Switch the back buffer and the front buffer
	if (m_bVSync)
	{
		hr = m_pSwapChain->Present(1, 0);
	}
	else
		hr = m_pSwapChain->Present(0, 0);

	//////////////////////////////////////
	
	for (size_t i = 0; i < m_pVBufferQueueList.size(); i++)
	{
		ID3D11Buffer* pVBufferQueue = m_pVBufferQueueList[i];

		if (pVBufferQueue)
			pVBufferQueue->Release();
	}

	m_pVBufferQueueList.clear();
	
	//////////////////////////////////////

	ComputeFPS(); // Calculate fps

	return 0;
}

int GPURenderDX::InitRender()
{
	if (CreateD3D11() != S_OK)
		return -1;

	ShowWindow(m_hWnd, SW_SHOWDEFAULT);
	
	if (!m_bInitD3D11)
		return -1;

	return 0;
}

int GPURenderDX::DestroyRender()
{
	DestroyD3D11();

	return 0;
}

HRESULT GPURenderDX::CreateD3D11()
{
	HRESULT hr = S_OK;

	RECT rc;
	GetClientRect(m_hWnd, &rc);
	UINT width = rc.right - rc.left;    // Get width of window
	UINT height = rc.bottom - rc.top;   // Get height of window

	IMAGE_FORMAT output_format = m_decodeD2->GetImageFormat();

	switch (output_format)
	{
		case IMAGE_FORMAT_RGBA8BIT:
		{
			m_formatTexture = DXGI_FORMAT_R8G8B8A8_UNORM;
			m_formatSwapChain = DXGI_FORMAT_R8G8B8A8_UNORM;
			break;
		}
		case IMAGE_FORMAT_BGRA8BIT:
		{
			m_formatTexture = DXGI_FORMAT_B8G8R8A8_UNORM;
			m_formatSwapChain = DXGI_FORMAT_B8G8R8A8_UNORM;
			break;
		}
		case IMAGE_FORMAT_RGBA16BIT:
		case IMAGE_FORMAT_BGRA16BIT: // problem! we do not have DXGI_FORMAT_B16G16R16A16_UNORM format
		{
			m_formatTexture = DXGI_FORMAT_R16G16B16A16_UNORM;
			m_formatSwapChain = DXGI_FORMAT_R10G10B10A2_UNORM;
			break;
		}
		default:
		{
			m_formatTexture = DXGI_FORMAT_R8G8B8A8_UNORM;
			m_formatSwapChain = DXGI_FORMAT_R8G8B8A8_UNORM;
		}
	}

	///////////////////////////////////

	UINT createDeviceFlags = 0;

#ifdef _DEBUG     
	createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

	D3D_FEATURE_LEVEL featureLevels[] = { 
		D3D_FEATURE_LEVEL_11_0,
		D3D_FEATURE_LEVEL_10_1,
		D3D_FEATURE_LEVEL_10_0 
	};
	UINT numFeatureLevels = ARRAYSIZE(featureLevels);

	// Set up the structure used to create the device and swapchain
	DXGI_SWAP_CHAIN_DESC sd;								
	ZeroMemory(&sd, sizeof(sd));							

	sd.BufferCount = 1;										
	sd.BufferDesc.Width = width;							
	sd.BufferDesc.Height = height;                          
	sd.BufferDesc.Format = m_formatSwapChain;				
	sd.BufferDesc.RefreshRate.Numerator = 0;				
	sd.BufferDesc.RefreshRate.Denominator = 0;
	sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;		
	sd.OutputWindow = m_hWnd;                               
	sd.SampleDesc.Count = 1;
	sd.SampleDesc.Quality = 0;
	sd.Windowed = TRUE;										
	sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

	D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_11_0;

	// Create device and swapchain
	if (m_pCapableAdapter)
	{
		hr = D3D11CreateDeviceAndSwapChain(m_pCapableAdapter, D3D_DRIVER_TYPE_UNKNOWN, NULL, createDeviceFlags, featureLevels, numFeatureLevels, D3D11_SDK_VERSION,
			&sd, &m_pSwapChain, &m_pd3dDevice, &featureLevel, &m_pd3dDeviceContext);
	}
	else
	{
		hr = D3D11CreateDeviceAndSwapChain(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, createDeviceFlags, featureLevels, numFeatureLevels, D3D11_SDK_VERSION,
			&sd, &m_pSwapChain, &m_pd3dDevice, &featureLevel, &m_pd3dDeviceContext); __hr(hr)
	}

	if (FAILED(hr))
		return hr;

	/////////////////////////

	com_ptr<IDXGIDevice> dxgiDevice = nullptr;
	com_ptr<IDXGIAdapter> adapter = nullptr;

	hr = m_pd3dDevice->QueryInterface(&dxgiDevice);
	if (SUCCEEDED(hr) && dxgiDevice) hr = dxgiDevice->GetAdapter(&adapter);

	DXGI_ADAPTER_DESC desc;
	if (SUCCEEDED(hr) && adapter) hr = adapter->GetDesc(&desc);

	if (SUCCEEDED(hr))
	{
		wprintf(L"D3DX11 Adapter: %s\n", desc.Description);
		printf("-------------------------------------\n");
	}

	/////////////////////////

	if (SUCCEEDED(hr)) hr = m_pd3dDeviceContext->QueryInterface(IID_ID3D10Multithread, (void**)&m_pMulty);
	if (SUCCEEDED(hr)) hr = m_pMulty->SetMultithreadProtected(TRUE);

	/////////////////////////

	// Create a render target view of the swapchain
	ID3D11Texture2D* pBackBuffer = NULL;
	hr = m_pSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&pBackBuffer); __hr(hr)

	if (FAILED(hr))
		return hr;

	hr = m_pd3dDevice->CreateRenderTargetView(pBackBuffer, NULL, &m_pRenderTargetView); __hr(hr)
	pBackBuffer->Release();

	if (FAILED(hr))
		return hr;

	m_pd3dDeviceContext->OMSetRenderTargets(1, &m_pRenderTargetView, NULL);

	// Set the viewport
	D3D11_VIEWPORT viewport;
	ZeroMemory(&viewport, sizeof(D3D11_VIEWPORT));

	viewport.TopLeftX = 0;
	viewport.TopLeftY = 0;
	viewport.Width = (float)width;
	viewport.Height = (float)height;

	m_pd3dDeviceContext->RSSetViewports(1, &viewport);

	/////////////////////////////

	// Vertex and Pixel shaders
	hr = m_pd3dDevice->CreateVertexShader(g_VShader, sizeof(g_VShader), NULL, &m_pVS); __hr(hr)
	
	if (FAILED(hr))
		return hr;

	hr = m_pd3dDevice->CreatePixelShader(g_PShader, sizeof(g_PShader), NULL, &m_pPS); __hr(hr)

	if (FAILED(hr))
		return hr;

	// Set the shader objects
	m_pd3dDeviceContext->VSSetShader(m_pVS, 0, 0);
	m_pd3dDeviceContext->PSSetShader(m_pPS, 0, 0);

	// Create the constant buffer
	{
		D3D11_BUFFER_DESC cbDesc;
		cbDesc.Usage = D3D11_USAGE_DYNAMIC;
		cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
		cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		cbDesc.MiscFlags = 0;
		cbDesc.ByteWidth = 16 * sizeof(ConstantBuffer);
		if (SUCCEEDED(hr)) hr = m_pd3dDevice->CreateBuffer(&cbDesc, NULL, &m_pConstantBuffer); __hr(hr)

		// Assign the buffer now : nothing in the code will interfere with this (very simple sample)
		if (m_pConstantBuffer)
		{
			m_pd3dDeviceContext->VSSetConstantBuffers(0, 1, &m_pConstantBuffer);
			m_pd3dDeviceContext->PSSetConstantBuffers(0, 1, &m_pConstantBuffer);
		}
	}

	// create the input layout object
	D3D11_INPUT_ELEMENT_DESC ied[] =
	{
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12 + 16, D3D11_INPUT_PER_VERTEX_DATA, 0 }
	};

	// Create the input layout object
	if (SUCCEEDED(hr)) hr = m_pd3dDevice->CreateInputLayout(ied, 3, g_VShader, sizeof(g_VShader), &m_pLayout); __hr(hr)
	
	if (FAILED(hr))
		return hr;

	// Setup Input Layout
	m_pd3dDeviceContext->IASetInputLayout(m_pLayout);

	m_pd3dDeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

	// SamplerState
	hr = m_pd3dDevice->CreateSamplerState(&samplerDesc, &m_pSamplerState); __hr(hr)

	if (FAILED(hr))
		return hr;

	m_pd3dDeviceContext->PSSetSamplers(0, 1, &m_pSamplerState);

	/////////////////////////////

	VERTEX OurVerticesNonRotate[] =
	{
		{ -1.0f, -1.0f, 0.0f, D3DXCOLOR(1.0f, 0.0f, 0.0f, 1.0f), 0.0f, 0.0f },
		{ -1.0f,  1.0f, 0.0f, D3DXCOLOR(0.0f, 1.0f, 0.0f, 1.0f), 0.0f, 1.0f },
		{ 1.0f, -1.0f, 0.0f,  D3DXCOLOR(0.0f, 0.0f, 1.0f, 1.0f), 1.0f, 0.0f },
		{ 1.0f,  1.0f, 0.0f,  D3DXCOLOR(1.0f, 0.0f, 1.0f, 1.0f), 1.0f, 1.0f }
	};

	VERTEX OurVerticesRotate[] =
	{
		{ -1.0f, -1.0f, 0.0f, D3DXCOLOR(1.0f, 0.0f, 0.0f, 1.0f), 0.0f, 1.0f },
		{ -1.0f,  1.0f, 0.0f, D3DXCOLOR(0.0f, 1.0f, 0.0f, 1.0f), 0.0f, 0.0f },
		{ 1.0f, -1.0f, 0.0f,  D3DXCOLOR(0.0f, 0.0f, 1.0f, 1.0f), 1.0f, 1.0f },
		{ 1.0f,  1.0f, 0.0f,  D3DXCOLOR(1.0f, 0.0f, 1.0f, 1.0f), 1.0f, 0.0f }
	};

	D3D11_BUFFER_DESC bd;
	ZeroMemory(&bd, sizeof(bd));

	bd.Usage = D3D11_USAGE_IMMUTABLE;
	bd.ByteWidth = sizeof(VERTEX) * 4;
	bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;

	D3D11_SUBRESOURCE_DATA vertexData = { 0 };

	vertexData.pSysMem = OurVerticesRotate;
	hr = m_pd3dDevice->CreateBuffer(&bd, &vertexData, &m_pVBufferRotate); __hr(hr)

	if (FAILED(hr))
		return hr;

	vertexData.pSysMem = OurVerticesNonRotate;
	hr = m_pd3dDevice->CreateBuffer(&bd, &vertexData, &m_pVBufferNonRotate); __hr(hr)

	if (FAILED(hr))
		return hr;

	/////////////////////////////
	
	D3D11_USAGE Usage = m_bUseGPU ? D3D11_USAGE_DEFAULT : D3D11_USAGE_DYNAMIC;
	hr = CreateD3DXTexture(m_formatTexture, Usage, image_width, image_height, &m_pTexture, &m_pTexture_Srv);

	if (FAILED(hr))
		return hr;

	bytePerPixel = 4 * sizeof(unsigned char);
	size_tex_data = image_width * image_height * bytePerPixel;

	if (m_formatTexture == DXGI_FORMAT_R16G16B16A16_UNORM)
	{
		bytePerPixel = 4 * sizeof(unsigned short);
		size_tex_data = image_width * image_height * bytePerPixel;
	}

	/////////////////////////////

	if (m_bUseGPU)
	{
		// Register this texture with CUDA
		cudaGraphicsD3D11RegisterResource(&cuda_tex_result_resource, m_pTexture, cudaGraphicsRegisterFlagsSurfaceLoadStore); __vrcu
	}

	/////////////////////////////

	// Alpha Blend State

	D3D11_BLEND_DESC omDesc;
	ZeroMemory(&omDesc, sizeof(D3D11_BLEND_DESC));
	omDesc.RenderTarget[0].BlendEnable = true;

	omDesc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
	omDesc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
	omDesc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
	omDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
	omDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
	omDesc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
	omDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;

	hr = m_pd3dDevice->CreateBlendState(&omDesc, &m_d3dBlendState); __hr(hr)

	m_pd3dDeviceContext->OMSetBlendState(m_d3dBlendState, 0, 0xffffffff);

	/////////////////////////////

	m_bInitD3D11 = true;
	m_bInitRender = true;

	return S_OK;
}

HRESULT GPURenderDX::DestroyD3D11()
{
	m_bInitD3D11 = false;
	m_bInitRender = false;

	if (m_pSwapChain)
		m_pSwapChain->SetFullscreenState(FALSE, NULL);    // switch to windowed mode
	
	////////////////////
	
	if (m_bUseGPU)
	{
		// Unregister resource with CUDA
		if (cuda_tex_result_resource)
			cudaGraphicsUnregisterResource(cuda_tex_result_resource); __vrcu
	}

	////////////////////

	if (m_pLayout)
		m_pLayout->Release();

	if (m_pVS)
		m_pVS->Release();

	if (m_pPS)
		m_pPS->Release();

	if (m_pVBufferRotate)
		m_pVBufferRotate->Release();

	if (m_pVBufferNonRotate)
		m_pVBufferNonRotate->Release();

	if (m_pSamplerState)
		m_pSamplerState->Release();

	if (m_pConstantBuffer)
		m_pConstantBuffer->Release();

	if (m_pTexture)
		m_pTexture->Release();
	
	if (m_pTexture_Srv)
		m_pTexture_Srv->Release();

	if (m_d3dBlendState)
		m_d3dBlendState->Release();

	////////////////////

	if (m_pRenderTargetView)
		m_pRenderTargetView->Release();

	if (m_pSwapChain)
		m_pSwapChain->Release();

	if (m_pd3dDeviceContext)
		m_pd3dDeviceContext->Release();

	if (m_pd3dDevice)
		m_pd3dDevice->Release();
	
	InitValues();

	return S_OK;
}

void GPURenderDX::CheckChangeSwapChainSize()
{
	RECT rc;
	GetClientRect(m_hWnd, &rc);
	UINT width = rc.right - rc.left;    // Get width of window
	UINT height = rc.bottom - rc.top;   // Get height of window

	DXGI_SWAP_CHAIN_DESC swapChainDesc = { 0 };
	m_pSwapChain->GetDesc(&swapChainDesc);

	if (width == 0 || height == 0)
	{
		width = swapChainDesc.BufferDesc.Width;
		height = swapChainDesc.BufferDesc.Height;
	}

	if (swapChainDesc.BufferDesc.Width != width || swapChainDesc.BufferDesc.Height != height)
	{
		HRESULT hr = S_OK;

		m_pRenderTargetView->Release();
		m_pRenderTargetView = NULL;

		hr = m_pSwapChain->ResizeBuffers(0, width, height, m_formatSwapChain, 0); __hr(hr)

		if (SUCCEEDED(hr))
		{
			ID3D11Texture2D* pBackBuffer = NULL;
			hr = m_pSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&pBackBuffer); __hr(hr)

			if (SUCCEEDED(hr))
			{
				hr = m_pd3dDevice->CreateRenderTargetView(pBackBuffer, NULL, &m_pRenderTargetView); __hr(hr)
				pBackBuffer->Release();
			}

			if (SUCCEEDED(hr)) m_pd3dDeviceContext->OMSetRenderTargets(1, &m_pRenderTargetView, NULL); __hr(hr)
		}
	}

	// Set size
	window_width = width;
	window_height = height;
}

HRESULT GPURenderDX::CreateD3DXTexture(DXGI_FORMAT format, D3D11_USAGE Usage, int iWidthTex, int iHeightTex, ID3D11Texture2D** pTexture, ID3D11ShaderResourceView** pTexture_Srv)
{
	HRESULT hr = S_OK;

	D3D11_TEXTURE2D_DESC texDesc = { 0 };

	texDesc.Width = iWidthTex;
	texDesc.Height = iHeightTex;
	texDesc.MipLevels = 1;
	texDesc.ArraySize = 1;
	texDesc.Format = format;
	texDesc.SampleDesc.Count = 1;
	texDesc.SampleDesc.Quality = 0;

	texDesc.Usage = Usage;
	texDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	texDesc.CPUAccessFlags = 0;

	if (texDesc.Usage == D3D11_USAGE_DEFAULT)
		texDesc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;

	if (texDesc.Usage == D3D11_USAGE_DYNAMIC)
		texDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

	hr = m_pd3dDevice->CreateTexture2D(&texDesc, NULL, pTexture); __hr(hr)
	if (SUCCEEDED(hr)) hr = m_pd3dDevice->CreateShaderResourceView(*pTexture, NULL, pTexture_Srv); __hr(hr)

	if (!SUCCEEDED(hr))
	{
		printf("CreateD3DXTexture failed: width = %d weight = %d\n", iWidthTex, iHeightTex);
		return hr;
	}

	return hr;
}

HRESULT GPURenderDX::CreateD3DXBuffer(ID3D11Buffer** pBuffer, size_t iSizeBuffer)
{
	HRESULT hr = S_OK;

	D3D11_BUFFER_DESC desc = {};

	desc.Usage = D3D11_USAGE_DEFAULT;
	desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;

	desc.ByteWidth = static_cast<UINT>(iSizeBuffer);

	hr = m_pd3dDevice->CreateBuffer(&desc, NULL, pBuffer); __hr(hr)

	if (!SUCCEEDED(hr))
	{
		printf("CreateD3DXBuffer failed: size of buffer = %zu\n", iSizeBuffer);
		return hr;
	}

	return hr;
}

int GPURenderDX::CopyCUDAImage(C_Block *pBlock)
{
	if (!m_bCopyToTexture)
		return 0;

	if (m_decodeD2->IsD3DX11Acc())
	{
		ID3D11Resource* pResourceDXD11 = pBlock->GetD3DX11ResourcePtr();
		D3D11_RESOURCE_DIMENSION resDim;
		pResourceDXD11->GetType(&resDim);

		MultithreadSyncBegin();

		if (resDim == D3D11_RESOURCE_DIMENSION_BUFFER)
		{
			// We want to copy image data to the texture
			// map buffer objects to get CUDA device pointers
			cudaError_t err;
			cudaArray *texture_ptr = nullptr;
			err = cudaGraphicsMapResources(1, &cuda_tex_result_resource, 0); __vrcu
			err = cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_result_resource, 0, 0); __vrcu

			// Register the resources of buffer
			err = cudaGraphicsD3D11RegisterResource(&cuda_tex_result_resource_buff, pBlock->GetD3DX11ResourcePtr(), cudaGraphicsRegisterFlagsNone); __vrcu

			void *buffer_ptr = nullptr;
			size_t buffer_size = 0;
			// Map the resources of buffer
			err = cudaGraphicsMapResources(1, &cuda_tex_result_resource_buff, 0); __vrcu
			// Get pointer of buffer
			err = cudaGraphicsResourceGetMappedPointer(&buffer_ptr, &buffer_size, cuda_tex_result_resource_buff); __vrcu

			ConvertMatrixCoeff iMatrixCoeff_YUYtoRGBA = (ConvertMatrixCoeff)(pBlock->iMatrixCoeff_YUYtoRGBA);

			IMAGE_FORMAT output_format = m_decodeD2->GetImageFormat();
			BUFFER_FORMAT buffer_format = m_decodeD2->GetBufferFormat();

			if (buffer_format == BUFFER_FORMAT_RGBA32 || buffer_format == BUFFER_FORMAT_RGBA64)
			{
				cudaMemcpy2DToArray(texture_ptr, 0, 0, buffer_ptr, pBlock->Pitch(), (pBlock->Width() * bytePerPixel), pBlock->Height(), cudaMemcpyDeviceToDevice); __vrcu
			}
			else if (buffer_format == BUFFER_FORMAT_YUY2)
			{
				if (output_format == IMAGE_FORMAT_RGBA8BIT)
				{
					h_convert_YUY2_to_RGBA32_BtT(buffer_ptr, texture_ptr, (int)pBlock->Width(), (int)pBlock->Height(), (int)pBlock->Pitch(), NULL, iMatrixCoeff_YUYtoRGBA); __vrcu
				}
				else if (output_format == IMAGE_FORMAT_BGRA8BIT)
				{
					h_convert_YUY2_to_BGRA32_BtT(buffer_ptr, texture_ptr, (int)pBlock->Width(), (int)pBlock->Height(), (int)pBlock->Pitch(), NULL, iMatrixCoeff_YUYtoRGBA); __vrcu
				}
			}
			else if (buffer_format == BUFFER_FORMAT_Y216)
			{
				if (output_format == IMAGE_FORMAT_RGBA16BIT)
				{
					h_convert_Y216_to_RGBA64_BtT(buffer_ptr, texture_ptr, (int)pBlock->Width(), (int)pBlock->Height(), (int)pBlock->Pitch(), NULL, iMatrixCoeff_YUYtoRGBA); __vrcu
				}
				else if (output_format == IMAGE_FORMAT_BGRA16BIT)
				{
					h_convert_Y216_to_BGRA64_BtT(buffer_ptr, texture_ptr, (int)pBlock->Width(), (int)pBlock->Height(), (int)pBlock->Pitch(), NULL, iMatrixCoeff_YUYtoRGBA); __vrcu
				}
				else if (output_format == IMAGE_FORMAT_RGBA8BIT)
				{
					h_convert_Y216_to_RGBA32_BtT(buffer_ptr, texture_ptr, (int)pBlock->Width(), (int)pBlock->Height(), (int)pBlock->Pitch(), NULL, iMatrixCoeff_YUYtoRGBA); __vrcu
				}
				else if (output_format == IMAGE_FORMAT_BGRA8BIT)
				{
					h_convert_Y216_to_BGRA32_BtT(buffer_ptr, texture_ptr, (int)pBlock->Width(), (int)pBlock->Height(), (int)pBlock->Pitch(), NULL, iMatrixCoeff_YUYtoRGBA); __vrcu
				}
			}
			else if (buffer_format == BUFFER_FORMAT_NV12)
			{
				if (output_format == IMAGE_FORMAT_RGBA8BIT)
				{
					h_convert_NV12_to_RGBA32_BtT(buffer_ptr, texture_ptr, (int)pBlock->Width(), (int)pBlock->Height(), (int)pBlock->Pitch(), NULL, iMatrixCoeff_YUYtoRGBA); __vrcu
				}
				else if (output_format == IMAGE_FORMAT_BGRA8BIT)
				{
					h_convert_NV12_to_BGRA32_BtT(buffer_ptr, texture_ptr, (int)pBlock->Width(), (int)pBlock->Height(), (int)pBlock->Pitch(), NULL, iMatrixCoeff_YUYtoRGBA); __vrcu
				}
			}
			else if (buffer_format == BUFFER_FORMAT_P016)
			{
				if (output_format == IMAGE_FORMAT_RGBA16BIT)
				{
					h_convert_P016_to_RGBA64_BtT(buffer_ptr, texture_ptr, (int)pBlock->Width(), (int)pBlock->Height(), (int)pBlock->Pitch(), NULL, iMatrixCoeff_YUYtoRGBA); __vrcu
				}
				else if (output_format == IMAGE_FORMAT_BGRA16BIT)
				{
					h_convert_P016_to_BGRA64_BtT(buffer_ptr, texture_ptr, (int)pBlock->Width(), (int)pBlock->Height(), (int)pBlock->Pitch(), NULL, iMatrixCoeff_YUYtoRGBA); __vrcu
				}
			}

			// Unmap the resources of texture
			err = cudaGraphicsUnmapResources(1, &cuda_tex_result_resource, 0); __vrcu

			// Unmap the resources of buffer
			err = cudaGraphicsUnmapResources(1, &cuda_tex_result_resource_buff, 0); __vrcu
			// Unregister the resources of buffer
			err = cudaGraphicsUnregisterResource(cuda_tex_result_resource_buff); __vrcu
		}
		else if (resDim == D3D11_RESOURCE_DIMENSION_TEXTURE2D)
		{
			// We want to copy image data to the texture
			// map buffer objects to get CUDA device pointers
			cudaError_t err;
			cudaArray *texture_ptr = nullptr;
			err = cudaGraphicsMapResources(1, &cuda_tex_result_resource, 0); __vrcu
			err = cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_result_resource, 0, 0); __vrcu

			// Register the resources of buffer
			err = cudaGraphicsD3D11RegisterResource(&cuda_tex_result_resource_buff, pBlock->GetD3DX11ResourcePtr(), cudaGraphicsRegisterFlagsSurfaceLoadStore); __vrcu

			cudaArray *buffer_ptr = nullptr;
			size_t buffer_size = 0;
			// Map the resources of buffer
			err = cudaGraphicsMapResources(1, &cuda_tex_result_resource_buff, 0); __vrcu
			// Get pointer of buffer
			err = cudaGraphicsSubResourceGetMappedArray(&buffer_ptr, cuda_tex_result_resource_buff, 0, 0); __vrcu

			ConvertMatrixCoeff iMatrixCoeff_YUYtoRGBA = (ConvertMatrixCoeff)(pBlock->iMatrixCoeff_YUYtoRGBA);

			IMAGE_FORMAT output_format = m_decodeD2->GetImageFormat();
			BUFFER_FORMAT buffer_format = m_decodeD2->GetBufferFormat();

			if (buffer_format == BUFFER_FORMAT_RGBA32 || buffer_format == BUFFER_FORMAT_RGBA64)
			{
				cudaMemcpyArrayToArray(texture_ptr, 0, 0, buffer_ptr, 0, 0, pBlock->Size(), cudaMemcpyDeviceToDevice); __vrcu
			}
			else if (buffer_format == BUFFER_FORMAT_YUY2)
			{
				if (output_format == IMAGE_FORMAT_RGBA8BIT)
				{
					h_convert_YUY2_to_RGBA32_TtT(buffer_ptr, texture_ptr, (int)pBlock->Width(), (int)pBlock->Height(), NULL, iMatrixCoeff_YUYtoRGBA); __vrcu
				}
				else if (output_format == IMAGE_FORMAT_BGRA8BIT)
				{
					h_convert_YUY2_to_BGRA32_TtT(buffer_ptr, texture_ptr, (int)pBlock->Width(), (int)pBlock->Height(), NULL, iMatrixCoeff_YUYtoRGBA); __vrcu
				}
			}
			else if (buffer_format == BUFFER_FORMAT_Y216)
			{
				if (output_format == IMAGE_FORMAT_RGBA16BIT)
				{
					h_convert_Y216_to_RGBA64_TtT(buffer_ptr, texture_ptr, (int)pBlock->Width(), (int)pBlock->Height(), NULL, iMatrixCoeff_YUYtoRGBA); __vrcu
				}
				else if (output_format == IMAGE_FORMAT_BGRA16BIT)
				{
					h_convert_Y216_to_BGRA64_TtT(buffer_ptr, texture_ptr, (int)pBlock->Width(), (int)pBlock->Height(), NULL, iMatrixCoeff_YUYtoRGBA); __vrcu
				}
				else if (output_format == IMAGE_FORMAT_RGBA8BIT)
				{
					h_convert_Y216_to_RGBA32_TtT(buffer_ptr, texture_ptr, (int)pBlock->Width(), (int)pBlock->Height(), NULL, iMatrixCoeff_YUYtoRGBA); __vrcu
				}
				else if (output_format == IMAGE_FORMAT_BGRA8BIT)
				{
					h_convert_Y216_to_BGRA32_TtT(buffer_ptr, texture_ptr, (int)pBlock->Width(), (int)pBlock->Height(), NULL, iMatrixCoeff_YUYtoRGBA); __vrcu
				}
			}
			else if (buffer_format == BUFFER_FORMAT_NV12)
			{
				if (output_format == IMAGE_FORMAT_RGBA8BIT)
				{
					h_convert_NV12_to_RGBA32_TtT(buffer_ptr, texture_ptr, (int)pBlock->Width(), (int)pBlock->Height(), NULL, iMatrixCoeff_YUYtoRGBA); __vrcu
				}
				else if (output_format == IMAGE_FORMAT_BGRA8BIT)
				{
					h_convert_NV12_to_BGRA32_TtT(buffer_ptr, texture_ptr, (int)pBlock->Width(), (int)pBlock->Height(), NULL, iMatrixCoeff_YUYtoRGBA); __vrcu
				}
			}
			else if (buffer_format == BUFFER_FORMAT_P016)
			{
				if (output_format == IMAGE_FORMAT_RGBA16BIT)
				{
					h_convert_P016_to_RGBA64_TtT(buffer_ptr, texture_ptr, (int)pBlock->Width(), (int)pBlock->Height(), NULL, iMatrixCoeff_YUYtoRGBA); __vrcu
				}
				else if (output_format == IMAGE_FORMAT_BGRA16BIT)
				{
					h_convert_P016_to_BGRA64_TtT(buffer_ptr, texture_ptr, (int)pBlock->Width(), (int)pBlock->Height(), NULL, iMatrixCoeff_YUYtoRGBA); __vrcu
				}
			}

			// Unmap the resources of texture
			err = cudaGraphicsUnmapResources(1, &cuda_tex_result_resource, 0); __vrcu

			// Unmap the resources of buffer
			err = cudaGraphicsUnmapResources(1, &cuda_tex_result_resource_buff, 0); __vrcu
			// Unregister the resources of buffer
			err = cudaGraphicsUnregisterResource(cuda_tex_result_resource_buff); __vrcu
		}

		MultithreadSyncEnd();
	}
	else
	{
		BaseGPURender::CopyCUDAImage(pBlock);
	}

	return 0;
}