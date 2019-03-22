#pragma once

#include "BaseGPURender.h"

#include <atltrace.h>

#define __hr(hr) \
	{ \
	if (hr != S_OK) \
	{ \
		printf("HRESULT error 0x%x (%s %d)\n", \
		hr, \
		__FILE__,__LINE__); \
		ATLTRACE("HRESULT error 0x%x (%s %d)\n", \
		hr, \
		__FILE__,__LINE__); \
		} \
	}

class GPURenderDX : public BaseGPURender
{
private:
	bool m_bInitD3D11;

	DXGI_FORMAT					m_formatTexture;		// format texture
	DXGI_FORMAT					m_formatSwapChain;		// format swap chain
	
	ID3D11Device*				m_pd3dDevice;          // the pointer to the device
	ID3D11DeviceContext*		m_pd3dDeviceContext;   // the pointer to the device context
	IDXGISwapChain*				m_pSwapChain;          // the pointer to the swap chain
	ID3D11RenderTargetView*		m_pRenderTargetView;   // the pointer to the target view

	ID3D11SamplerState*			m_pSamplerState;
	ID3D11Buffer*				m_pConstantBuffer;

	ID3D11InputLayout*			m_pLayout;				// the pointer to the input layout
	ID3D11VertexShader*			m_pVS;					// the pointer to the vertex shader
	ID3D11PixelShader*			m_pPS;					// the pointer to the pixel shader
	ID3D11Buffer*				m_pVBufferRotate;       // the pointer to the vertex buffer
	ID3D11Buffer*				m_pVBufferNonRotate;    // the pointer to the vertex buffer
	
	ID3D11Texture2D*			m_pTexture;				// the pointer to the texture
	ID3D11ShaderResourceView*	m_pTexture_Srv;			// the pointer to the texture resource view

	ID3D11BlendState*			m_d3dBlendState;		// the pointer to the blend state
	
	std::vector<ID3D11Buffer*>	m_pVBufferQueueList;

public:
	GPURenderDX();
	virtual ~GPURenderDX();

private:
	virtual int RenderWindow();
	virtual int InitRender();
	virtual int DestroyRender();
	virtual int GenerateImage(bool & bRotateFrame);
	virtual int CopyBufferToTexture(C_Block *pBlock);
	virtual int CopyCUDAImage(C_Block *pBlock);

private:
	HRESULT CreateD3D11();
	HRESULT DestroyD3D11();

	void InitValues();
	void CheckChangeSwapChainSize();

	HRESULT CreateD3DXTexture(DXGI_FORMAT format, D3D11_USAGE Usage, int iWidthTex, int iHeightTex, ID3D11Texture2D** pTexture, ID3D11ShaderResourceView** pTexture_Srv);

public:
	int CreateD3DXBuffer(ID3D11Buffer** pBuffer, size_t iSizeBuffer);
};

// D3DX10math.inl
typedef struct D3DXCOLOR {
	FLOAT r;
	FLOAT g;
	FLOAT b;
	FLOAT a;

	D3DXCOLOR(FLOAT fr, FLOAT fg, FLOAT fb, FLOAT fa)
	{
		r = fr;
		g = fg;
		b = fb;
		a = fa;
	}
} D3DXCOLOR, *LPD3DXCOLOR;

struct VERTEX 
{
	FLOAT XX, YY, ZZ;	// position
	D3DXCOLOR Color;	// color
	float tu, tv;		// texcoord
};

static D3D11_SAMPLER_DESC samplerDesc = 
{
	D3D11_FILTER_MIN_MAG_MIP_LINEAR,
	D3D11_TEXTURE_ADDRESS_CLAMP,
	D3D11_TEXTURE_ADDRESS_CLAMP,
	D3D11_TEXTURE_ADDRESS_CLAMP,
	0.0f,
	1,
	D3D11_COMPARISON_ALWAYS,
	{ 0, 0, 0, 0 },
	0,
	D3D11_FLOAT32_MAX
};

struct ConstantBuffer
{
	int     UseCase;
};

