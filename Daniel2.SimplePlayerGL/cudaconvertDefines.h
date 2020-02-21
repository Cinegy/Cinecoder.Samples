#pragma once

enum ConvertMatrixCoeff
{// Kr      Kb
	ConvertMatrixCoeff_Default = 0,		//	{ 0.30,   0.11 },	// no sequence display extension
	ConvertMatrixCoeff_ITU_R709,		//	{ 0.2125, 0.0721 }, /* ITU-R Rec. 709 (1990) */
	ConvertMatrixCoeff_Unspecified,		//	{ 0.299,  0.114 },	/* unspecified */
	ConvertMatrixCoeff_Reserved,		//	{ 0.299,  0.114 },	/* reserved */
	ConvertMatrixCoeff_FCC,				//	{ 0.30,   0.11 },	/* FCC */
	ConvertMatrixCoeff_ITU_R6244,		//	{ 0.299,  0.114 },	/* ITU-R Rec. 624-4 System B, G */
	ConvertMatrixCoeff_SMPTE_170M,		//	{ 0.299,  0.114 },	/* SMPTE 170M */
	ConvertMatrixCoeff_SMPTE_240M,		//	{ 0.212,  0.087 },	/* SMPTE 240M (1987) */
	ConvertMatrixCoeff_YCgCo,			//	{ 0.30,   0.11 },	// YCgCo
	ConvertMatrixCoeff_ITU_R_BT2020		//	{ 0.2627, 0.0593 }, /* Rec. ITU-R BT.2020 */
};

typedef cudaError_t(*FTh_convert_YUY2_to_RGBA32_BtB)(void* pYUY2, void* pRGBA32, int Width, int Height, int PitchYUY2, int PitchRGBA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_YUY2_to_RGBA32_BtT)(void* pYUY2, cudaArray* pTextureRGBA32, int Width, int Height, int PitchYUY2, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_YUY2_to_RGBA32_TtB)(cudaArray* pTextureYUY2, void* pRGBA32, int Width, int Height, int PitchRGBA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_YUY2_to_RGBA32_TtT)(cudaArray* pTextureYUY2, cudaArray* pTextureRGBA32, int Width, int Height, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);

typedef cudaError_t(*FTh_convert_YUY2_to_BGRA32_BtB)(void* pYUY2, void* pBGRA32, int Width, int Height, int PitchYUY2, int PitchBGRA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_YUY2_to_BGRA32_BtT)(void* pYUY2, cudaArray* pTextureBGRA32, int Width, int Height, int PitchYUY2, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_YUY2_to_BGRA32_TtB)(cudaArray* pTextureYUY2, void* pBGRA32, int Width, int Height, int PitchBGRA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_YUY2_to_BGRA32_TtT)(cudaArray* pTextureYUY2, cudaArray* pTextureBGRA32, int Width, int Height, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);

typedef cudaError_t(*FTh_convert_RGBA32_to_YUY2_BtB)(void* pRGBA32, void* pYUY2, int Width, int Height, int PitchRGBA, int PitchYUY2, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_RGBA32_to_YUY2_BtT)(void* pRGBA32, cudaArray* pTextureYUY2, int Width, int Height, int PitchRGBA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_RGBA32_to_YUY2_TtB)(cudaArray* pTextureRGBA32, void* pYUY2, int Width, int Height, int PitchYUY2, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_RGBA32_to_YUY2_TtT)(cudaArray* pTextureRGBA32, cudaArray* pTextureYUY2, int Width, int Height, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);

typedef cudaError_t(*FTh_convert_YUY2_to_RGBA64_BtB)(void* pYUY2, void* pRGBA64, int Width, int Height, int PitchYUY2, int PitchRGBA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_YUY2_to_RGBA64_BtT)(void* pYUY2, cudaArray* pTextureRGBA64, int Width, int Height, int PitchYUY2, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_YUY2_to_RGBA64_TtB)(cudaArray* pTextureYUY2, void* pRGBA64, int Width, int Height, int PitchRGBA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_YUY2_to_RGBA64_TtT)(cudaArray* pTextureYUY2, cudaArray* pTextureRGBA64, int Width, int Height, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);

typedef cudaError_t(*FTh_convert_YUY2_to_BGRA64_BtB)(void* pYUY2, void* pBGRA64, int Width, int Height, int PitchYUY2, int PitchBGRA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_YUY2_to_BGRA64_BtT)(void* pYUY2, cudaArray* pTextureBGRA64, int Width, int Height, int PitchYUY2, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_YUY2_to_BGRA64_TtB)(cudaArray* pTextureYUY2, void* pBGRA64, int Width, int Height, int PitchBGRA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_YUY2_to_BGRA64_TtT)(cudaArray* pTextureYUY2, cudaArray* pTextureBGRA64, int Width, int Height, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);

typedef cudaError_t(*FTh_convert_RGBA64_to_YUY2_BtB)(void* pRGBA64, void* pYUY2, int Width, int Height, int PitchRGBA, int PitchYUY2, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_RGBA64_to_YUY2_BtT)(void* pRGBA64, cudaArray* pTextureYUY2, int Width, int Height, int PitchRGBA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_RGBA64_to_YUY2_TtB)(cudaArray* pTextureRGBA64, void* pYUY2, int Width, int Height, int PitchYUY2, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_RGBA64_to_YUY2_TtT)(cudaArray* pTextureRGBA64, cudaArray* pTextureYUY2, int Width, int Height, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);

typedef cudaError_t(*FTh_convert_Y216_to_RGBA32_BtB)(void* pY216, void* pRGBA32, int Width, int Height, int PitchY216, int PitchRGBA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_Y216_to_RGBA32_BtT)(void* pY216, cudaArray* pTextureRGBA32, int Width, int Height, int PitchY216, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_Y216_to_RGBA32_TtB)(cudaArray* pTextureY216, void* pRGBA32, int Width, int Height, int PitchRGBA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_Y216_to_RGBA32_TtT)(cudaArray* pTextureY216, cudaArray* pTextureRGBA32, int Width, int Height, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);

typedef cudaError_t(*FTh_convert_Y216_to_BGRA32_BtB)(void* pY216, void* pBGRA32, int Width, int Height, int PitchY216, int PitchBGRA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_Y216_to_BGRA32_BtT)(void* pY216, cudaArray* pTextureBGRA32, int Width, int Height, int PitchY216, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_Y216_to_BGRA32_TtB)(cudaArray* pTextureY216, void* pBGRA32, int Width, int Height, int PitchBGRA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_Y216_to_BGRA32_TtT)(cudaArray* pTextureY216, cudaArray* pTextureBGRA32, int Width, int Height, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);

typedef cudaError_t(*FTh_convert_RGBA32_to_Y216_BtB)(void* pRGBA32, void* pY216, int Width, int Height, int PitchRGBA, int PitchY216, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_RGBA32_to_Y216_BtT)(void* pRGBA32, cudaArray* pTextureY216, int Width, int Height, int PitchRGBA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_RGBA32_to_Y216_TtB)(cudaArray* pTextureRGBA32, void* pY216, int Width, int Height, int PitchY216, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_RGBA32_to_Y216_TtT)(cudaArray* pTextureRGBA32, cudaArray* pTextureY216, int Width, int Height, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);

typedef cudaError_t(*FTh_convert_Y216_to_RGBA64_BtB)(void* pY216, void* pRGBA64, int Width, int Height, int PitchY216, int PitchRGBA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_Y216_to_RGBA64_BtT)(void* pY216, cudaArray* pTextureRGBA64, int Width, int Height, int PitchY216, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_Y216_to_RGBA64_TtB)(cudaArray* pTextureY216, void* pRGBA64, int Width, int Height, int PitchRGBA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_Y216_to_RGBA64_TtT)(cudaArray* pTextureY216, cudaArray* pTextureRGBA64, int Width, int Height, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);

typedef cudaError_t(*FTh_convert_Y216_to_BGRA64_BtB)(void* pY216, void* pBGRA64, int Width, int Height, int PitchY216, int PitchBGRA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_Y216_to_BGRA64_BtT)(void* pY216, cudaArray* pTextureBGRA64, int Width, int Height, int PitchY216, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_Y216_to_BGRA64_TtB)(cudaArray* pTextureY216, void* pBGRA64, int Width, int Height, int PitchBGRA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_Y216_to_BGRA64_TtT)(cudaArray* pTextureY216, cudaArray* pTextureBGRA64, int Width, int Height, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);

typedef cudaError_t(*FTh_convert_RGBA64_to_Y216_BtB)(void* pRGBA64, void* pY216, int Width, int Height, int PitchRGBA, int PitchY216, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_RGBA64_to_Y216_BtT)(void* pRGBA64, cudaArray* pTextureY216, int Width, int Height, int PitchRGBA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_RGBA64_to_Y216_TtB)(cudaArray* pTextureRGBA64, void* pY216, int Width, int Height, int PitchY216, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_RGBA64_to_Y216_TtT)(cudaArray* pTextureRGBA64, cudaArray* pTextureY216, int Width, int Height, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);

typedef cudaError_t(*FTh_convert_Y210_to_RGBA32_BtB)(void* pY210, void* pRGBA32, int Width, int Height, int PitchY210, int PitchRGBA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_Y210_to_RGBA32_BtT)(void* pY210, cudaArray* pTextureRGBA32, int Width, int Height, int PitchY210, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_Y210_to_RGBA32_TtB)(cudaArray* pTextureY210, void* pRGBA32, int Width, int Height, int PitchRGBA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_Y210_to_RGBA32_TtT)(cudaArray* pTextureY210, cudaArray* pTextureRGBA32, int Width, int Height, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);

typedef cudaError_t(*FTh_convert_Y210_to_BGRA32_BtB)(void* pY210, void* pBGRA32, int Width, int Height, int PitchY210, int PitchBGRA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_Y210_to_BGRA32_BtT)(void* pY210, cudaArray* pTextureBGRA32, int Width, int Height, int PitchY210, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_Y210_to_BGRA32_TtB)(cudaArray* pTextureY210, void* pBGRA32, int Width, int Height, int PitchBGRA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_Y210_to_BGRA32_TtT)(cudaArray* pTextureY210, cudaArray* pTextureBGRA32, int Width, int Height, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);

typedef cudaError_t(*FTh_convert_RGBA32_to_Y210_BtB)(void* pRGBA32, void* pY210, int Width, int Height, int PitchRGBA, int PitchY210, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_RGBA32_to_Y210_BtT)(void* pRGBA32, cudaArray* pTextureY210, int Width, int Height, int PitchRGBA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_RGBA32_to_Y210_TtB)(cudaArray* pTextureRGBA32, void* pY210, int Width, int Height, int PitchY210, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_RGBA32_to_Y210_TtT)(cudaArray* pTextureRGBA32, cudaArray* pTextureY210, int Width, int Height, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);

typedef cudaError_t(*FTh_convert_Y210_to_RGBA64_BtB)(void* pY210, void* pRGBA64, int Width, int Height, int PitchY210, int PitchRGBA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_Y210_to_RGBA64_BtT)(void* pY210, cudaArray* pTextureRGBA64, int Width, int Height, int PitchY210, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_Y210_to_RGBA64_TtB)(cudaArray* pTextureY210, void* pRGBA64, int Width, int Height, int PitchRGBA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_Y210_to_RGBA64_TtT)(cudaArray* pTextureY210, cudaArray* pTextureRGBA64, int Width, int Height, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);

typedef cudaError_t(*FTh_convert_Y210_to_BGRA64_BtB)(void* pY210, void* pBGRA64, int Width, int Height, int PitchY210, int PitchBGRA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_Y210_to_BGRA64_BtT)(void* pY210, cudaArray* pTextureBGRA64, int Width, int Height, int PitchY210, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_Y210_to_BGRA64_TtB)(cudaArray* pTextureY210, void* pBGRA64, int Width, int Height, int PitchBGRA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_Y210_to_BGRA64_TtT)(cudaArray* pTextureY210, cudaArray* pTextureBGRA64, int Width, int Height, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);

typedef cudaError_t(*FTh_convert_RGBA64_to_Y210_BtB)(void* pRGBA64, void* pY210, int Width, int Height, int PitchRGBA, int PitchY210, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_RGBA64_to_Y210_BtT)(void* pRGBA64, cudaArray* pTextureY210, int Width, int Height, int PitchRGBA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_RGBA64_to_Y210_TtB)(cudaArray* pTextureRGBA64, void* pY210, int Width, int Height, int PitchY210, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_RGBA64_to_Y210_TtT)(cudaArray* pTextureRGBA64, cudaArray* pTextureY210, int Width, int Height, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);

typedef cudaError_t(*FTh_convert_V210_to_RGBA32_BtB)(void* pV210, void* pRGBA32, int Width, int Height, int PitchV210, int PitchRGBA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_V210_to_RGBA32_BtT)(void* pV210, cudaArray* pTextureRGBA32, int Width, int Height, int PitchV210, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_V210_to_RGBA32_TtB)(cudaArray* pTextureV210, void* pRGBA32, int Width, int Height, int PitchV210, int PitchRGBA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_V210_to_RGBA32_TtT)(cudaArray* pV210, cudaArray* pTextureRGBA32, int Width, int Height, int PitchV210, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);

typedef cudaError_t(*FTh_convert_V210_to_BGRA32_BtB)(void* pV210, void* pBGRA32, int Width, int Height, int PitchV210, int PitchBGRA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_V210_to_BGRA32_BtT)(void* pV210, cudaArray* pTextureBGRA32, int Width, int Height, int PitchV210, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_V210_to_BGRA32_TtB)(cudaArray* pTextureV210, void* pBGRA32, int Width, int Height, int PitchV210, int PitchBGRA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_V210_to_BGRA32_TtT)(cudaArray* pV210, cudaArray* pTextureBGRA32, int Width, int Height, int PitchV210, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);

typedef cudaError_t(*FTh_convert_V210_to_RGBA64_BtB)(void* pV210, void* pRGBA64, int Width, int Height, int PitchV210, int PitchRGBA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_V210_to_RGBA64_BtT)(void* pV210, cudaArray* pTextureRGBA64, int Width, int Height, int PitchV210, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_V210_to_RGBA64_TtB)(cudaArray* pTextureV210, void* pRGBA64, int Width, int Height, int PitchV210, int PitchRGBA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_V210_to_RGBA64_TtT)(cudaArray* pV210, cudaArray* pTextureRGBA64, int Width, int Height, int PitchV210, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);

typedef cudaError_t(*FTh_convert_V210_to_BGRA64_BtB)(void* pV210, void* pBGRA64, int Width, int Height, int PitchV210, int PitchBGRA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_V210_to_BGRA64_BtT)(void* pV210, cudaArray* pTextureBGRA64, int Width, int Height, int PitchV210, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_V210_to_BGRA64_TtB)(cudaArray* pTextureV210, void* pBGRA64, int Width, int Height, int PitchV210, int PitchBGRA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_V210_to_BGRA64_TtT)(cudaArray* pV210, cudaArray* pTextureBGRA64, int Width, int Height, int PitchV210, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);

typedef cudaError_t(*FTh_convert_NV12_to_RGBA32_BtB)(void* pNV12, void* pRGBA32, int Width, int Height, int PitchNV12, int PitchRGBA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_NV12_to_RGBA32_BtT)(void* pNV12, cudaArray* pTextureRGBA32, int Width, int Height, int PitchNV12, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_NV12_to_RGBA32_TtB)(cudaArray* pTextureNV12, void* pRGBA32, int Width, int Height, int PitchRGBA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_NV12_to_RGBA32_TtT)(cudaArray* pTextureNV12, cudaArray* pTextureRGBA32, int Width, int Height, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);

typedef cudaError_t(*FTh_convert_NV12_to_BGRA32_BtB)(void* pNV12, void* pBGRA32, int Width, int Height, int PitchNV12, int PitchBGRA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_NV12_to_BGRA32_BtT)(void* pNV12, cudaArray* pTextureBGRA32, int Width, int Height, int PitchNV12, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_NV12_to_BGRA32_TtB)(cudaArray* pTextureNV12, void* pBGRA32, int Width, int Height, int PitchBGRA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_NV12_to_BGRA32_TtT)(cudaArray* pTextureNV12, cudaArray* pTextureBGRA32, int Width, int Height, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);

typedef cudaError_t(*FTh_convert_P016_to_RGBA64_BtB)(void* pP016, void* pRGBA64, int Width, int Height, int PitchP016, int PitchRGBA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_P016_to_RGBA64_BtT)(void* pP016, cudaArray* pTextureRGBA64, int Width, int Height, int PitchP016, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_P016_to_RGBA64_TtB)(cudaArray* pTextureP016, void* pRGBA64, int Width, int Height, int PitchRGBA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_P016_to_RGBA64_TtT)(cudaArray* pTextureP016, cudaArray* pTextureRGBA64, int Width, int Height, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);

typedef cudaError_t(*FTh_convert_P016_to_BGRA64_BtB)(void* pP016, void* pBGRA64, int Width, int Height, int PitchP016, int PitchBGRA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_P016_to_BGRA64_BtT)(void* pP016, cudaArray* pTextureBGRA64, int Width, int Height, int PitchP016, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_P016_to_BGRA64_TtB)(cudaArray* pTextureP016, void* pBGRA64, int Width, int Height, int PitchBGRA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_P016_to_BGRA64_TtT)(cudaArray* pTextureP016, cudaArray* pTextureBGRA64, int Width, int Height, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);

typedef cudaError_t(*FTh_convert_P016_to_RGBA32_BtB)(void* pP016, void* pRGBA32, int Width, int Height, int PitchP016, int PitchRGBA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_P016_to_RGBA32_BtT)(void* pP016, cudaArray* pTextureRGBA32, int Width, int Height, int PitchP016, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_P016_to_RGBA32_TtB)(cudaArray* pTextureP016, void* pRGBA32, int Width, int Height, int PitchRGBA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_P016_to_RGBA32_TtT)(cudaArray* pTextureP016, cudaArray* pTextureRGBA32, int Width, int Height, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);

typedef cudaError_t(*FTh_convert_P016_to_BGRA32_BtB)(void* pP016, void* pBGRA32, int Width, int Height, int PitchP016, int PitchBGRA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_P016_to_BGRA32_BtT)(void* pP016, cudaArray* pTextureBGRA32, int Width, int Height, int PitchP016, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_P016_to_BGRA32_TtB)(cudaArray* pTextureP016, void* pBGRA32, int Width, int Height, int PitchBGRA, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);
typedef cudaError_t(*FTh_convert_P016_to_BGRA32_TtT)(cudaArray* pTextureP016, cudaArray* pTextureBGRA32, int Width, int Height, cudaStream_t Stream, ConvertMatrixCoeff iMatrixCoeff);

typedef cudaError_t(*FTh_convert_RGBA32_to_RGBA64_BtB)(void* pRGBA32, void* pRGBA64, int Width, int Height, int PitchRGBA32, int PitchRGBA64, cudaStream_t Stream);
typedef cudaError_t(*FTh_convert_RGBA32_to_RGBA64_BtT)(void* pRGBA32, cudaArray* pTextureRGBA64, int Width, int Height, int PitchRGBA32, cudaStream_t Stream);
typedef cudaError_t(*FTh_convert_RGBA32_to_RGBA64_TtB)(cudaArray* pTextureRGBA32, void* pRGBA64, int Width, int Height, int PitchRGBA64, cudaStream_t Stream);
typedef cudaError_t(*FTh_convert_RGBA32_to_RGBA64_TtT)(cudaArray* pTextureRGBA32, cudaArray* pTextureRGBA64, int Width, int Height, cudaStream_t Stream);

typedef cudaError_t(*FTh_convert_RGBA64_to_RGBA32_BtB)(void* pRGBA64, void* pRGBA32, int Width, int Height, int PitchRGBA64, int PitchRGBA32, cudaStream_t Stream);
typedef cudaError_t(*FTh_convert_RGBA64_to_RGBA32_BtT)(void* pRGBA64, cudaArray* pTextureRGBA32, int Width, int Height, int PitchRGBA64, cudaStream_t Stream);
typedef cudaError_t(*FTh_convert_RGBA64_to_RGBA32_TtB)(cudaArray* pTextureRGBA64, void* pRGBA32, int Width, int Height, int PitchRGBA32, cudaStream_t Stream);
typedef cudaError_t(*FTh_convert_RGBA64_to_RGBA32_TtT)(cudaArray* pTextureRGBA64, cudaArray* pTextureRGBA32, int Width, int Height, cudaStream_t Stream);

extern FTh_convert_YUY2_to_RGBA32_BtB FUNC_CUDA(h_convert_YUY2_to_RGBA32_BtB);
extern FTh_convert_YUY2_to_RGBA32_BtT FUNC_CUDA(h_convert_YUY2_to_RGBA32_BtT);
extern FTh_convert_YUY2_to_RGBA32_TtB FUNC_CUDA(h_convert_YUY2_to_RGBA32_TtB);
extern FTh_convert_YUY2_to_RGBA32_TtT FUNC_CUDA(h_convert_YUY2_to_RGBA32_TtT);

extern FTh_convert_YUY2_to_BGRA32_BtB FUNC_CUDA(h_convert_YUY2_to_BGRA32_BtB);
extern FTh_convert_YUY2_to_BGRA32_BtT FUNC_CUDA(h_convert_YUY2_to_BGRA32_BtT);
extern FTh_convert_YUY2_to_BGRA32_TtB FUNC_CUDA(h_convert_YUY2_to_BGRA32_TtB);
extern FTh_convert_YUY2_to_BGRA32_TtT FUNC_CUDA(h_convert_YUY2_to_BGRA32_TtT);

extern FTh_convert_RGBA32_to_YUY2_BtB FUNC_CUDA(h_convert_RGBA32_to_YUY2_BtB);
extern FTh_convert_RGBA32_to_YUY2_BtT FUNC_CUDA(h_convert_RGBA32_to_YUY2_BtT);
extern FTh_convert_RGBA32_to_YUY2_TtB FUNC_CUDA(h_convert_RGBA32_to_YUY2_TtB);
extern FTh_convert_RGBA32_to_YUY2_TtT FUNC_CUDA(h_convert_RGBA32_to_YUY2_TtT);

extern FTh_convert_YUY2_to_RGBA64_BtB FUNC_CUDA(h_convert_YUY2_to_RGBA64_BtB);
extern FTh_convert_YUY2_to_RGBA64_BtT FUNC_CUDA(h_convert_YUY2_to_RGBA64_BtT);
extern FTh_convert_YUY2_to_RGBA64_TtB FUNC_CUDA(h_convert_YUY2_to_RGBA64_TtB);
extern FTh_convert_YUY2_to_RGBA64_TtT FUNC_CUDA(h_convert_YUY2_to_RGBA64_TtT);

extern FTh_convert_YUY2_to_BGRA64_BtB FUNC_CUDA(h_convert_YUY2_to_BGRA64_BtB);
extern FTh_convert_YUY2_to_BGRA64_BtT FUNC_CUDA(h_convert_YUY2_to_BGRA64_BtT);
extern FTh_convert_YUY2_to_BGRA64_TtB FUNC_CUDA(h_convert_YUY2_to_BGRA64_TtB);
extern FTh_convert_YUY2_to_BGRA64_TtT FUNC_CUDA(h_convert_YUY2_to_BGRA64_TtT);

extern FTh_convert_RGBA64_to_YUY2_BtB FUNC_CUDA(h_convert_RGBA64_to_YUY2_BtB);
extern FTh_convert_RGBA64_to_YUY2_BtT FUNC_CUDA(h_convert_RGBA64_to_YUY2_BtT);
extern FTh_convert_RGBA64_to_YUY2_TtB FUNC_CUDA(h_convert_RGBA64_to_YUY2_TtB);
extern FTh_convert_RGBA64_to_YUY2_TtT FUNC_CUDA(h_convert_RGBA64_to_YUY2_TtT);

extern FTh_convert_Y216_to_RGBA32_BtB FUNC_CUDA(h_convert_Y216_to_RGBA32_BtB);
extern FTh_convert_Y216_to_RGBA32_BtT FUNC_CUDA(h_convert_Y216_to_RGBA32_BtT);
extern FTh_convert_Y216_to_RGBA32_TtB FUNC_CUDA(h_convert_Y216_to_RGBA32_TtB);
extern FTh_convert_Y216_to_RGBA32_TtT FUNC_CUDA(h_convert_Y216_to_RGBA32_TtT);

extern FTh_convert_Y216_to_BGRA32_BtB FUNC_CUDA(h_convert_Y216_to_BGRA32_BtB);
extern FTh_convert_Y216_to_BGRA32_BtT FUNC_CUDA(h_convert_Y216_to_BGRA32_BtT);
extern FTh_convert_Y216_to_BGRA32_TtB FUNC_CUDA(h_convert_Y216_to_BGRA32_TtB);
extern FTh_convert_Y216_to_BGRA32_TtT FUNC_CUDA(h_convert_Y216_to_BGRA32_TtT);

extern FTh_convert_RGBA32_to_Y216_BtB FUNC_CUDA(h_convert_RGBA32_to_Y216_BtB);
extern FTh_convert_RGBA32_to_Y216_BtT FUNC_CUDA(h_convert_RGBA32_to_Y216_BtT);
extern FTh_convert_RGBA32_to_Y216_TtB FUNC_CUDA(h_convert_RGBA32_to_Y216_TtB);
extern FTh_convert_RGBA32_to_Y216_TtT FUNC_CUDA(h_convert_RGBA32_to_Y216_TtT);

extern FTh_convert_Y216_to_RGBA64_BtB FUNC_CUDA(h_convert_Y216_to_RGBA64_BtB);
extern FTh_convert_Y216_to_RGBA64_BtT FUNC_CUDA(h_convert_Y216_to_RGBA64_BtT);
extern FTh_convert_Y216_to_RGBA64_TtB FUNC_CUDA(h_convert_Y216_to_RGBA64_TtB);
extern FTh_convert_Y216_to_RGBA64_TtT FUNC_CUDA(h_convert_Y216_to_RGBA64_TtT);

extern FTh_convert_Y216_to_BGRA64_BtB FUNC_CUDA(h_convert_Y216_to_BGRA64_BtB);
extern FTh_convert_Y216_to_BGRA64_BtT FUNC_CUDA(h_convert_Y216_to_BGRA64_BtT);
extern FTh_convert_Y216_to_BGRA64_TtB FUNC_CUDA(h_convert_Y216_to_BGRA64_TtB);
extern FTh_convert_Y216_to_BGRA64_TtT FUNC_CUDA(h_convert_Y216_to_BGRA64_TtT);

extern FTh_convert_RGBA64_to_Y216_BtB FUNC_CUDA(h_convert_RGBA64_to_Y216_BtB);
extern FTh_convert_RGBA64_to_Y216_BtT FUNC_CUDA(h_convert_RGBA64_to_Y216_BtT);
extern FTh_convert_RGBA64_to_Y216_TtB FUNC_CUDA(h_convert_RGBA64_to_Y216_TtB);
extern FTh_convert_RGBA64_to_Y216_TtT FUNC_CUDA(h_convert_RGBA64_to_Y216_TtT);

extern FTh_convert_Y210_to_RGBA32_BtB FUNC_CUDA(h_convert_Y210_to_RGBA32_BtB);
extern FTh_convert_Y210_to_RGBA32_BtT FUNC_CUDA(h_convert_Y210_to_RGBA32_BtT);
extern FTh_convert_Y210_to_RGBA32_TtB FUNC_CUDA(h_convert_Y210_to_RGBA32_TtB);
extern FTh_convert_Y210_to_RGBA32_TtT FUNC_CUDA(h_convert_Y210_to_RGBA32_TtT);

extern FTh_convert_Y210_to_BGRA32_BtB FUNC_CUDA(h_convert_Y210_to_BGRA32_BtB);
extern FTh_convert_Y210_to_BGRA32_BtT FUNC_CUDA(h_convert_Y210_to_BGRA32_BtT);
extern FTh_convert_Y210_to_BGRA32_TtB FUNC_CUDA(h_convert_Y210_to_BGRA32_TtB);
extern FTh_convert_Y210_to_BGRA32_TtT FUNC_CUDA(h_convert_Y210_to_BGRA32_TtT);

extern FTh_convert_RGBA32_to_Y210_BtB FUNC_CUDA(h_convert_RGBA32_to_Y210_BtB);
extern FTh_convert_RGBA32_to_Y210_BtT FUNC_CUDA(h_convert_RGBA32_to_Y210_BtT);
extern FTh_convert_RGBA32_to_Y210_TtB FUNC_CUDA(h_convert_RGBA32_to_Y210_TtB);
extern FTh_convert_RGBA32_to_Y210_TtT FUNC_CUDA(h_convert_RGBA32_to_Y210_TtT);

extern FTh_convert_Y210_to_RGBA64_BtB FUNC_CUDA(h_convert_Y210_to_RGBA64_BtB);
extern FTh_convert_Y210_to_RGBA64_BtT FUNC_CUDA(h_convert_Y210_to_RGBA64_BtT);
extern FTh_convert_Y210_to_RGBA64_TtB FUNC_CUDA(h_convert_Y210_to_RGBA64_TtB);
extern FTh_convert_Y210_to_RGBA64_TtT FUNC_CUDA(h_convert_Y210_to_RGBA64_TtT);

extern FTh_convert_Y210_to_BGRA64_BtB FUNC_CUDA(h_convert_Y210_to_BGRA64_BtB);
extern FTh_convert_Y210_to_BGRA64_BtT FUNC_CUDA(h_convert_Y210_to_BGRA64_BtT);
extern FTh_convert_Y210_to_BGRA64_TtB FUNC_CUDA(h_convert_Y210_to_BGRA64_TtB);
extern FTh_convert_Y210_to_BGRA64_TtT FUNC_CUDA(h_convert_Y210_to_BGRA64_TtT);

extern FTh_convert_RGBA64_to_Y210_BtB FUNC_CUDA(h_convert_RGBA64_to_Y210_BtB);
extern FTh_convert_RGBA64_to_Y210_BtT FUNC_CUDA(h_convert_RGBA64_to_Y210_BtT);
extern FTh_convert_RGBA64_to_Y210_TtB FUNC_CUDA(h_convert_RGBA64_to_Y210_TtB);
extern FTh_convert_RGBA64_to_Y210_TtT FUNC_CUDA(h_convert_RGBA64_to_Y210_TtT);

extern FTh_convert_V210_to_RGBA32_BtB FUNC_CUDA(h_convert_V210_to_RGBA32_BtB);
extern FTh_convert_V210_to_RGBA32_BtT FUNC_CUDA(h_convert_V210_to_RGBA32_BtT);
extern FTh_convert_V210_to_RGBA32_TtB FUNC_CUDA(h_convert_V210_to_RGBA32_TtB);
extern FTh_convert_V210_to_RGBA32_TtT FUNC_CUDA(h_convert_V210_to_RGBA32_TtT);

extern FTh_convert_V210_to_BGRA32_BtB FUNC_CUDA(h_convert_V210_to_BGRA32_BtB);
extern FTh_convert_V210_to_BGRA32_BtT FUNC_CUDA(h_convert_V210_to_BGRA32_BtT);
extern FTh_convert_V210_to_BGRA32_TtB FUNC_CUDA(h_convert_V210_to_BGRA32_TtB);
extern FTh_convert_V210_to_BGRA32_TtT FUNC_CUDA(h_convert_V210_to_BGRA32_TtT);

extern FTh_convert_V210_to_RGBA64_BtB FUNC_CUDA(h_convert_V210_to_RGBA64_BtB);
extern FTh_convert_V210_to_RGBA64_BtT FUNC_CUDA(h_convert_V210_to_RGBA64_BtT);
extern FTh_convert_V210_to_RGBA64_TtB FUNC_CUDA(h_convert_V210_to_RGBA64_TtB);
extern FTh_convert_V210_to_RGBA64_TtT FUNC_CUDA(h_convert_V210_to_RGBA64_TtT);

extern FTh_convert_V210_to_BGRA64_BtB FUNC_CUDA(h_convert_V210_to_BGRA64_BtB);
extern FTh_convert_V210_to_BGRA64_BtT FUNC_CUDA(h_convert_V210_to_BGRA64_BtT);
extern FTh_convert_V210_to_BGRA64_TtB FUNC_CUDA(h_convert_V210_to_BGRA64_TtB);
extern FTh_convert_V210_to_BGRA64_TtT FUNC_CUDA(h_convert_V210_to_BGRA64_TtT);

extern FTh_convert_NV12_to_RGBA32_BtB FUNC_CUDA(h_convert_NV12_to_RGBA32_BtB);
extern FTh_convert_NV12_to_RGBA32_BtT FUNC_CUDA(h_convert_NV12_to_RGBA32_BtT);
extern FTh_convert_NV12_to_RGBA32_TtB FUNC_CUDA(h_convert_NV12_to_RGBA32_TtB);
extern FTh_convert_NV12_to_RGBA32_TtT FUNC_CUDA(h_convert_NV12_to_RGBA32_TtT);

extern FTh_convert_NV12_to_BGRA32_BtB FUNC_CUDA(h_convert_NV12_to_BGRA32_BtB);
extern FTh_convert_NV12_to_BGRA32_BtT FUNC_CUDA(h_convert_NV12_to_BGRA32_BtT);
extern FTh_convert_NV12_to_BGRA32_TtB FUNC_CUDA(h_convert_NV12_to_BGRA32_TtB);
extern FTh_convert_NV12_to_BGRA32_TtT FUNC_CUDA(h_convert_NV12_to_BGRA32_TtT);

extern FTh_convert_P016_to_RGBA64_BtB FUNC_CUDA(h_convert_P016_to_RGBA64_BtB);
extern FTh_convert_P016_to_RGBA64_BtT FUNC_CUDA(h_convert_P016_to_RGBA64_BtT);
extern FTh_convert_P016_to_RGBA64_TtB FUNC_CUDA(h_convert_P016_to_RGBA64_TtB);
extern FTh_convert_P016_to_RGBA64_TtT FUNC_CUDA(h_convert_P016_to_RGBA64_TtT);

extern FTh_convert_P016_to_BGRA64_BtB FUNC_CUDA(h_convert_P016_to_BGRA64_BtB);
extern FTh_convert_P016_to_BGRA64_BtT FUNC_CUDA(h_convert_P016_to_BGRA64_BtT);
extern FTh_convert_P016_to_BGRA64_TtB FUNC_CUDA(h_convert_P016_to_BGRA64_TtB);
extern FTh_convert_P016_to_BGRA64_TtT FUNC_CUDA(h_convert_P016_to_BGRA64_TtT);

extern FTh_convert_P016_to_RGBA32_BtB FUNC_CUDA(h_convert_P016_to_RGBA32_BtB);
extern FTh_convert_P016_to_RGBA32_BtT FUNC_CUDA(h_convert_P016_to_RGBA32_BtT);
extern FTh_convert_P016_to_RGBA32_TtB FUNC_CUDA(h_convert_P016_to_RGBA32_TtB);
extern FTh_convert_P016_to_RGBA32_TtT FUNC_CUDA(h_convert_P016_to_RGBA32_TtT);

extern FTh_convert_P016_to_BGRA32_BtB FUNC_CUDA(h_convert_P016_to_BGRA32_BtB);
extern FTh_convert_P016_to_BGRA32_BtT FUNC_CUDA(h_convert_P016_to_BGRA32_BtT);
extern FTh_convert_P016_to_BGRA32_TtB FUNC_CUDA(h_convert_P016_to_BGRA32_TtB);
extern FTh_convert_P016_to_BGRA32_TtT FUNC_CUDA(h_convert_P016_to_BGRA32_TtT);

extern FTh_convert_RGBA32_to_RGBA64_BtB FUNC_CUDA(h_convert_RGBA32_to_RGBA64_BtB);
extern FTh_convert_RGBA32_to_RGBA64_BtT FUNC_CUDA(h_convert_RGBA32_to_RGBA64_BtT);
extern FTh_convert_RGBA32_to_RGBA64_TtB FUNC_CUDA(h_convert_RGBA32_to_RGBA64_TtB);
extern FTh_convert_RGBA32_to_RGBA64_TtT FUNC_CUDA(h_convert_RGBA32_to_RGBA64_TtT);

extern FTh_convert_RGBA64_to_RGBA32_BtB FUNC_CUDA(h_convert_RGBA64_to_RGBA32_BtB);
extern FTh_convert_RGBA64_to_RGBA32_BtT FUNC_CUDA(h_convert_RGBA64_to_RGBA32_BtT);
extern FTh_convert_RGBA64_to_RGBA32_TtB FUNC_CUDA(h_convert_RGBA64_to_RGBA32_TtB);
extern FTh_convert_RGBA64_to_RGBA32_TtT FUNC_CUDA(h_convert_RGBA64_to_RGBA32_TtT);

static HMODULE hCudaConvertLib = nullptr;

static int InitCudaConvertLib()
{
	hCudaConvertLib = LoadLibraryA(CUDACONVERTLIBRARY_FILENAME);

	if (hCudaConvertLib)
	{
		FUNC_CUDA(h_convert_YUY2_to_RGBA32_BtB) = (FTh_convert_YUY2_to_RGBA32_BtB)GetProcAddress(hCudaConvertLib, "h_convert_YUY2_to_RGBA32_BtB"); CHECK_FUNC_CUDA(h_convert_YUY2_to_RGBA32_BtB)
		FUNC_CUDA(h_convert_YUY2_to_RGBA32_BtT) = (FTh_convert_YUY2_to_RGBA32_BtT)GetProcAddress(hCudaConvertLib, "h_convert_YUY2_to_RGBA32_BtT"); CHECK_FUNC_CUDA(h_convert_YUY2_to_RGBA32_BtT)
		FUNC_CUDA(h_convert_YUY2_to_RGBA32_TtB) = (FTh_convert_YUY2_to_RGBA32_TtB)GetProcAddress(hCudaConvertLib, "h_convert_YUY2_to_RGBA32_TtB"); CHECK_FUNC_CUDA(h_convert_YUY2_to_RGBA32_TtB)
		FUNC_CUDA(h_convert_YUY2_to_RGBA32_TtT) = (FTh_convert_YUY2_to_RGBA32_TtT)GetProcAddress(hCudaConvertLib, "h_convert_YUY2_to_RGBA32_TtT"); CHECK_FUNC_CUDA(h_convert_YUY2_to_RGBA32_TtT)

		FUNC_CUDA(h_convert_YUY2_to_BGRA32_BtB) = (FTh_convert_YUY2_to_BGRA32_BtB)GetProcAddress(hCudaConvertLib, "h_convert_YUY2_to_BGRA32_BtB"); CHECK_FUNC_CUDA(h_convert_YUY2_to_BGRA32_BtB)
		FUNC_CUDA(h_convert_YUY2_to_BGRA32_BtT) = (FTh_convert_YUY2_to_BGRA32_BtT)GetProcAddress(hCudaConvertLib, "h_convert_YUY2_to_BGRA32_BtT"); CHECK_FUNC_CUDA(h_convert_YUY2_to_BGRA32_BtT)
		FUNC_CUDA(h_convert_YUY2_to_BGRA32_TtB) = (FTh_convert_YUY2_to_BGRA32_TtB)GetProcAddress(hCudaConvertLib, "h_convert_YUY2_to_BGRA32_TtB"); CHECK_FUNC_CUDA(h_convert_YUY2_to_BGRA32_TtB)
		FUNC_CUDA(h_convert_YUY2_to_BGRA32_TtT) = (FTh_convert_YUY2_to_BGRA32_TtT)GetProcAddress(hCudaConvertLib, "h_convert_YUY2_to_BGRA32_TtT"); CHECK_FUNC_CUDA(h_convert_YUY2_to_BGRA32_TtT)

		FUNC_CUDA(h_convert_RGBA32_to_YUY2_BtB) = (FTh_convert_RGBA32_to_YUY2_BtB)GetProcAddress(hCudaConvertLib, "h_convert_RGBA32_to_YUY2_BtB"); CHECK_FUNC_CUDA(h_convert_RGBA32_to_YUY2_BtB)
		FUNC_CUDA(h_convert_RGBA32_to_YUY2_BtT) = (FTh_convert_RGBA32_to_YUY2_BtT)GetProcAddress(hCudaConvertLib, "h_convert_RGBA32_to_YUY2_BtT"); CHECK_FUNC_CUDA(h_convert_RGBA32_to_YUY2_BtT)
		FUNC_CUDA(h_convert_RGBA32_to_YUY2_TtB) = (FTh_convert_RGBA32_to_YUY2_TtB)GetProcAddress(hCudaConvertLib, "h_convert_RGBA32_to_YUY2_TtB"); CHECK_FUNC_CUDA(h_convert_RGBA32_to_YUY2_TtB)
		FUNC_CUDA(h_convert_RGBA32_to_YUY2_TtT) = (FTh_convert_RGBA32_to_YUY2_TtT)GetProcAddress(hCudaConvertLib, "h_convert_RGBA32_to_YUY2_TtT"); CHECK_FUNC_CUDA(h_convert_RGBA32_to_YUY2_TtT)

		FUNC_CUDA(h_convert_YUY2_to_RGBA64_BtB) = (FTh_convert_YUY2_to_RGBA64_BtB)GetProcAddress(hCudaConvertLib, "h_convert_YUY2_to_RGBA64_BtB"); CHECK_FUNC_CUDA(h_convert_YUY2_to_RGBA64_BtB)
		FUNC_CUDA(h_convert_YUY2_to_RGBA64_BtT) = (FTh_convert_YUY2_to_RGBA64_BtT)GetProcAddress(hCudaConvertLib, "h_convert_YUY2_to_RGBA64_BtT"); CHECK_FUNC_CUDA(h_convert_YUY2_to_RGBA64_BtT)
		FUNC_CUDA(h_convert_YUY2_to_RGBA64_TtB) = (FTh_convert_YUY2_to_RGBA64_TtB)GetProcAddress(hCudaConvertLib, "h_convert_YUY2_to_RGBA64_TtB"); CHECK_FUNC_CUDA(h_convert_YUY2_to_RGBA64_TtB)
		FUNC_CUDA(h_convert_YUY2_to_RGBA64_TtT) = (FTh_convert_YUY2_to_RGBA64_TtT)GetProcAddress(hCudaConvertLib, "h_convert_YUY2_to_RGBA64_TtT"); CHECK_FUNC_CUDA(h_convert_YUY2_to_RGBA64_TtT)

		FUNC_CUDA(h_convert_YUY2_to_BGRA64_BtB) = (FTh_convert_YUY2_to_BGRA64_BtB)GetProcAddress(hCudaConvertLib, "h_convert_YUY2_to_BGRA64_BtB"); CHECK_FUNC_CUDA(h_convert_YUY2_to_BGRA64_BtB)
		FUNC_CUDA(h_convert_YUY2_to_BGRA64_BtT) = (FTh_convert_YUY2_to_BGRA64_BtT)GetProcAddress(hCudaConvertLib, "h_convert_YUY2_to_BGRA64_BtT"); CHECK_FUNC_CUDA(h_convert_YUY2_to_BGRA64_BtT)
		FUNC_CUDA(h_convert_YUY2_to_BGRA64_TtB) = (FTh_convert_YUY2_to_BGRA64_TtB)GetProcAddress(hCudaConvertLib, "h_convert_YUY2_to_BGRA64_TtB"); CHECK_FUNC_CUDA(h_convert_YUY2_to_BGRA64_TtB)
		FUNC_CUDA(h_convert_YUY2_to_BGRA64_TtT) = (FTh_convert_YUY2_to_BGRA64_TtT)GetProcAddress(hCudaConvertLib, "h_convert_YUY2_to_BGRA64_TtT"); CHECK_FUNC_CUDA(h_convert_YUY2_to_BGRA64_TtT)

		FUNC_CUDA(h_convert_RGBA64_to_YUY2_BtB) = (FTh_convert_RGBA64_to_YUY2_BtB)GetProcAddress(hCudaConvertLib, "h_convert_RGBA64_to_YUY2_BtB"); CHECK_FUNC_CUDA(h_convert_RGBA64_to_YUY2_BtB)
		FUNC_CUDA(h_convert_RGBA64_to_YUY2_BtT) = (FTh_convert_RGBA64_to_YUY2_BtT)GetProcAddress(hCudaConvertLib, "h_convert_RGBA64_to_YUY2_BtT"); CHECK_FUNC_CUDA(h_convert_RGBA64_to_YUY2_BtT)
		FUNC_CUDA(h_convert_RGBA64_to_YUY2_TtB) = (FTh_convert_RGBA64_to_YUY2_TtB)GetProcAddress(hCudaConvertLib, "h_convert_RGBA64_to_YUY2_TtB"); CHECK_FUNC_CUDA(h_convert_RGBA64_to_YUY2_TtB)
		FUNC_CUDA(h_convert_RGBA64_to_YUY2_TtT) = (FTh_convert_RGBA64_to_YUY2_TtT)GetProcAddress(hCudaConvertLib, "h_convert_RGBA64_to_YUY2_TtT"); CHECK_FUNC_CUDA(h_convert_RGBA64_to_YUY2_TtT)

		FUNC_CUDA(h_convert_Y216_to_RGBA32_BtB) = (FTh_convert_Y216_to_RGBA32_BtB)GetProcAddress(hCudaConvertLib, "h_convert_Y216_to_RGBA32_BtB"); CHECK_FUNC_CUDA(h_convert_Y216_to_RGBA32_BtB)
		FUNC_CUDA(h_convert_Y216_to_RGBA32_BtT) = (FTh_convert_Y216_to_RGBA32_BtT)GetProcAddress(hCudaConvertLib, "h_convert_Y216_to_RGBA32_BtT"); CHECK_FUNC_CUDA(h_convert_Y216_to_RGBA32_BtT)
		FUNC_CUDA(h_convert_Y216_to_RGBA32_TtB) = (FTh_convert_Y216_to_RGBA32_TtB)GetProcAddress(hCudaConvertLib, "h_convert_Y216_to_RGBA32_TtB"); CHECK_FUNC_CUDA(h_convert_Y216_to_RGBA32_TtB)
		FUNC_CUDA(h_convert_Y216_to_RGBA32_TtT) = (FTh_convert_Y216_to_RGBA32_TtT)GetProcAddress(hCudaConvertLib, "h_convert_Y216_to_RGBA32_TtT"); CHECK_FUNC_CUDA(h_convert_Y216_to_RGBA32_TtT)

		FUNC_CUDA(h_convert_Y216_to_BGRA32_BtB) = (FTh_convert_Y216_to_BGRA32_BtB)GetProcAddress(hCudaConvertLib, "h_convert_Y216_to_BGRA32_BtB"); CHECK_FUNC_CUDA(h_convert_Y216_to_BGRA32_BtB)
		FUNC_CUDA(h_convert_Y216_to_BGRA32_BtT) = (FTh_convert_Y216_to_BGRA32_BtT)GetProcAddress(hCudaConvertLib, "h_convert_Y216_to_BGRA32_BtT"); CHECK_FUNC_CUDA(h_convert_Y216_to_BGRA32_BtT)
		FUNC_CUDA(h_convert_Y216_to_BGRA32_TtB) = (FTh_convert_Y216_to_BGRA32_TtB)GetProcAddress(hCudaConvertLib, "h_convert_Y216_to_BGRA32_TtB"); CHECK_FUNC_CUDA(h_convert_Y216_to_BGRA32_TtB)
		FUNC_CUDA(h_convert_Y216_to_BGRA32_TtT) = (FTh_convert_Y216_to_BGRA32_TtT)GetProcAddress(hCudaConvertLib, "h_convert_Y216_to_BGRA32_TtT"); CHECK_FUNC_CUDA(h_convert_Y216_to_BGRA32_TtT)

		FUNC_CUDA(h_convert_RGBA32_to_Y216_BtB) = (FTh_convert_RGBA32_to_Y216_BtB)GetProcAddress(hCudaConvertLib, "h_convert_RGBA32_to_Y216_BtB"); CHECK_FUNC_CUDA(h_convert_RGBA32_to_Y216_BtB)
		FUNC_CUDA(h_convert_RGBA32_to_Y216_BtT) = (FTh_convert_RGBA32_to_Y216_BtT)GetProcAddress(hCudaConvertLib, "h_convert_RGBA32_to_Y216_BtT"); CHECK_FUNC_CUDA(h_convert_RGBA32_to_Y216_BtT)
		FUNC_CUDA(h_convert_RGBA32_to_Y216_TtB) = (FTh_convert_RGBA32_to_Y216_TtB)GetProcAddress(hCudaConvertLib, "h_convert_RGBA32_to_Y216_TtB"); CHECK_FUNC_CUDA(h_convert_RGBA32_to_Y216_TtB)
		FUNC_CUDA(h_convert_RGBA32_to_Y216_TtT) = (FTh_convert_RGBA32_to_Y216_TtT)GetProcAddress(hCudaConvertLib, "h_convert_RGBA32_to_Y216_TtT"); CHECK_FUNC_CUDA(h_convert_RGBA32_to_Y216_TtT)

		FUNC_CUDA(h_convert_Y216_to_RGBA64_BtB) = (FTh_convert_Y216_to_RGBA64_BtB)GetProcAddress(hCudaConvertLib, "h_convert_Y216_to_RGBA64_BtB"); CHECK_FUNC_CUDA(h_convert_Y216_to_RGBA64_BtB)
		FUNC_CUDA(h_convert_Y216_to_RGBA64_BtT) = (FTh_convert_Y216_to_RGBA64_BtT)GetProcAddress(hCudaConvertLib, "h_convert_Y216_to_RGBA64_BtT"); CHECK_FUNC_CUDA(h_convert_Y216_to_RGBA64_BtT)
		FUNC_CUDA(h_convert_Y216_to_RGBA64_TtB) = (FTh_convert_Y216_to_RGBA64_TtB)GetProcAddress(hCudaConvertLib, "h_convert_Y216_to_RGBA64_TtB"); CHECK_FUNC_CUDA(h_convert_Y216_to_RGBA64_TtB)
		FUNC_CUDA(h_convert_Y216_to_RGBA64_TtT) = (FTh_convert_Y216_to_RGBA64_TtT)GetProcAddress(hCudaConvertLib, "h_convert_Y216_to_RGBA64_TtT"); CHECK_FUNC_CUDA(h_convert_Y216_to_RGBA64_TtT)

		FUNC_CUDA(h_convert_Y216_to_BGRA64_BtB) = (FTh_convert_Y216_to_BGRA64_BtB)GetProcAddress(hCudaConvertLib, "h_convert_Y216_to_BGRA64_BtB"); CHECK_FUNC_CUDA(h_convert_Y216_to_BGRA64_BtB)
		FUNC_CUDA(h_convert_Y216_to_BGRA64_BtT) = (FTh_convert_Y216_to_BGRA64_BtT)GetProcAddress(hCudaConvertLib, "h_convert_Y216_to_BGRA64_BtT"); CHECK_FUNC_CUDA(h_convert_Y216_to_BGRA64_BtT)
		FUNC_CUDA(h_convert_Y216_to_BGRA64_TtB) = (FTh_convert_Y216_to_BGRA64_TtB)GetProcAddress(hCudaConvertLib, "h_convert_Y216_to_BGRA64_TtB"); CHECK_FUNC_CUDA(h_convert_Y216_to_BGRA64_TtB)
		FUNC_CUDA(h_convert_Y216_to_BGRA64_TtT) = (FTh_convert_Y216_to_BGRA64_TtT)GetProcAddress(hCudaConvertLib, "h_convert_Y216_to_BGRA64_TtT"); CHECK_FUNC_CUDA(h_convert_Y216_to_BGRA64_TtT)

		FUNC_CUDA(h_convert_RGBA64_to_Y216_BtB) = (FTh_convert_RGBA64_to_Y216_BtB)GetProcAddress(hCudaConvertLib, "h_convert_RGBA64_to_Y216_BtB"); CHECK_FUNC_CUDA(h_convert_RGBA64_to_Y216_BtB)
		FUNC_CUDA(h_convert_RGBA64_to_Y216_BtT) = (FTh_convert_RGBA64_to_Y216_BtT)GetProcAddress(hCudaConvertLib, "h_convert_RGBA64_to_Y216_BtT"); CHECK_FUNC_CUDA(h_convert_RGBA64_to_Y216_BtT)
		FUNC_CUDA(h_convert_RGBA64_to_Y216_TtB) = (FTh_convert_RGBA64_to_Y216_TtB)GetProcAddress(hCudaConvertLib, "h_convert_RGBA64_to_Y216_TtB"); CHECK_FUNC_CUDA(h_convert_RGBA64_to_Y216_TtB)
		FUNC_CUDA(h_convert_RGBA64_to_Y216_TtT) = (FTh_convert_RGBA64_to_Y216_TtT)GetProcAddress(hCudaConvertLib, "h_convert_RGBA64_to_Y216_TtT"); CHECK_FUNC_CUDA(h_convert_RGBA64_to_Y216_TtT)

		FUNC_CUDA(h_convert_Y210_to_RGBA32_BtB) = (FTh_convert_Y210_to_RGBA32_BtB)GetProcAddress(hCudaConvertLib, "h_convert_Y210_to_RGBA32_BtB"); CHECK_FUNC_CUDA(h_convert_Y210_to_RGBA32_BtB)
		FUNC_CUDA(h_convert_Y210_to_RGBA32_BtT) = (FTh_convert_Y210_to_RGBA32_BtT)GetProcAddress(hCudaConvertLib, "h_convert_Y210_to_RGBA32_BtT"); CHECK_FUNC_CUDA(h_convert_Y210_to_RGBA32_BtT)
		FUNC_CUDA(h_convert_Y210_to_RGBA32_TtB) = (FTh_convert_Y210_to_RGBA32_TtB)GetProcAddress(hCudaConvertLib, "h_convert_Y210_to_RGBA32_TtB"); CHECK_FUNC_CUDA(h_convert_Y210_to_RGBA32_TtB)
		FUNC_CUDA(h_convert_Y210_to_RGBA32_TtT) = (FTh_convert_Y210_to_RGBA32_TtT)GetProcAddress(hCudaConvertLib, "h_convert_Y210_to_RGBA32_TtT"); CHECK_FUNC_CUDA(h_convert_Y210_to_RGBA32_TtT)

		FUNC_CUDA(h_convert_Y210_to_BGRA32_BtB) = (FTh_convert_Y210_to_BGRA32_BtB)GetProcAddress(hCudaConvertLib, "h_convert_Y210_to_BGRA32_BtB"); CHECK_FUNC_CUDA(h_convert_Y210_to_BGRA32_BtB)
		FUNC_CUDA(h_convert_Y210_to_BGRA32_BtT) = (FTh_convert_Y210_to_BGRA32_BtT)GetProcAddress(hCudaConvertLib, "h_convert_Y210_to_BGRA32_BtT"); CHECK_FUNC_CUDA(h_convert_Y210_to_BGRA32_BtT)
		FUNC_CUDA(h_convert_Y210_to_BGRA32_TtB) = (FTh_convert_Y210_to_BGRA32_TtB)GetProcAddress(hCudaConvertLib, "h_convert_Y210_to_BGRA32_TtB"); CHECK_FUNC_CUDA(h_convert_Y210_to_BGRA32_TtB)
		FUNC_CUDA(h_convert_Y210_to_BGRA32_TtT) = (FTh_convert_Y210_to_BGRA32_TtT)GetProcAddress(hCudaConvertLib, "h_convert_Y210_to_BGRA32_TtT"); CHECK_FUNC_CUDA(h_convert_Y210_to_BGRA32_TtT)

		FUNC_CUDA(h_convert_RGBA32_to_Y210_BtB) = (FTh_convert_RGBA32_to_Y210_BtB)GetProcAddress(hCudaConvertLib, "h_convert_RGBA32_to_Y210_BtB"); CHECK_FUNC_CUDA(h_convert_RGBA32_to_Y210_BtB)
		FUNC_CUDA(h_convert_RGBA32_to_Y210_BtT) = (FTh_convert_RGBA32_to_Y210_BtT)GetProcAddress(hCudaConvertLib, "h_convert_RGBA32_to_Y210_BtT"); CHECK_FUNC_CUDA(h_convert_RGBA32_to_Y210_BtT)
		FUNC_CUDA(h_convert_RGBA32_to_Y210_TtB) = (FTh_convert_RGBA32_to_Y210_TtB)GetProcAddress(hCudaConvertLib, "h_convert_RGBA32_to_Y210_TtB"); CHECK_FUNC_CUDA(h_convert_RGBA32_to_Y210_TtB)
		FUNC_CUDA(h_convert_RGBA32_to_Y210_TtT) = (FTh_convert_RGBA32_to_Y210_TtT)GetProcAddress(hCudaConvertLib, "h_convert_RGBA32_to_Y210_TtT"); CHECK_FUNC_CUDA(h_convert_RGBA32_to_Y210_TtT)

		FUNC_CUDA(h_convert_Y210_to_RGBA64_BtB) = (FTh_convert_Y210_to_RGBA64_BtB)GetProcAddress(hCudaConvertLib, "h_convert_Y210_to_RGBA64_BtB"); CHECK_FUNC_CUDA(h_convert_Y210_to_RGBA64_BtB)
		FUNC_CUDA(h_convert_Y210_to_RGBA64_BtT) = (FTh_convert_Y210_to_RGBA64_BtT)GetProcAddress(hCudaConvertLib, "h_convert_Y210_to_RGBA64_BtT"); CHECK_FUNC_CUDA(h_convert_Y210_to_RGBA64_BtT)
		FUNC_CUDA(h_convert_Y210_to_RGBA64_TtB) = (FTh_convert_Y210_to_RGBA64_TtB)GetProcAddress(hCudaConvertLib, "h_convert_Y210_to_RGBA64_TtB"); CHECK_FUNC_CUDA(h_convert_Y210_to_RGBA64_TtB)
		FUNC_CUDA(h_convert_Y210_to_RGBA64_TtT) = (FTh_convert_Y210_to_RGBA64_TtT)GetProcAddress(hCudaConvertLib, "h_convert_Y210_to_RGBA64_TtT"); CHECK_FUNC_CUDA(h_convert_Y210_to_RGBA64_TtT)

		FUNC_CUDA(h_convert_Y210_to_BGRA64_BtB) = (FTh_convert_Y210_to_BGRA64_BtB)GetProcAddress(hCudaConvertLib, "h_convert_Y210_to_BGRA64_BtB"); CHECK_FUNC_CUDA(h_convert_Y210_to_BGRA64_BtB)
		FUNC_CUDA(h_convert_Y210_to_BGRA64_BtT) = (FTh_convert_Y210_to_BGRA64_BtT)GetProcAddress(hCudaConvertLib, "h_convert_Y210_to_BGRA64_BtT"); CHECK_FUNC_CUDA(h_convert_Y210_to_BGRA64_BtT)
		FUNC_CUDA(h_convert_Y210_to_BGRA64_TtB) = (FTh_convert_Y210_to_BGRA64_TtB)GetProcAddress(hCudaConvertLib, "h_convert_Y210_to_BGRA64_TtB"); CHECK_FUNC_CUDA(h_convert_Y210_to_BGRA64_TtB)
		FUNC_CUDA(h_convert_Y210_to_BGRA64_TtT) = (FTh_convert_Y210_to_BGRA64_TtT)GetProcAddress(hCudaConvertLib, "h_convert_Y210_to_BGRA64_TtT"); CHECK_FUNC_CUDA(h_convert_Y210_to_BGRA64_TtT)

		FUNC_CUDA(h_convert_RGBA64_to_Y210_BtB) = (FTh_convert_RGBA64_to_Y210_BtB)GetProcAddress(hCudaConvertLib, "h_convert_RGBA64_to_Y210_BtB"); CHECK_FUNC_CUDA(h_convert_RGBA64_to_Y210_BtB)
		FUNC_CUDA(h_convert_RGBA64_to_Y210_BtT) = (FTh_convert_RGBA64_to_Y210_BtT)GetProcAddress(hCudaConvertLib, "h_convert_RGBA64_to_Y210_BtT"); CHECK_FUNC_CUDA(h_convert_RGBA64_to_Y210_BtT)
		FUNC_CUDA(h_convert_RGBA64_to_Y210_TtB) = (FTh_convert_RGBA64_to_Y210_TtB)GetProcAddress(hCudaConvertLib, "h_convert_RGBA64_to_Y210_TtB"); CHECK_FUNC_CUDA(h_convert_RGBA64_to_Y210_TtB)
		FUNC_CUDA(h_convert_RGBA64_to_Y210_TtT) = (FTh_convert_RGBA64_to_Y210_TtT)GetProcAddress(hCudaConvertLib, "h_convert_RGBA64_to_Y210_TtT"); CHECK_FUNC_CUDA(h_convert_RGBA64_to_Y210_TtT)

		FUNC_CUDA(h_convert_V210_to_RGBA32_BtB) = (FTh_convert_V210_to_RGBA32_BtB)GetProcAddress(hCudaConvertLib, "h_convert_V210_to_RGBA32_BtB"); CHECK_FUNC_CUDA(h_convert_V210_to_RGBA32_BtB)
		FUNC_CUDA(h_convert_V210_to_RGBA32_BtT) = (FTh_convert_V210_to_RGBA32_BtT)GetProcAddress(hCudaConvertLib, "h_convert_V210_to_RGBA32_BtT"); CHECK_FUNC_CUDA(h_convert_V210_to_RGBA32_BtT)
		FUNC_CUDA(h_convert_V210_to_RGBA32_TtB) = (FTh_convert_V210_to_RGBA32_TtB)GetProcAddress(hCudaConvertLib, "h_convert_V210_to_RGBA32_TtB"); CHECK_FUNC_CUDA(h_convert_V210_to_RGBA32_TtB)
		FUNC_CUDA(h_convert_V210_to_RGBA32_TtT) = (FTh_convert_V210_to_RGBA32_TtT)GetProcAddress(hCudaConvertLib, "h_convert_V210_to_RGBA32_TtT"); CHECK_FUNC_CUDA(h_convert_V210_to_RGBA32_TtT)

		FUNC_CUDA(h_convert_V210_to_BGRA32_BtB) = (FTh_convert_V210_to_BGRA32_BtB)GetProcAddress(hCudaConvertLib, "h_convert_V210_to_BGRA32_BtB"); CHECK_FUNC_CUDA(h_convert_V210_to_BGRA32_BtB)
		FUNC_CUDA(h_convert_V210_to_BGRA32_BtT) = (FTh_convert_V210_to_BGRA32_BtT)GetProcAddress(hCudaConvertLib, "h_convert_V210_to_BGRA32_BtT"); CHECK_FUNC_CUDA(h_convert_V210_to_BGRA32_BtT)
		FUNC_CUDA(h_convert_V210_to_BGRA32_TtB) = (FTh_convert_V210_to_BGRA32_TtB)GetProcAddress(hCudaConvertLib, "h_convert_V210_to_BGRA32_TtB"); CHECK_FUNC_CUDA(h_convert_V210_to_BGRA32_TtB)
		FUNC_CUDA(h_convert_V210_to_BGRA32_TtT) = (FTh_convert_V210_to_BGRA32_TtT)GetProcAddress(hCudaConvertLib, "h_convert_V210_to_BGRA32_TtT"); CHECK_FUNC_CUDA(h_convert_V210_to_BGRA32_TtT)

		FUNC_CUDA(h_convert_V210_to_RGBA64_BtB) = (FTh_convert_V210_to_RGBA64_BtB)GetProcAddress(hCudaConvertLib, "h_convert_V210_to_RGBA64_BtB"); CHECK_FUNC_CUDA(h_convert_V210_to_RGBA64_BtB)
		FUNC_CUDA(h_convert_V210_to_RGBA64_BtT) = (FTh_convert_V210_to_RGBA64_BtT)GetProcAddress(hCudaConvertLib, "h_convert_V210_to_RGBA64_BtT"); CHECK_FUNC_CUDA(h_convert_V210_to_RGBA64_BtT)
		FUNC_CUDA(h_convert_V210_to_RGBA64_TtB) = (FTh_convert_V210_to_RGBA64_TtB)GetProcAddress(hCudaConvertLib, "h_convert_V210_to_RGBA64_TtB"); CHECK_FUNC_CUDA(h_convert_V210_to_RGBA64_TtB)
		FUNC_CUDA(h_convert_V210_to_RGBA64_TtT) = (FTh_convert_V210_to_RGBA64_TtT)GetProcAddress(hCudaConvertLib, "h_convert_V210_to_RGBA64_TtT"); CHECK_FUNC_CUDA(h_convert_V210_to_RGBA64_TtT)

		FUNC_CUDA(h_convert_V210_to_BGRA64_BtB) = (FTh_convert_V210_to_BGRA64_BtB)GetProcAddress(hCudaConvertLib, "h_convert_V210_to_BGRA64_BtB"); CHECK_FUNC_CUDA(h_convert_V210_to_BGRA64_BtB)
		FUNC_CUDA(h_convert_V210_to_BGRA64_BtT) = (FTh_convert_V210_to_BGRA64_BtT)GetProcAddress(hCudaConvertLib, "h_convert_V210_to_BGRA64_BtT"); CHECK_FUNC_CUDA(h_convert_V210_to_BGRA64_BtT)
		FUNC_CUDA(h_convert_V210_to_BGRA64_TtB) = (FTh_convert_V210_to_BGRA64_TtB)GetProcAddress(hCudaConvertLib, "h_convert_V210_to_BGRA64_TtB"); CHECK_FUNC_CUDA(h_convert_V210_to_BGRA64_TtB)
		FUNC_CUDA(h_convert_V210_to_BGRA64_TtT) = (FTh_convert_V210_to_BGRA64_TtT)GetProcAddress(hCudaConvertLib, "h_convert_V210_to_BGRA64_TtT"); CHECK_FUNC_CUDA(h_convert_V210_to_BGRA64_TtT)

		FUNC_CUDA(h_convert_NV12_to_RGBA32_BtB) = (FTh_convert_NV12_to_RGBA32_BtB)GetProcAddress(hCudaConvertLib, "h_convert_NV12_to_RGBA32_BtB"); CHECK_FUNC_CUDA(h_convert_NV12_to_RGBA32_BtB)
		FUNC_CUDA(h_convert_NV12_to_RGBA32_BtT) = (FTh_convert_NV12_to_RGBA32_BtT)GetProcAddress(hCudaConvertLib, "h_convert_NV12_to_RGBA32_BtT"); CHECK_FUNC_CUDA(h_convert_NV12_to_RGBA32_BtT)
		FUNC_CUDA(h_convert_NV12_to_RGBA32_TtB) = (FTh_convert_NV12_to_RGBA32_TtB)GetProcAddress(hCudaConvertLib, "h_convert_NV12_to_RGBA32_TtB"); CHECK_FUNC_CUDA(h_convert_NV12_to_RGBA32_TtB)
		FUNC_CUDA(h_convert_NV12_to_RGBA32_TtT) = (FTh_convert_NV12_to_RGBA32_TtT)GetProcAddress(hCudaConvertLib, "h_convert_NV12_to_RGBA32_TtT"); CHECK_FUNC_CUDA(h_convert_NV12_to_RGBA32_TtT)

		FUNC_CUDA(h_convert_NV12_to_BGRA32_BtB) = (FTh_convert_NV12_to_BGRA32_BtB)GetProcAddress(hCudaConvertLib, "h_convert_NV12_to_BGRA32_BtB"); CHECK_FUNC_CUDA(h_convert_NV12_to_BGRA32_BtB)
		FUNC_CUDA(h_convert_NV12_to_BGRA32_BtT) = (FTh_convert_NV12_to_BGRA32_BtT)GetProcAddress(hCudaConvertLib, "h_convert_NV12_to_BGRA32_BtT"); CHECK_FUNC_CUDA(h_convert_NV12_to_BGRA32_BtT)
		FUNC_CUDA(h_convert_NV12_to_BGRA32_TtB) = (FTh_convert_NV12_to_BGRA32_TtB)GetProcAddress(hCudaConvertLib, "h_convert_NV12_to_BGRA32_TtB"); CHECK_FUNC_CUDA(h_convert_NV12_to_BGRA32_TtB)
		FUNC_CUDA(h_convert_NV12_to_BGRA32_TtT) = (FTh_convert_NV12_to_BGRA32_TtT)GetProcAddress(hCudaConvertLib, "h_convert_NV12_to_BGRA32_TtT"); CHECK_FUNC_CUDA(h_convert_NV12_to_BGRA32_TtT)

		FUNC_CUDA(h_convert_P016_to_RGBA64_BtB) = (FTh_convert_P016_to_RGBA64_BtB)GetProcAddress(hCudaConvertLib, "h_convert_P016_to_RGBA64_BtB"); CHECK_FUNC_CUDA(h_convert_P016_to_RGBA64_BtB)
		FUNC_CUDA(h_convert_P016_to_RGBA64_BtT) = (FTh_convert_P016_to_RGBA64_BtT)GetProcAddress(hCudaConvertLib, "h_convert_P016_to_RGBA64_BtT"); CHECK_FUNC_CUDA(h_convert_P016_to_RGBA64_BtT)
		FUNC_CUDA(h_convert_P016_to_RGBA64_TtB) = (FTh_convert_P016_to_RGBA64_TtB)GetProcAddress(hCudaConvertLib, "h_convert_P016_to_RGBA64_TtB"); CHECK_FUNC_CUDA(h_convert_P016_to_RGBA64_TtB)
		FUNC_CUDA(h_convert_P016_to_RGBA64_TtT) = (FTh_convert_P016_to_RGBA64_TtT)GetProcAddress(hCudaConvertLib, "h_convert_P016_to_RGBA64_TtT"); CHECK_FUNC_CUDA(h_convert_P016_to_RGBA64_TtT)

		FUNC_CUDA(h_convert_P016_to_BGRA64_BtB) = (FTh_convert_P016_to_BGRA64_BtB)GetProcAddress(hCudaConvertLib, "h_convert_P016_to_BGRA64_BtB"); CHECK_FUNC_CUDA(h_convert_P016_to_BGRA64_BtB)
		FUNC_CUDA(h_convert_P016_to_BGRA64_BtT) = (FTh_convert_P016_to_BGRA64_BtT)GetProcAddress(hCudaConvertLib, "h_convert_P016_to_BGRA64_BtT"); CHECK_FUNC_CUDA(h_convert_P016_to_BGRA64_BtT)
		FUNC_CUDA(h_convert_P016_to_BGRA64_TtB) = (FTh_convert_P016_to_BGRA64_TtB)GetProcAddress(hCudaConvertLib, "h_convert_P016_to_BGRA64_TtB"); CHECK_FUNC_CUDA(h_convert_P016_to_BGRA64_TtB)
		FUNC_CUDA(h_convert_P016_to_BGRA64_TtT) = (FTh_convert_P016_to_BGRA64_TtT)GetProcAddress(hCudaConvertLib, "h_convert_P016_to_BGRA64_TtT"); CHECK_FUNC_CUDA(h_convert_P016_to_BGRA64_TtT)

		FUNC_CUDA(h_convert_P016_to_RGBA32_BtB) = (FTh_convert_P016_to_RGBA32_BtB)GetProcAddress(hCudaConvertLib, "h_convert_P016_to_RGBA32_BtB"); CHECK_FUNC_CUDA(h_convert_P016_to_RGBA32_BtB)
		FUNC_CUDA(h_convert_P016_to_RGBA32_BtT) = (FTh_convert_P016_to_RGBA32_BtT)GetProcAddress(hCudaConvertLib, "h_convert_P016_to_RGBA32_BtT"); CHECK_FUNC_CUDA(h_convert_P016_to_RGBA32_BtT)
		FUNC_CUDA(h_convert_P016_to_RGBA32_TtB) = (FTh_convert_P016_to_RGBA32_TtB)GetProcAddress(hCudaConvertLib, "h_convert_P016_to_RGBA32_TtB"); CHECK_FUNC_CUDA(h_convert_P016_to_RGBA32_TtB)
		FUNC_CUDA(h_convert_P016_to_RGBA32_TtT) = (FTh_convert_P016_to_RGBA32_TtT)GetProcAddress(hCudaConvertLib, "h_convert_P016_to_RGBA32_TtT"); CHECK_FUNC_CUDA(h_convert_P016_to_RGBA32_TtT)

		FUNC_CUDA(h_convert_P016_to_BGRA32_BtB) = (FTh_convert_P016_to_BGRA32_BtB)GetProcAddress(hCudaConvertLib, "h_convert_P016_to_BGRA32_BtB"); CHECK_FUNC_CUDA(h_convert_P016_to_BGRA32_BtB)
		FUNC_CUDA(h_convert_P016_to_BGRA32_BtT) = (FTh_convert_P016_to_BGRA32_BtT)GetProcAddress(hCudaConvertLib, "h_convert_P016_to_BGRA32_BtT"); CHECK_FUNC_CUDA(h_convert_P016_to_BGRA32_BtT)
		FUNC_CUDA(h_convert_P016_to_BGRA32_TtB) = (FTh_convert_P016_to_BGRA32_TtB)GetProcAddress(hCudaConvertLib, "h_convert_P016_to_BGRA32_TtB"); CHECK_FUNC_CUDA(h_convert_P016_to_BGRA32_TtB)
		FUNC_CUDA(h_convert_P016_to_BGRA32_TtT) = (FTh_convert_P016_to_BGRA32_TtT)GetProcAddress(hCudaConvertLib, "h_convert_P016_to_BGRA32_TtT"); CHECK_FUNC_CUDA(h_convert_P016_to_BGRA32_TtT)

		FUNC_CUDA(h_convert_RGBA32_to_RGBA64_BtB) = (FTh_convert_RGBA32_to_RGBA64_BtB)GetProcAddress(hCudaConvertLib, "h_convert_RGBA32_to_RGBA64_BtB"); CHECK_FUNC_CUDA(h_convert_RGBA32_to_RGBA64_BtB)
		FUNC_CUDA(h_convert_RGBA32_to_RGBA64_BtT) = (FTh_convert_RGBA32_to_RGBA64_BtT)GetProcAddress(hCudaConvertLib, "h_convert_RGBA32_to_RGBA64_BtT"); CHECK_FUNC_CUDA(h_convert_RGBA32_to_RGBA64_BtT)
		FUNC_CUDA(h_convert_RGBA32_to_RGBA64_TtB) = (FTh_convert_RGBA32_to_RGBA64_TtB)GetProcAddress(hCudaConvertLib, "h_convert_RGBA32_to_RGBA64_TtB"); CHECK_FUNC_CUDA(h_convert_RGBA32_to_RGBA64_TtB)
		FUNC_CUDA(h_convert_RGBA32_to_RGBA64_TtT) = (FTh_convert_RGBA32_to_RGBA64_TtT)GetProcAddress(hCudaConvertLib, "h_convert_RGBA32_to_RGBA64_TtT"); CHECK_FUNC_CUDA(h_convert_RGBA32_to_RGBA64_TtT)

		FUNC_CUDA(h_convert_RGBA64_to_RGBA32_BtB) = (FTh_convert_RGBA64_to_RGBA32_BtB)GetProcAddress(hCudaConvertLib, "h_convert_RGBA64_to_RGBA32_BtB"); CHECK_FUNC_CUDA(h_convert_RGBA64_to_RGBA32_BtB)
		FUNC_CUDA(h_convert_RGBA64_to_RGBA32_BtT) = (FTh_convert_RGBA64_to_RGBA32_BtT)GetProcAddress(hCudaConvertLib, "h_convert_RGBA64_to_RGBA32_BtT"); CHECK_FUNC_CUDA(h_convert_RGBA64_to_RGBA32_BtT)
		FUNC_CUDA(h_convert_RGBA64_to_RGBA32_TtB) = (FTh_convert_RGBA64_to_RGBA32_TtB)GetProcAddress(hCudaConvertLib, "h_convert_RGBA64_to_RGBA32_TtB"); CHECK_FUNC_CUDA(h_convert_RGBA64_to_RGBA32_TtB)
		FUNC_CUDA(h_convert_RGBA64_to_RGBA32_TtT) = (FTh_convert_RGBA64_to_RGBA32_TtT)GetProcAddress(hCudaConvertLib, "h_convert_RGBA64_to_RGBA32_TtT"); CHECK_FUNC_CUDA(h_convert_RGBA64_to_RGBA32_TtT)
	}
	else
		return -1;

	return 0;
}