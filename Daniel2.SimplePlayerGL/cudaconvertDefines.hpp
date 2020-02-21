#ifdef USE_CUDA_SDK // CUDA
FTh_convert_YUY2_to_RGBA32_BtB FUNC_CUDA(h_convert_YUY2_to_RGBA32_BtB) = nullptr;
FTh_convert_YUY2_to_RGBA32_BtT FUNC_CUDA(h_convert_YUY2_to_RGBA32_BtT) = nullptr;
FTh_convert_YUY2_to_RGBA32_TtB FUNC_CUDA(h_convert_YUY2_to_RGBA32_TtB) = nullptr;
FTh_convert_YUY2_to_RGBA32_TtT FUNC_CUDA(h_convert_YUY2_to_RGBA32_TtT) = nullptr;

FTh_convert_YUY2_to_BGRA32_BtB FUNC_CUDA(h_convert_YUY2_to_BGRA32_BtB) = nullptr;
FTh_convert_YUY2_to_BGRA32_BtT FUNC_CUDA(h_convert_YUY2_to_BGRA32_BtT) = nullptr;
FTh_convert_YUY2_to_BGRA32_TtB FUNC_CUDA(h_convert_YUY2_to_BGRA32_TtB) = nullptr;
FTh_convert_YUY2_to_BGRA32_TtT FUNC_CUDA(h_convert_YUY2_to_BGRA32_TtT) = nullptr;

FTh_convert_RGBA32_to_YUY2_BtB FUNC_CUDA(h_convert_RGBA32_to_YUY2_BtB) = nullptr;
FTh_convert_RGBA32_to_YUY2_BtT FUNC_CUDA(h_convert_RGBA32_to_YUY2_BtT) = nullptr;
FTh_convert_RGBA32_to_YUY2_TtB FUNC_CUDA(h_convert_RGBA32_to_YUY2_TtB) = nullptr;
FTh_convert_RGBA32_to_YUY2_TtT FUNC_CUDA(h_convert_RGBA32_to_YUY2_TtT) = nullptr;

FTh_convert_YUY2_to_RGBA64_BtB FUNC_CUDA(h_convert_YUY2_to_RGBA64_BtB) = nullptr;
FTh_convert_YUY2_to_RGBA64_BtT FUNC_CUDA(h_convert_YUY2_to_RGBA64_BtT) = nullptr;
FTh_convert_YUY2_to_RGBA64_TtB FUNC_CUDA(h_convert_YUY2_to_RGBA64_TtB) = nullptr;
FTh_convert_YUY2_to_RGBA64_TtT FUNC_CUDA(h_convert_YUY2_to_RGBA64_TtT) = nullptr;

FTh_convert_YUY2_to_BGRA64_BtB FUNC_CUDA(h_convert_YUY2_to_BGRA64_BtB) = nullptr;
FTh_convert_YUY2_to_BGRA64_BtT FUNC_CUDA(h_convert_YUY2_to_BGRA64_BtT) = nullptr;
FTh_convert_YUY2_to_BGRA64_TtB FUNC_CUDA(h_convert_YUY2_to_BGRA64_TtB) = nullptr;
FTh_convert_YUY2_to_BGRA64_TtT FUNC_CUDA(h_convert_YUY2_to_BGRA64_TtT) = nullptr;

FTh_convert_RGBA64_to_YUY2_BtB FUNC_CUDA(h_convert_RGBA64_to_YUY2_BtB) = nullptr;
FTh_convert_RGBA64_to_YUY2_BtT FUNC_CUDA(h_convert_RGBA64_to_YUY2_BtT) = nullptr;
FTh_convert_RGBA64_to_YUY2_TtB FUNC_CUDA(h_convert_RGBA64_to_YUY2_TtB) = nullptr;
FTh_convert_RGBA64_to_YUY2_TtT FUNC_CUDA(h_convert_RGBA64_to_YUY2_TtT) = nullptr;

FTh_convert_Y216_to_RGBA32_BtB FUNC_CUDA(h_convert_Y216_to_RGBA32_BtB) = nullptr;
FTh_convert_Y216_to_RGBA32_BtT FUNC_CUDA(h_convert_Y216_to_RGBA32_BtT) = nullptr;
FTh_convert_Y216_to_RGBA32_TtB FUNC_CUDA(h_convert_Y216_to_RGBA32_TtB) = nullptr;
FTh_convert_Y216_to_RGBA32_TtT FUNC_CUDA(h_convert_Y216_to_RGBA32_TtT) = nullptr;

FTh_convert_Y216_to_BGRA32_BtB FUNC_CUDA(h_convert_Y216_to_BGRA32_BtB) = nullptr;
FTh_convert_Y216_to_BGRA32_BtT FUNC_CUDA(h_convert_Y216_to_BGRA32_BtT) = nullptr;
FTh_convert_Y216_to_BGRA32_TtB FUNC_CUDA(h_convert_Y216_to_BGRA32_TtB) = nullptr;
FTh_convert_Y216_to_BGRA32_TtT FUNC_CUDA(h_convert_Y216_to_BGRA32_TtT) = nullptr;

FTh_convert_RGBA32_to_Y216_BtB FUNC_CUDA(h_convert_RGBA32_to_Y216_BtB) = nullptr;
FTh_convert_RGBA32_to_Y216_BtT FUNC_CUDA(h_convert_RGBA32_to_Y216_BtT) = nullptr;
FTh_convert_RGBA32_to_Y216_TtB FUNC_CUDA(h_convert_RGBA32_to_Y216_TtB) = nullptr;
FTh_convert_RGBA32_to_Y216_TtT FUNC_CUDA(h_convert_RGBA32_to_Y216_TtT) = nullptr;

FTh_convert_Y216_to_RGBA64_BtB FUNC_CUDA(h_convert_Y216_to_RGBA64_BtB) = nullptr;
FTh_convert_Y216_to_RGBA64_BtT FUNC_CUDA(h_convert_Y216_to_RGBA64_BtT) = nullptr;
FTh_convert_Y216_to_RGBA64_TtB FUNC_CUDA(h_convert_Y216_to_RGBA64_TtB) = nullptr;
FTh_convert_Y216_to_RGBA64_TtT FUNC_CUDA(h_convert_Y216_to_RGBA64_TtT) = nullptr;

FTh_convert_Y216_to_BGRA64_BtB FUNC_CUDA(h_convert_Y216_to_BGRA64_BtB) = nullptr;
FTh_convert_Y216_to_BGRA64_BtT FUNC_CUDA(h_convert_Y216_to_BGRA64_BtT) = nullptr;
FTh_convert_Y216_to_BGRA64_TtB FUNC_CUDA(h_convert_Y216_to_BGRA64_TtB) = nullptr;
FTh_convert_Y216_to_BGRA64_TtT FUNC_CUDA(h_convert_Y216_to_BGRA64_TtT) = nullptr;

FTh_convert_RGBA64_to_Y216_BtB FUNC_CUDA(h_convert_RGBA64_to_Y216_BtB) = nullptr;
FTh_convert_RGBA64_to_Y216_BtT FUNC_CUDA(h_convert_RGBA64_to_Y216_BtT) = nullptr;
FTh_convert_RGBA64_to_Y216_TtB FUNC_CUDA(h_convert_RGBA64_to_Y216_TtB) = nullptr;
FTh_convert_RGBA64_to_Y216_TtT FUNC_CUDA(h_convert_RGBA64_to_Y216_TtT) = nullptr;

FTh_convert_Y210_to_RGBA32_BtB FUNC_CUDA(h_convert_Y210_to_RGBA32_BtB) = nullptr;
FTh_convert_Y210_to_RGBA32_BtT FUNC_CUDA(h_convert_Y210_to_RGBA32_BtT) = nullptr;
FTh_convert_Y210_to_RGBA32_TtB FUNC_CUDA(h_convert_Y210_to_RGBA32_TtB) = nullptr;
FTh_convert_Y210_to_RGBA32_TtT FUNC_CUDA(h_convert_Y210_to_RGBA32_TtT) = nullptr;

FTh_convert_Y210_to_BGRA32_BtB FUNC_CUDA(h_convert_Y210_to_BGRA32_BtB) = nullptr;
FTh_convert_Y210_to_BGRA32_BtT FUNC_CUDA(h_convert_Y210_to_BGRA32_BtT) = nullptr;
FTh_convert_Y210_to_BGRA32_TtB FUNC_CUDA(h_convert_Y210_to_BGRA32_TtB) = nullptr;
FTh_convert_Y210_to_BGRA32_TtT FUNC_CUDA(h_convert_Y210_to_BGRA32_TtT) = nullptr;

FTh_convert_RGBA32_to_Y210_BtB FUNC_CUDA(h_convert_RGBA32_to_Y210_BtB) = nullptr;
FTh_convert_RGBA32_to_Y210_BtT FUNC_CUDA(h_convert_RGBA32_to_Y210_BtT) = nullptr;
FTh_convert_RGBA32_to_Y210_TtB FUNC_CUDA(h_convert_RGBA32_to_Y210_TtB) = nullptr;
FTh_convert_RGBA32_to_Y210_TtT FUNC_CUDA(h_convert_RGBA32_to_Y210_TtT) = nullptr;

FTh_convert_Y210_to_RGBA64_BtB FUNC_CUDA(h_convert_Y210_to_RGBA64_BtB) = nullptr;
FTh_convert_Y210_to_RGBA64_BtT FUNC_CUDA(h_convert_Y210_to_RGBA64_BtT) = nullptr;
FTh_convert_Y210_to_RGBA64_TtB FUNC_CUDA(h_convert_Y210_to_RGBA64_TtB) = nullptr;
FTh_convert_Y210_to_RGBA64_TtT FUNC_CUDA(h_convert_Y210_to_RGBA64_TtT) = nullptr;

FTh_convert_Y210_to_BGRA64_BtB FUNC_CUDA(h_convert_Y210_to_BGRA64_BtB) = nullptr;
FTh_convert_Y210_to_BGRA64_BtT FUNC_CUDA(h_convert_Y210_to_BGRA64_BtT) = nullptr;
FTh_convert_Y210_to_BGRA64_TtB FUNC_CUDA(h_convert_Y210_to_BGRA64_TtB) = nullptr;
FTh_convert_Y210_to_BGRA64_TtT FUNC_CUDA(h_convert_Y210_to_BGRA64_TtT) = nullptr;

FTh_convert_RGBA64_to_Y210_BtB FUNC_CUDA(h_convert_RGBA64_to_Y210_BtB) = nullptr;
FTh_convert_RGBA64_to_Y210_BtT FUNC_CUDA(h_convert_RGBA64_to_Y210_BtT) = nullptr;
FTh_convert_RGBA64_to_Y210_TtB FUNC_CUDA(h_convert_RGBA64_to_Y210_TtB) = nullptr;
FTh_convert_RGBA64_to_Y210_TtT FUNC_CUDA(h_convert_RGBA64_to_Y210_TtT) = nullptr;

FTh_convert_V210_to_RGBA32_BtB FUNC_CUDA(h_convert_V210_to_RGBA32_BtB) = nullptr;
FTh_convert_V210_to_RGBA32_BtT FUNC_CUDA(h_convert_V210_to_RGBA32_BtT) = nullptr;
FTh_convert_V210_to_RGBA32_TtB FUNC_CUDA(h_convert_V210_to_RGBA32_TtB) = nullptr;
FTh_convert_V210_to_RGBA32_TtT FUNC_CUDA(h_convert_V210_to_RGBA32_TtT) = nullptr;

FTh_convert_V210_to_BGRA32_BtB FUNC_CUDA(h_convert_V210_to_BGRA32_BtB) = nullptr;
FTh_convert_V210_to_BGRA32_BtT FUNC_CUDA(h_convert_V210_to_BGRA32_BtT) = nullptr;
FTh_convert_V210_to_BGRA32_TtB FUNC_CUDA(h_convert_V210_to_BGRA32_TtB) = nullptr;
FTh_convert_V210_to_BGRA32_TtT FUNC_CUDA(h_convert_V210_to_BGRA32_TtT) = nullptr;

FTh_convert_V210_to_RGBA64_BtB FUNC_CUDA(h_convert_V210_to_RGBA64_BtB) = nullptr;
FTh_convert_V210_to_RGBA64_BtT FUNC_CUDA(h_convert_V210_to_RGBA64_BtT) = nullptr;
FTh_convert_V210_to_RGBA64_TtB FUNC_CUDA(h_convert_V210_to_RGBA64_TtB) = nullptr;
FTh_convert_V210_to_RGBA64_TtT FUNC_CUDA(h_convert_V210_to_RGBA64_TtT) = nullptr;

FTh_convert_V210_to_BGRA64_BtB FUNC_CUDA(h_convert_V210_to_BGRA64_BtB) = nullptr;
FTh_convert_V210_to_BGRA64_BtT FUNC_CUDA(h_convert_V210_to_BGRA64_BtT) = nullptr;
FTh_convert_V210_to_BGRA64_TtB FUNC_CUDA(h_convert_V210_to_BGRA64_TtB) = nullptr;
FTh_convert_V210_to_BGRA64_TtT FUNC_CUDA(h_convert_V210_to_BGRA64_TtT) = nullptr;

FTh_convert_NV12_to_RGBA32_BtB FUNC_CUDA(h_convert_NV12_to_RGBA32_BtB) = nullptr;
FTh_convert_NV12_to_RGBA32_BtT FUNC_CUDA(h_convert_NV12_to_RGBA32_BtT) = nullptr;
FTh_convert_NV12_to_RGBA32_TtB FUNC_CUDA(h_convert_NV12_to_RGBA32_TtB) = nullptr;
FTh_convert_NV12_to_RGBA32_TtT FUNC_CUDA(h_convert_NV12_to_RGBA32_TtT) = nullptr;

FTh_convert_NV12_to_BGRA32_BtB FUNC_CUDA(h_convert_NV12_to_BGRA32_BtB) = nullptr;
FTh_convert_NV12_to_BGRA32_BtT FUNC_CUDA(h_convert_NV12_to_BGRA32_BtT) = nullptr;
FTh_convert_NV12_to_BGRA32_TtB FUNC_CUDA(h_convert_NV12_to_BGRA32_TtB) = nullptr;
FTh_convert_NV12_to_BGRA32_TtT FUNC_CUDA(h_convert_NV12_to_BGRA32_TtT) = nullptr;

FTh_convert_P016_to_RGBA64_BtB FUNC_CUDA(h_convert_P016_to_RGBA64_BtB) = nullptr;
FTh_convert_P016_to_RGBA64_BtT FUNC_CUDA(h_convert_P016_to_RGBA64_BtT) = nullptr;
FTh_convert_P016_to_RGBA64_TtB FUNC_CUDA(h_convert_P016_to_RGBA64_TtB) = nullptr;
FTh_convert_P016_to_RGBA64_TtT FUNC_CUDA(h_convert_P016_to_RGBA64_TtT) = nullptr;

FTh_convert_P016_to_BGRA64_BtB FUNC_CUDA(h_convert_P016_to_BGRA64_BtB) = nullptr;
FTh_convert_P016_to_BGRA64_BtT FUNC_CUDA(h_convert_P016_to_BGRA64_BtT) = nullptr;
FTh_convert_P016_to_BGRA64_TtB FUNC_CUDA(h_convert_P016_to_BGRA64_TtB) = nullptr;
FTh_convert_P016_to_BGRA64_TtT FUNC_CUDA(h_convert_P016_to_BGRA64_TtT) = nullptr;

FTh_convert_P016_to_RGBA32_BtB FUNC_CUDA(h_convert_P016_to_RGBA32_BtB) = nullptr;
FTh_convert_P016_to_RGBA32_BtT FUNC_CUDA(h_convert_P016_to_RGBA32_BtT) = nullptr;
FTh_convert_P016_to_RGBA32_TtB FUNC_CUDA(h_convert_P016_to_RGBA32_TtB) = nullptr;
FTh_convert_P016_to_RGBA32_TtT FUNC_CUDA(h_convert_P016_to_RGBA32_TtT) = nullptr;

FTh_convert_P016_to_BGRA32_BtB FUNC_CUDA(h_convert_P016_to_BGRA32_BtB) = nullptr;
FTh_convert_P016_to_BGRA32_BtT FUNC_CUDA(h_convert_P016_to_BGRA32_BtT) = nullptr;
FTh_convert_P016_to_BGRA32_TtB FUNC_CUDA(h_convert_P016_to_BGRA32_TtB) = nullptr;
FTh_convert_P016_to_BGRA32_TtT FUNC_CUDA(h_convert_P016_to_BGRA32_TtT) = nullptr;

FTh_convert_RGBA32_to_RGBA64_BtB FUNC_CUDA(h_convert_RGBA32_to_RGBA64_BtB) = nullptr;
FTh_convert_RGBA32_to_RGBA64_BtT FUNC_CUDA(h_convert_RGBA32_to_RGBA64_BtT) = nullptr;
FTh_convert_RGBA32_to_RGBA64_TtB FUNC_CUDA(h_convert_RGBA32_to_RGBA64_TtB) = nullptr;
FTh_convert_RGBA32_to_RGBA64_TtT FUNC_CUDA(h_convert_RGBA32_to_RGBA64_TtT) = nullptr;

FTh_convert_RGBA64_to_RGBA32_BtB FUNC_CUDA(h_convert_RGBA64_to_RGBA32_BtB) = nullptr;
FTh_convert_RGBA64_to_RGBA32_BtT FUNC_CUDA(h_convert_RGBA64_to_RGBA32_BtT) = nullptr;
FTh_convert_RGBA64_to_RGBA32_TtB FUNC_CUDA(h_convert_RGBA64_to_RGBA32_TtB) = nullptr;
FTh_convert_RGBA64_to_RGBA32_TtT FUNC_CUDA(h_convert_RGBA64_to_RGBA32_TtT) = nullptr;
#endif
