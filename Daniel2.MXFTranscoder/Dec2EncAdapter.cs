using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using Cinecoder.Interop;

namespace Daniel2.MXFTranscoder
{
    class Dec2EncAdapter : ICC_DataReadyCallback, ICC_Breakable, IDisposable
    {
        CC_COLOR_FMT _exFormat;
        ICC_VideoEncoder _encoder;
        IntPtr _buffer;
        int _bufsize;
        bool _useCudaMemory;

        public Dec2EncAdapter(CC_COLOR_FMT exchg_fmt, ICC_VideoEncoder encoder, bool useCudaMemory)
        {
            _exFormat = exchg_fmt;
            _encoder = encoder;
            _useCudaMemory = useCudaMemory;
        }
        public void Dispose()
        {
            if (_useCudaMemory)
                cudaFreeHost(_buffer);
            else
                Marshal.FreeHGlobal(_buffer);
        }
        public void DataReady(object pDataProducer)
        {
            var decoder = pDataProducer as ICC_VideoProducer;

            if (_buffer == IntPtr.Zero)
            {
                _bufsize = (int)decoder.GetFrame(_exFormat, IntPtr.Zero, 0);
                if (_useCudaMemory)
                    cudaMallocHost(out _buffer, _bufsize);
                else
                    _buffer = Marshal.AllocHGlobal(_bufsize);
            }

            decoder.GetFrame(_exFormat, _buffer, (uint)_bufsize);
            _encoder.AddFrame(_exFormat, _buffer, (uint)_bufsize);
        }
        public bool Break(bool bFlush)
        {
            if (bFlush)
                _encoder.Done(true);

            return bFlush;
        }

        #region DllImports
        [DllImport("cudart64_80", PreserveSig = true)]
        public static extern int cudaMallocHost(out IntPtr ptr, int size);

        [DllImport("cudart64_80", PreserveSig = true)]
        public static extern int cudaFreeHost(IntPtr ptr);
        #endregion

    }
}
