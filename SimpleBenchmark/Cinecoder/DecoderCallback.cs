using System;
using System.Diagnostics.Metrics;
using System.Runtime.InteropServices;
using Cinecoder.Interop;
using Cinegy.Marshaling.Extensions;
using Microsoft.Extensions.Logging;
using SimpleBenchmark.Extensions;

namespace SimpleBenchmark.Cinecoder
{
    public class DecoderCallback : ICC_DataReadyCallback
    {
        private IntPtr _frameBufferPtr = IntPtr.Zero;
        private uint _frameBufferSize;
        private readonly ILogger _logger;

        private readonly ObservableGauge<long> _latestPtsMilliseconds;
        
        private long _pts;
        private long _dts;
        private int _denom;
        private int _num;
        private bool _initialized;

        public DecoderCallback(ILogger logger, ICC_VideoEncoder encoder, Meter metricsMeter)
        {
            _logger = logger;
            _latestPtsMilliseconds = metricsMeter.CreateObservableGauge("latestPts", () => _pts / 720, "ms");
        }

        public unsafe void DataReady(object pDataProducer)
        {
            try
            {
                var videoProducer = pDataProducer?.As<ICC_VideoProducer>();

                if (videoProducer == null) return;

                var videoFrameInfo = videoProducer.GetVideoFrameInfo();

                if (videoFrameInfo == null)
                {
                    videoProducer.Free();
                    return;
                }

                if (_frameBufferPtr == IntPtr.Zero)
                {
                    _frameBufferSize = videoProducer.GetFrame(CC_COLOR_FMT.CCF_B8G8R8A8, IntPtr.Zero, 0);
                    _frameBufferPtr = Marshal.AllocHGlobal((int)_frameBufferSize);
                }

                _pts = videoFrameInfo.pts;
                _dts = videoFrameInfo.dts;
                
                //wait for a proper PTS value before we do anything with frames...
                if (_pts == 0)
                {
                    videoProducer.Free();
                    videoFrameInfo.Free();
                    videoFrameInfo.Free();
                    return;
                }

                //only init once, if we are not already initializing
                if (_initialized)
                {
                    var videoStreamInfo = videoProducer.GetVideoStreamInfo();
                    if (videoStreamInfo == null)
                     throw new NullReferenceException("Unable to retrieve VideoStreamInfo from videoProducer");
                    
                    _logger.LogInformation($"Input video format {videoStreamInfo.FrameSize.cx}x{videoStreamInfo.FrameSize.cy} at {Math.Round((double)videoStreamInfo.FrameRate.num/videoStreamInfo.FrameRate.denom,2)} fps");
                    _denom = (int)videoStreamInfo.FrameRate.denom;
                    _num = videoStreamInfo.FrameRate.num;
                    
                    videoProducer.Free();
                    videoFrameInfo.Free();
                    videoFrameInfo.Free();
                    _initialized = true;
                    return;
                }
                
                videoProducer.GetFrame(CC_COLOR_FMT.CCF_B8G8R8A8, _frameBufferPtr, _frameBufferSize);
                videoProducer.Free();
                
                //TODO: Do nothing with frame - just add some counters!
                //_pipelineService.AddFrame(_frameBufferPtr, (int)_frameBufferSize, _pts, _dts);
                

            }
            catch (Exception ex)
            {
                _logger.LogError($"Exception inside Decoder callback: {ex.Message}");
            }
        }

    }

}