/* Copyright 2022-2023 Cinegy GmbH.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

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
        private int _framesLastSecond;
        private readonly Histogram<int> _framesLastSecondHistogram;
        // ReSharper disable once NotAccessedField.Local
        private readonly ObservableGauge<int> _framesLastSecondGauge;

        private long _pts;
        private int _fps;
        private bool _initialized;
        private int _lastFpsSecond = DateTime.UtcNow.TimeOfDay.Seconds;

        public DecoderCallback(ILogger logger, Meter metricsMeter)
        {
            _logger = logger;
            _framesLastSecondHistogram = metricsMeter.CreateHistogram<int>("benchmarkRate", "frames", "Encoded / Muxed / Demuxed / Decoded frames measured in the last second (histogram)");
            _framesLastSecondGauge = metricsMeter.CreateObservableGauge("benchmarkRateGauge", () => _framesLastSecond, "frames", "Encoded / Muxed / Demuxed / Decoded frames measured in last second");
        }

        public void DataReady(object pDataProducer)
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

                    videoProducer.Free();
                    videoFrameInfo.Free();
                    videoFrameInfo.Free();
                    _initialized = true;
                    return;
                }
                
                videoProducer.GetFrame(CC_COLOR_FMT.CCF_B8G8R8A8, _frameBufferPtr, _frameBufferSize);
                videoProducer.Free();

                //Do nothing with frame - just add some counters!
                _fps++;

                if (_lastFpsSecond == DateTime.UtcNow.Second) return;

                _logger.LogInformation($"Encoded / decoded {_fps} frames in last second");
                _framesLastSecondHistogram.Record(_fps);
                _framesLastSecond = _fps;
                _fps = 0;
                _lastFpsSecond = DateTime.UtcNow.Second;
            }
            catch (Exception ex)
            {
                _logger.LogError($"Exception inside Decoder callback: {ex.Message}");
            }
        }



    }

}