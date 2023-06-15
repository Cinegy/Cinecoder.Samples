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
using Cinecoder.Interop;
using Cinegy.Marshaling.Extensions;
using Microsoft.Extensions.Logging;

namespace SimpleBenchmark.Cinecoder;


public class MuxerCallback : ICC_ByteStreamCallback
{
    private long _totalDataMuxed;
    private readonly Meter _metricsMeter;
    private ICC_Demultiplexer _demuxer;
    private readonly CinecoderFactory _cinecoderFactory;
    private readonly ILogger _logger;
    private int _videoPid;
    private bool _useGpu;
    private ICC_Decoder _decoder;

    // ReSharper disable once NotAccessedField.Local
    private readonly ObservableCounter<long> _muxedDataCounter;

    public MuxerCallback(ILogger logger, CinecoderFactory cinecoderFactory, Meter metricsMeter, bool useGpu = false)
    {
        _logger = logger;
        _cinecoderFactory = cinecoderFactory;
        _metricsMeter = metricsMeter;
        _muxedDataCounter = metricsMeter.CreateObservableCounter("muxedData", () => _totalDataMuxed, "bytes");
        _useGpu = useGpu;

        PrepareCinecoderPipeline();
    }

    public void ProcessData(IntPtr pbData, uint cbSize, long pts = -1, object pSender = null)
    {
        unsafe
        {
            _totalDataMuxed+= cbSize;

            var data = new Span<byte>(pbData.ToPointer(), (int)cbSize);
            AddPacketToDemuxer(data.ToArray(),(int)cbSize);
        }
    }

    private void PrepareCinecoderPipeline()
    {
        _demuxer = _cinecoderFactory.CreateInstanceByName<ICC_Demultiplexer>("TransportStreamDemultiplexer");

        if (_demuxer != null)
        {
            _logger.LogInformation("Created demuxer");
        }
        else
        {
            _logger.LogCritical("Failed to create Cinecoder Demultiplexer");
            throw new InvalidOperationException("Demuxer creation fault");
        }

        _demuxer.Init();

    }


    private unsafe void AddPacketToDemuxer(byte[] data, int dataLen)
    {
        if (_demuxer == null) return;

        fixed (byte* bPtr = data)
        {
            _demuxer.ProcessData((IntPtr)bPtr, (uint)dataLen);
        }

        if (_videoPid != 0 || _demuxer.StreamInfo == null) return;

        var info = _demuxer.GetStreamInfo();
        string decoderString;

        if (info.GetProgram(0).GetStream(0).StreamType == CC_ELEMENTARY_STREAM_TYPE.CC_ES_TYPE_VIDEO_MPEG2)
        {
            decoderString = "MpegVideoDecoder";
        }
        else if (info.GetProgram(0).GetStream(0).StreamType == CC_ELEMENTARY_STREAM_TYPE.CC_ES_TYPE_VIDEO_H264)
        {
            decoderString = _useGpu ? "H264VideoDecoder_NV" : "H264VideoDecoder";
        }
        else if (info.GetProgram(0).GetStream(0).StreamType == CC_ELEMENTARY_STREAM_TYPE.CC_ES_TYPE_VIDEO_H265)
        {
            if (!_useGpu)
            {
                _logger.LogCritical("H265 decoding is not supported without GPU");
                return;
            }

            decoderString = "HEVCVideoDecoder_NV";
        }
        else
        {
            _logger.LogCritical($"Unsupported input video codec: {info.GetProgram(0).GetStream(0).StreamType}");
            return;
        }

        _decoder = _cinecoderFactory.CreateInstanceByName<ICC_Decoder>(decoderString);

        if (_decoder != null)
        {
            _logger.LogInformation($"Created {decoderString} decoder");
        }
        else
        {
            _logger.LogCritical($"Critical problem creating {decoderString} decoder");
            return;
        }


        _logger.LogInformation("Creating decoder callback object and assigning");

        try
        {
            var decoderCallback = new DecoderCallback(_logger, _metricsMeter);
            var callbackAsInterface = decoderCallback.As<ICC_DataReadyCallback>();
            _decoder.OutputCallback = callbackAsInterface;
        }
        catch (Exception ex)
        {
            _logger.LogCritical($"Exception working with decoder callback: {ex.Message}");
            return;
        }

        _logger.LogInformation("Decoder callback assigned");

        _videoPid = info.GetProgram(0).GetStream(0).pid;
        _logger.LogInformation($"Found video PID: {_videoPid}");

        _demuxer.Break(false);
        _demuxer.Init();

        try
        {
            var demuxerCallback = new DemuxerCallback(_logger, _videoPid, _decoder, _demuxer);
            var callbackAsInterface = demuxerCallback.As<ICC_DemultiplexedDataCallbackExt>();
            _demuxer.OutputCallback = callbackAsInterface;
        }
        catch (Exception ex)
        {
            _logger.LogCritical($"Exception working with demuxer callback: {ex.Message}");
        }
    }
}