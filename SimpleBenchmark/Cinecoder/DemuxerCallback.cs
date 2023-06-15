using System;
using Cinecoder.Interop;
using Microsoft.Extensions.Logging;

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

namespace SimpleBenchmark.Cinecoder;

public class DemuxerCallback : ICC_DemultiplexedDataCallbackExt
{
    private readonly ILogger _logger;
    private readonly int _videoPid;
    private long _lastPts = -1;
    private readonly ICC_Decoder _decoder;
    private readonly ICC_Demultiplexer _demuxer;
    private const int PtsStepTolerance = 2000 * 720;

    public DemuxerCallback(ILogger logger, int videoPid, ICC_Decoder decoder, ICC_Demultiplexer demuxer)
    {
        _logger = logger;
        _videoPid = videoPid;
        _decoder = decoder;
        _demuxer = demuxer;
    }

    public void ProcessData(uint streamId, IntPtr pbData, uint cbSize, long pts, long dts, long pktOffset, uint flags,
        object pDescr)
    {
        if (streamId != _videoPid) return;

        if (pts != -1 && _lastPts != pts)
        {
            var ptsDelta = pts - _lastPts;
            _lastPts = pts;
            if (Math.Abs(ptsDelta) > PtsStepTolerance)
            {
                _logger.LogWarning($"PTS step of greater than {PtsStepTolerance / 720}ms, resetting decoder");
                _demuxer.Break(false);
                _decoder.Break(false);
            }
        }
        _decoder.ProcessData(pbData, cbSize, 0, pts);
    }
}