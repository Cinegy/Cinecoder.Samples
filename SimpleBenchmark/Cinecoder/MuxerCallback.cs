using System;
using System.Diagnostics.Metrics;
using System.IO;
using Cinecoder.Interop;

namespace SimpleBenchmark.Cinecoder;


public class MuxerCallback : ICC_ByteStreamCallback
{
    private long _totalDataMuxed;
    private readonly Meter _metricsMeter;

    private readonly ObservableCounter<long> _muxedDataCounter;

    public MuxerCallback( Meter metricsMeter)
    {
        _metricsMeter = metricsMeter;
        _muxedDataCounter = metricsMeter.CreateObservableCounter("muxedData", () => _totalDataMuxed, "bytes");
    }

    public void ProcessData(IntPtr pbData, uint cbSize, long pts = -1, object pSender = null)
    {   
        _totalDataMuxed+= cbSize;
    }

}