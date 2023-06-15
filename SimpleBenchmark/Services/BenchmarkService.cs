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
using System.Collections.Generic;
using System.Diagnostics.Metrics;
using System.Threading;
using System.Threading.Tasks;
using Cinecoder.Interop;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using SimpleBenchmark.Cinecoder;
using SimpleBenchmark.SerializableModels.Settings;

namespace SimpleBenchmark.Services;

public class BenchmarkService : IHostedService
{
    private readonly ILogger _logger;
    private readonly AppConfig _appConfig;
    private CancellationToken _cancellationToken;
    private readonly Meter _metricsMeter = new($"Cinecoder.SimpleBenchmark.{nameof(BenchmarkService)}");
    
    private const string LineBreak = "---------------------------------------------------------------------";
    private static bool _pendingExit;
    
    private static readonly DateTime StartTime = DateTime.UtcNow;
    private static readonly List<string> ConsoleLines = new(1024);
    private readonly CinecoderFactory _cinecoderFactory;
    private long _nextFrameDueTime;
    private int _totalFrames;
    
    private static string _encoderString = string.Empty;
    private static string _encoderSettingsString = string.Empty;

    #region Constructor and IHostedService

    public BenchmarkService(ILoggerFactory loggerFactory, IConfiguration configuration)
    {
        _logger = loggerFactory.CreateLogger<BenchmarkService>();
        _appConfig = configuration.Get<AppConfig>();

        _cinecoderFactory = new CinecoderFactory(loggerFactory.CreateLogger<CinecoderFactory>());
        _cinecoderFactory.Initialize();

    }

    public Task StartAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("Starting Cinecoder Simple Benchmark service activity");

        _cancellationToken = cancellationToken;

        StartWorker();

        return Task.CompletedTask;
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("Shutting down Cinecoder Simple Benchmark service activity");

        _pendingExit = true;

        _logger.LogInformation("Cinecoder Simple Benchmark service stopped");

        return Task.CompletedTask;
    }

    #endregion

    public void StartWorker()
    {
        new Thread(new ThreadStart(delegate { EncodeFramesWorker(); })).Start();

        if (_appConfig.LiveConsole) Console.Clear();

        var lastConsoleHeartbeatMinute = -1;
        var runtimeFormatString = "{0} hours {1} mins";
        while (!_pendingExit)
        {
            PrintConsoleFeedback();

            Thread.Sleep(60);

            if (DateTime.UtcNow.Minute == lastConsoleHeartbeatMinute) continue;

            lastConsoleHeartbeatMinute = DateTime.UtcNow.Minute;
            var run = DateTime.UtcNow.Subtract(StartTime);
            var runtimeStr = string.Format(runtimeFormatString, Math.Floor(run.TotalHours), run.Minutes);

            _logger.LogInformation($"Running: {runtimeStr}");
        }

        _logger.LogInformation("Logging stopped.");
    }

    private void EncodeFramesWorker()
    {
        var useNvidia = _appConfig.UseGpu;
        SetEncoderStrings(useNvidia);

        _logger.LogInformation("Finished initializing Cinecoder Factory");
        _logger.LogInformation(useNvidia ? "Using Nvidia acceleration" : "Using software encode / decode");

        var encoder = _cinecoderFactory.CreateInstanceByName<ICC_H264VideoEncoder>(_encoderString);

        if (encoder != null)
        {
            _logger.LogInformation($"Created {_encoderString} Cinecoder encoder");
        }
        else
        {
            var msg = $"Failed to create Cinecoder {_encoderString} Encoder object";
            _logger.LogCritical(msg);
            throw new InvalidOperationException(msg);
        }

        var encoderSettings = _cinecoderFactory.CreateInstanceByName<ICC_H264VideoEncoderSettings>(_encoderSettingsString);

        if (encoderSettings != null)
        {
            _logger.LogInformation($"Created {_encoderSettingsString} Cinecoder EncoderSettings");
        }
        else
        {
            var msg = $"Failed to create Cinecoder EncoderSettings object {_encoderSettingsString}";
            _logger.LogCritical(msg);
            throw new InvalidOperationException(msg);
        }

        encoderSettings.FrameSize = new tagSIZE { cx = _appConfig.VideoWidth, cy = _appConfig.VideoHeight };
        encoderSettings.FrameRate = new CC_RATIONAL { denom = 1, num = 50 };
        encoderSettings.BitRate = _appConfig.OutputBitrate;
        encoderSettings.AspectRatio = new CC_RATIONAL { denom = 9, num = 16 };
        encoderSettings.GOP = new CC_GOP_DESCR { n = (uint)_appConfig.GopN, m = (uint)_appConfig.GopM };
        encoderSettings.RateMode = CC_BITRATE_MODE.CC_CBR;
        //encoderSettings.EncoderPreset = CC_NVENC_PRESET.CC_NVENC_PRESET_HQ;

        encoder.Init(encoderSettings);

        const string tsMultiplexerString = "TransportStreamMultiplexer";
        var muxer = _cinecoderFactory.CreateInstanceByName<ICC_Multiplexer>(tsMultiplexerString);

        if (muxer != null)
        {
            _logger.LogInformation($"Created {tsMultiplexerString} Cinecoder Muxer");
        }
        else
        {
            const string msg = $"Failed to create {tsMultiplexerString} Cinecoder Muxer object";
            _logger.LogCritical(msg);
            throw new InvalidOperationException(msg);
        }

        var transportMuxerSettings = _cinecoderFactory.CreateInstanceByName<ICC_TransportMultiplexerSettings2>("TransportMultiplexerSettings");

        transportMuxerSettings.RateMode = CC_BITRATE_MODE.CC_CBR;
        transportMuxerSettings.OutputPolicy = CC_MUX_OUTPUT_POLICY.CC_FLUSH_AT_BUFFER_FULL;
        transportMuxerSettings.MaxOutputBlkSize = 7 * 188;
        transportMuxerSettings.PAT_Period = 100;
        transportMuxerSettings.PMT_Period = 100;
        transportMuxerSettings.PCR_Period = 35;

        muxer.Init(transportMuxerSettings);

        const string tsMuxerPinSettingsString = "TransportMuxerPinSettings";
        var muxerVideoPinSettings =
            _cinecoderFactory.CreateInstanceByName<ICC_TransportMuxerPinSettings>(tsMuxerPinSettingsString);

        if (muxerVideoPinSettings != null)
        {
            _logger.LogInformation($"Created {tsMuxerPinSettingsString} Cinecoder Settings");
        }
        else
        {
            const string msg = $"Failed to create {tsMuxerPinSettingsString} Cinecoder Settings object";
            _logger.LogCritical(msg);
            throw new InvalidOperationException(msg);
        }

        muxerVideoPinSettings.StreamType = CC_ELEMENTARY_STREAM_TYPE.CC_ES_TYPE_VIDEO_H264;
        muxerVideoPinSettings.pid = 0x1001;
        muxerVideoPinSettings.BitRate = encoderSettings.BitRate;
        muxerVideoPinSettings.FrameRate = encoderSettings.FrameRate;

        var videoPin = muxer.CreatePin(muxerVideoPinSettings);

        if (videoPin != null)
        {
            _logger.LogInformation($"Created Cinecoder Muxer pin from {tsMuxerPinSettingsString} settings");
        }
        else
        {
            const string msg = $"Failed to create Cinecoder Muxer pin object from {tsMuxerPinSettingsString} settings";
            _logger.LogCritical(msg);
            throw new InvalidOperationException(msg);
        }

        encoder.OutputCallback = videoPin;
        muxer.OutputCallback = new MuxerCallback(_logger,_cinecoderFactory, _metricsMeter);
        byte[] frameBuffer = null;

        var startDateTime = DateTime.UtcNow;
        _nextFrameDueTime = startDateTime.Ticks;
        var startTimeTicks = startDateTime.Ticks + TimeSpan.TicksPerMillisecond;

        while (!_cancellationToken.IsCancellationRequested)
        {
            //prepare a buffer of appropriate max size to contain result
            //TODO: Make this contain anything other than pure black :)
            frameBuffer ??= new byte[_appConfig.VideoHeight * _appConfig.VideoWidth * 4];

            //if it is time for a frame, dequeue one and decorate it before adding to encoder
            if (DateTime.UtcNow.Ticks >= _nextFrameDueTime)
            {
                unsafe
                {
                    fixed (byte* bPtr = frameBuffer)
                    {
                        encoder.AddFrame(CC_COLOR_FMT.CCF_B8G8R8A8, (IntPtr)bPtr, (uint)frameBuffer.Length);
                    }
                }

                _totalFrames++;
                if (!_appConfig.AsapMode)
                {
                    _nextFrameDueTime = startTimeTicks + _totalFrames * TimeSpan.TicksPerSecond *
                        encoderSettings.FrameRate.denom / encoderSettings.FrameRate.num;
                }
            }
            
            if (!_appConfig.AsapMode)
            {
                Thread.Sleep(1);
            }
        }
        
        Console.WriteLine("Finished!");
    }

    private static void SetEncoderStrings(bool useNvidia)
    {
        if (useNvidia)
        {
            _encoderString = "H264VideoEncoder_NV";
            _encoderSettingsString = "H264VideoEncoderSettings_NV";
        }
        else
        {
            _encoderString = "H264VideoEncoder";
            _encoderSettingsString = "H264VideoEncoderSettings";
        }
    }


    #region ConsoleOutput

    private void PrintConsoleFeedback()
    {
        if (_appConfig.LiveConsole == false) return;

        var runningTime = DateTime.UtcNow.Subtract(StartTime);

        PrintToConsole("Running: {0:hh\\:mm\\:ss}", runningTime);

        PrintToConsole(LineBreak);
        
        Console.CursorVisible = false;
        Console.SetCursorPosition(0, 0);

        foreach (var consoleLine in ConsoleLines)
        {
            ClearCurrentConsoleLine();
            Console.WriteLine(consoleLine);
        }

        Console.CursorVisible = true;

        ConsoleLines.Clear();

    }
    
    private static void PrintToConsole(string message, params object[] arguments)
    {
        if (OperatingSystem.IsWindows())
        {
            // if (_options.SuppressOutput) return;
            ConsoleLines.Add(string.Format(message, arguments));
        }
    }
    
    private static void ClearCurrentConsoleLine()
    {
        if (OperatingSystem.IsWindows())
        {
            // Write space to end of line, and then CR with no LF
            Console.Write("\r".PadLeft(Console.WindowWidth - Console.CursorLeft - 1));
        }
    }

    #endregion

    #region IDispose

    public void Dispose()
    {
        _pendingExit = true;
    }

    #endregion
}


