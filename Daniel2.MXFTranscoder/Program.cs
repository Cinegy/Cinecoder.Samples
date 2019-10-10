using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Reflection;
using Cinecoder.Interop;
using Cinecoder.Plugin.Multiplexers.Interop;

namespace Daniel2.MXFTranscoder
{
    class Program
    {
        static public ICC_ClassFactory Factory;
        static public VideoFileReader VideoFile;

        static unsafe int Main(string[] args)
        {
            var buildVersion = Assembly.GetEntryAssembly().GetName().Version.ToString();

            Console.WriteLine($"Daniel2 MXF Transcoder App v{buildVersion}. Copyright (c) 2018 Cinegy LLC\n");

            if(args.Length < 2)
            {
                print_help();
                return 1;
            }

            Console.WriteLine($"Cinecoder version: {Cinecoder_.Version.VersionHi}.{Cinecoder_.Version.VersionLo}.{Cinecoder_.Version.EditionNo}.{Cinecoder_.Version.RevisionNo}\n");

            try
            {
                Cinecoder_.ErrorHandler = new ErrorHandler();

                Factory = Cinecoder_.CreateClassFactory();
                Factory.AssignLicense(License.Companyname, License.Licensekey);
                Factory.LoadPlugin("Cinecoder.Plugin.Multiplexers.dll");

                VideoFile = new VideoFileReader();
                var openResult = VideoFile.Open(args[0]);
                if(openResult != VideoFileReader.OpenResult.OK)
                {
                    Console.Error.WriteLine($"Can't open source file '{args[0]}': {openResult}");
                    return -1;
                }

                var streamInfo = GetStreamInfo();

                Console.WriteLine($"Source file  : {args[0]}");
                Console.WriteLine($"Stream type  : {VideoFile.StreamType}");
                Console.WriteLine($"Frame size   : {streamInfo.FrameSize.cx} x {streamInfo.FrameSize.cy}");
                Console.WriteLine($"Frame rate   : {streamInfo.FrameRate.num * 1000 / streamInfo.FrameRate.denom / 1000.0}");
                Console.WriteLine($"Aspect ratio : {streamInfo.AspectRatio.num}:{streamInfo.AspectRatio.denom}");
                Console.WriteLine("Bit depth    : {0}{1}", streamInfo.Assigned("BitDepth") ? "" : "<unknown>, assuming ", streamInfo.BitDepth);
                Console.WriteLine("Chroma format: {0}{1}", streamInfo.Assigned("ChromaFormat") ? "" : "<unknown>, assuming ", streamInfo.ChromaFormat);

                bool use_cuda = false;
                var encParams = ParseArgs(args, streamInfo, ref use_cuda);

                CC_COLOR_FMT exch_fmt =
                    encParams.ChromaFormat == CC_CHROMA_FORMAT.CC_CHROMA_RGB ||
                    encParams.ChromaFormat == CC_CHROMA_FORMAT.CC_CHROMA_RGBA ?
                        (encParams.BitDepth == 8 ? CC_COLOR_FMT.CCF_RGBA : CC_COLOR_FMT.CCF_RGBA64) :
                        (encParams.BitDepth == 8 ? CC_COLOR_FMT.CCF_YUY2 : CC_COLOR_FMT.CCF_Y216);

                if (use_cuda)
                    encParams.InputColorFormat = exch_fmt;

                Console.WriteLine();
                Console.WriteLine($"Target file  : {args[1]}");
                Console.WriteLine($"Stream type  : {(encParams as ICC_ElementaryStreamInfo).StreamType}");
                Console.WriteLine($"Frame size   : {encParams.FrameSize.cx} x {encParams.FrameSize.cy}");
                Console.WriteLine($"Frame rate   : {encParams.FrameRate.num * 1000 / encParams.FrameRate.denom / 1000.0}");
                Console.WriteLine($"Aspect ratio : {encParams.AspectRatio.num}:{encParams.AspectRatio.denom}");
                Console.WriteLine($"Bit depth    : {encParams.BitDepth}");
                Console.WriteLine($"Chroma format: {encParams.ChromaFormat}");
                if (encParams.RateMode == CC_BITRATE_MODE.CC_CQ)
                    Console.WriteLine($"QuantScale   : {encParams.QuantScale}");
                else
                    Console.WriteLine($"Bitrate      : {encParams.BitRate / 1E6:F2} Mbps");
                Console.WriteLine($"Coding method: {encParams.CodingMethod}");

                var decoder = CreateDecoder(VideoFile.StreamType);
                var encoder = Factory.CreateInstanceByName(use_cuda ? "DanielVideoEncoder_CUDA" : "DanielVideoEncoder") as ICC_VideoEncoder;
                var muxer = Factory.CreateInstanceByName("MXF_OP1A_Multiplexer") as ICC_Multiplexer;
                var pinDescr = Factory.CreateInstanceByName("MXF_MultiplexerPinSettings") as ICC_ElementaryStreamSettings;
                var fileWriter = Factory.CreateInstanceByName("OutputFile") as ICC_OutputFile;

                muxer.Init();
                muxer.OutputCallback = fileWriter;

                pinDescr.StreamType = CC_ELEMENTARY_STREAM_TYPE.CC_ES_TYPE_VIDEO_DANIEL;
                pinDescr.BitRate = encParams.BitRate;
                pinDescr.FrameRate = encParams.FrameRate;

                encoder.Init(encParams);
                encoder.OutputCallback = muxer.CreatePin(pinDescr);

                decoder.Init();
                decoder.OutputCallback = new Dec2EncAdapter(exch_fmt, encoder, use_cuda);

                if (decoder is ICC_ProcessDataPolicyProp)
                    (decoder as ICC_ProcessDataPolicyProp).ProcessDataPolicy = CC_PROCESS_DATA_POLICY.CC_PDP_PARSED_DATA;

                fileWriter.Create(args[1]);

                Console.WriteLine($"\nTotal frames: {VideoFile.Length}");

                long totalFrames = VideoFile.Length;
                long codedFrames = 0;
                DateTime t00 = DateTime.Now, t0 = t00;
                double fps = 0;
                long i0 = 0;

                for (long i = 0; i < totalFrames; i++)
                {
                    Console.Write($"\rframe {i} ({i*100.0/totalFrames:F2}%), {fps:F2} fps \b");
                    var frameData = VideoFile.ReadFrame(i);

                    fixed (byte* p = frameData)
                        decoder.ProcessData((IntPtr)p, (uint)frameData.Length);

                    codedFrames++;

                    if (Console.KeyAvailable && Console.ReadKey().Key == ConsoleKey.Escape)
                    {
                        Console.WriteLine("\nCancelled.");
                        break;
                    }

                    DateTime t1 = DateTime.Now;
                    if((t1 - t0).TotalMilliseconds > 500)
                    {
                        fps = (i - i0) / (t1 - t0).TotalSeconds;
                        i0 = i;
                        t0 = t1;
                    }
                }

                decoder.Done(true);
                encoder.Done(true);
                muxer.Done(true);

                Console.WriteLine($"\nTotal frame(s) processed: {codedFrames}, average fps: {codedFrames/(DateTime.Now-t00).TotalSeconds:F2}, average bitrate: {fileWriter.Length*8/1E6/codedFrames*encParams.FrameRate.num/encParams.FrameRate.denom:F2} Mbps");

                fileWriter.Close();
            }
            catch (Exception e)
            {
                if (e.Message != null)
                    Console.Error.WriteLine($"Error: {e.Message}");

                return e.HResult;
            }

            return 0;
        }

        static ICC_VideoDecoder CreateDecoder(CC_ELEMENTARY_STREAM_TYPE type)
        {
            switch(type)
            {
                case CC_ELEMENTARY_STREAM_TYPE.CC_ES_TYPE_VIDEO_MPEG1:
                case CC_ELEMENTARY_STREAM_TYPE.CC_ES_TYPE_VIDEO_MPEG2:
                    return Factory.CreateInstanceByName("MpegVideoDecoder") as ICC_VideoDecoder;

                case CC_ELEMENTARY_STREAM_TYPE.CC_ES_TYPE_VIDEO_H264:
                    return Factory.CreateInstanceByName("H264VideoDecoder") as ICC_VideoDecoder;

                case CC_ELEMENTARY_STREAM_TYPE.CC_ES_TYPE_VIDEO_AVC_INTRA:
                    return Factory.CreateInstanceByName("AVCIntraDecoder2") as ICC_VideoDecoder;

                case CC_ELEMENTARY_STREAM_TYPE.CC_ES_TYPE_VIDEO_J2K:
                    return Factory.CreateInstanceByName("J2K_VideoDecoder") as ICC_VideoDecoder;

                case CC_ELEMENTARY_STREAM_TYPE.CC_ES_TYPE_VIDEO_DANIEL:
                    return Factory.CreateInstanceByName("DanielVideoDecoder") as ICC_VideoDecoder;
            }

            throw new Exception($"Cannot create a decoder instance for {type}");
        }

        static ICC_VideoSplitter CreateSplitter(CC_ELEMENTARY_STREAM_TYPE type)
        {
            switch (type)
            {
                case CC_ELEMENTARY_STREAM_TYPE.CC_ES_TYPE_VIDEO_MPEG1:
                case CC_ELEMENTARY_STREAM_TYPE.CC_ES_TYPE_VIDEO_MPEG2:
                    return Factory.CreateInstanceByName("MpegVideoSplitter") as ICC_VideoSplitter;

                case CC_ELEMENTARY_STREAM_TYPE.CC_ES_TYPE_VIDEO_H264:
                case CC_ELEMENTARY_STREAM_TYPE.CC_ES_TYPE_VIDEO_AVC_INTRA:
                    return Factory.CreateInstanceByName("H264VideoSplitter") as ICC_VideoSplitter;

                case CC_ELEMENTARY_STREAM_TYPE.CC_ES_TYPE_VIDEO_J2K:
                    return Factory.CreateInstanceByName("J2K_VideoSplitter") as ICC_VideoSplitter;

                case CC_ELEMENTARY_STREAM_TYPE.CC_ES_TYPE_VIDEO_DANIEL:
                    return Factory.CreateInstanceByName("DanielVideoSplitter") as ICC_VideoSplitter;
            }

            throw new Exception($"Cannot create a splitter instance for {type}");
        }

        static unsafe ICC_DanielVideoEncoderSettings GetStreamInfo()
        {
            var frameData = VideoFile.ReadFrame(0);

            var VideoSplitter = CreateSplitter(VideoFile.StreamType);

            fixed (byte* p = frameData)
                VideoSplitter.ProcessData((IntPtr)p, (uint)frameData.Length);

            VideoSplitter.Break(true);

            var sourceInfo = VideoSplitter.GetVideoStreamInfo();

            var sourceInfoDaniel = sourceInfo as ICC_DanielVideoEncoderSettings;
            if (sourceInfoDaniel != null)
                return sourceInfoDaniel;

            var targetInfo = Factory.CreateInstanceByName("DanielVideoEncoderSettings") as ICC_DanielVideoEncoderSettings;

            targetInfo.FrameRate = sourceInfo.FrameRate;
            targetInfo.FrameSize = sourceInfo.FrameSize;
            targetInfo.AspectRatio = sourceInfo.AspectRatio;

            var sourceInfoExt = sourceInfo as ICC_VideoStreamInfoExt;
            if (sourceInfoExt != null)
            {
                targetInfo.BitDepth = sourceInfoExt.BitDepthLuma;
                targetInfo.ChromaFormat = sourceInfoExt.ChromaFormat;
                targetInfo.ColorCoefs = sourceInfoExt.ColorCoefs;
            }

            return targetInfo;
        }

        static void print_help()
        {
            Console.Write(
                "This application transcodes any MXF file into a\n"+
                "Daniel2 MXF file optimized to show fast usage and managed code interop\n"+
                "\n"+
                "Usage: Daniel2.MXFTranscoder.exe <inputfile.MXF> <output_file.MXF> [<switches>]\n"+
                "\n"+
                "Where the switches are:\n"+
                "  /cbr=#     - CBR mode encoding where the arg is the bitrate value is in Mbps\n"+
                "  /cq=#      - CQ mode encoding where the arg is the quant scale\n" +
                "  /method=#  - the encoding method (0,[2])\n" +
                "  /nenc=#    - the number of frame encoders working in a loop ([4])\n" +
                "  /cuda      - use CUDA encoder\n"+
                "\n"+
                "The most of the video stream parameters are obtained from the source stream,\n"+
                "but you can ovveride some of them if you need by the switches:\n"+
                "  /fps=#     - the target frame rate (i.e. 25, 29.97, 60, etc)\n" +
                "  /chroma=#  - the target chroma format (420,422,rgb,rgba)\n" +
                "  /bits=#    - the target bitdepth\n" +
                "\n" +
                "Sample usage (raw file):\n"+      
                "> Daniel2.MXFTranscoder.exe Source.MXF Target.MXF\n"
            );

            Console.WriteLine("\n\nPress Enter to Exit");
            Console.ReadKey();
        }

        static ICC_DanielVideoEncoderSettings ParseArgs(string[] args, ICC_DanielVideoEncoderSettings src, ref bool use_cuda)
        {
            ICC_DanielVideoEncoderSettings encParams = src;

            for (int i = 2; i < args.Length; i++)
            {
            	string arg = args[i].ToLower();

                if (arg.StartsWith("/fps="))
                {
                    double fps = Double.Parse(arg.Substring(5));

                    int fps_n = (int)(fps + 0.5);
                    uint fps_d = 1;

                    if (fps_n != (int)fps)
                    {
                        fps_n *= 1000;
                        fps_d = (uint)(fps_n / fps + 0.5);
                    }

                    encParams.FrameRate = new CC_RATIONAL() { num = fps_n, denom = fps_d };
                }
                else if (arg.StartsWith("/bits="))
                {
                    uint bitdepth = UInt32.Parse(arg.Substring(6));
                    encParams.BitDepth = bitdepth;
                }
                else if (arg == "/chroma=420")
                    encParams.ChromaFormat = CC_CHROMA_FORMAT.CC_CHROMA_420;
                else if (arg == "/chroma=422")
                    encParams.ChromaFormat = CC_CHROMA_FORMAT.CC_CHROMA_422;
                else if (arg == "/chroma=444")
                    encParams.ChromaFormat = CC_CHROMA_FORMAT.CC_CHROMA_4444;
                else if (arg == "/chroma=4444")
                    encParams.ChromaFormat = CC_CHROMA_FORMAT.CC_CHROMA_444;
                else if (arg == "/chroma=rgb")
                    encParams.ChromaFormat = CC_CHROMA_FORMAT.CC_CHROMA_RGB;
                else if (arg == "/chroma=rgba")
                    encParams.ChromaFormat = CC_CHROMA_FORMAT.CC_CHROMA_RGBA;

                else if(arg.StartsWith("/cbr="))
                {
                    int mbs = Int32.Parse(arg.Substring(5));
                    encParams.BitRate = mbs * 1000000L;
                    encParams.RateMode = CC_BITRATE_MODE.CC_CBR;
                }

                else if (arg.StartsWith("/cq="))
                {
                    encParams.QuantScale = float.Parse(arg.Substring(4));
                    encParams.RateMode = CC_BITRATE_MODE.CC_CQ;
                }

                else if (arg.StartsWith("/method="))
                {
                    encParams.CodingMethod = (CC_DANIEL2_CODING_METHOD)Int32.Parse(arg.Substring(8));
                }

                else if (arg == "/cuda")
                {
                    use_cuda = true;
                }

                else if (arg.StartsWith("/nenc="))
                {
                    encParams.NumSingleEncoders = UInt32.Parse(args[i].Substring(6));
                }

                else
                    throw new Exception($"Unknown switch or incorrect switch format: {args[i]}");
            }

            return encParams;
        }
    }
}
