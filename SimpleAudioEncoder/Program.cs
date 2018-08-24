using System;
using System.IO;
using System.Reflection;
using Cinecoder.Interop;

namespace SimpleAudioEncoder
{
    class Program
    {
        static unsafe int Main(string[] args)
        {
            var buildVersion = Assembly.GetEntryAssembly().GetName().Version.ToString();

            Console.WriteLine($"Simple Audio Encoder Test App v{buildVersion}. Copyright (c) 2018 Cinegy LLC\n");

            if (args.Length < 3)
            {
                Console.WriteLine("This sample application encodes a PCM audio file into elementary coded audio file (.mp3, .aac or whatever)\n");
                Console.WriteLine("Usage: SimpleAudioEncoder.exe {MPEG|AAC} <input_file> <output_file> <switches>");
                Console.WriteLine("Where switches are:");
                Console.WriteLine("  /br=#   - bitrate (Kbps)");
                Console.WriteLine("  /nch=#  - number of channels");
                Console.WriteLine("  /freq=# - sample rate");
                return 1;
            }

            Console.WriteLine($"Cinecoder version: {Cinecoder_.Version.VersionHi}.{Cinecoder_.Version.VersionLo}.{Cinecoder_.Version.EditionNo}.{Cinecoder_.Version.RevisionNo}\n");

            try
            {
                Cinecoder_.ErrorHandler = new ErrorHandler();

                ICC_ClassFactory Factory = Cinecoder_.CreateClassFactory();
                Factory.AssignLicense(License.Companyname, License.Licensekey);

                int nch = 2;
                int freq = 48000;
                int br = 192;

                for(int i = 3; i < args.Length; i++)
                {
                    if (args[i].ToLower().StartsWith("/br="))
                        br = Int32.Parse(args[i].Substring(4));
                    else if (args[i].ToLower().StartsWith("/nch="))
                        nch = Int32.Parse(args[i].Substring(5));
                    else if (args[i].ToLower().StartsWith("/freq="))
                        freq = Int32.Parse(args[i].Substring(6));
                    else
                        throw new Exception($"Unknow argument: {args[i]}");
                }

                ICC_AudioEncoder audioEncoder = null;

                if (args[0] == "MPEG")
                {
                    if (nch != 1 && nch != 2)
                        throw new Exception("MPEG Audio Encoder: only 1 or 2 audio channels are supported");

                    var encPar = Factory.CreateInstanceByName("MpegAudioEncoderSettings") as ICC_MpegAudioEncoderSettings;
                    encPar.BitRate = br * 1000;
                    encPar.ChannelMode = nch == 1 ? CC_MPG_AUDIO_CHANNEL_MODE.CC_MPG_ACH_MONO : CC_MPG_AUDIO_CHANNEL_MODE.CC_MPG_ACH_STEREO;
                    encPar.SampleRate = (uint)freq;

                    audioEncoder = Factory.CreateInstanceByName("MpegAudioEncoder") as ICC_AudioEncoder;
                    audioEncoder.Init(encPar);
                }
                else if (args[0] == "AAC")
                {
                    var encPar = Factory.CreateInstanceByName("AAC_AudioEncoderSettings") as ICC_AAC_AudioEncoderSettings;
                    encPar.BitRate = br * 1000;
                    encPar.NumChannels = (uint)nch;
                    encPar.SampleRate = (uint)freq;

                    audioEncoder = Factory.CreateInstanceByName("AAC_AudioEncoder") as ICC_AudioEncoder;
                    audioEncoder.Init(encPar);
                }
                else
                    throw new Exception($"Unknown audio encoder type: {args[0]}");

                BinaryReader inputFile = new BinaryReader(new FileStream(args[1], FileMode.Open));

                var outputFile = Factory.CreateInstanceByName("OutputFile") as ICC_OutputFile;
                outputFile.Create(args[2]);

                audioEncoder.OutputCallback = outputFile;

                var buffer = new byte[8 * 1024];

                for (;;)
                {
                    int bytes_read = inputFile.Read(buffer, 0, buffer.Length);

                    if (bytes_read == 0)
                        break;

                    fixed (byte* p = buffer)
                        audioEncoder.ProcessAudio(CC_AUDIO_FMT.CAF_PCM16, (IntPtr)p, (uint)bytes_read);
                }

                audioEncoder.Done(true);
                outputFile.Close();
            }
            catch (Exception e)
            {
                if (e.Message != null)
                    Console.Error.WriteLine($"Error: {e.Message}");

                return e.HResult;
            }

            return 0;
        }
    }
}
