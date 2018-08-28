using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Reflection;
using System.Threading.Tasks;
using Cinecoder.Interop;

namespace SimpleAudioDecoder
{
    class Program
    {
        static unsafe int Main(string[] args)
        {
            var buildVersion = Assembly.GetEntryAssembly().GetName().Version.ToString();

            Console.WriteLine($"Simple Audio Decoder Test App v{buildVersion}. Copyright (c) 2018 Cinegy LLC\n");

            if (args.Length < 2)
            {
                Console.WriteLine("This sample application decodes an encoded audio elementary file (.mp3, .aac or whatever) into PCM-file\n");
                Console.WriteLine("Usage: SimpleAudioDecoder.exe {MPEG|AAC|LATM|AES3} <input_file> [output_file]");
                return 1;
            }

            Console.WriteLine($"Cinecoder version: {Cinecoder_.Version.VersionHi}.{Cinecoder_.Version.VersionLo}.{Cinecoder_.Version.EditionNo}.{Cinecoder_.Version.RevisionNo}\n");

            string decClassName = null;

            switch(args[0].ToUpper())
            {
                case "MPEG": decClassName = "MpegAudioDecoder"; break;
                case "AAC": decClassName = "AAC_AudioDecoder"; break;
                case "LATM": decClassName = "LATM_AAC_AudioDecoder"; break;
                case "AES3": decClassName = "Aes3AudioDecoder"; break;
                default: Console.Error.WriteLine($"Wrong audio decoder type specified: {args[0]}"); return -1;
            }

            try
            {
                Cinecoder_.ErrorHandler = new ErrorHandler();

                ICC_ClassFactory Factory = Cinecoder_.CreateClassFactory();
                Factory.AssignLicense(License.Companyname, License.Licensekey);

                var audioDecoder = Factory.CreateInstanceByName(decClassName) as ICC_AudioDecoder;

                BinaryReader inputFile = new BinaryReader(new FileStream(args[1], FileMode.Open));

                audioDecoder.OutputCallback = new AudioWriterCallback(args.Length > 2 ? args[2] : args[1] + ".pcm");

                var buffer = new byte[8 * 1024];

                for(;;)
                {
                    int bytes_read = inputFile.Read(buffer, 0, buffer.Length);

                    if (bytes_read == 0)
                        break;

                    fixed (byte* p = buffer)
                        audioDecoder.ProcessData((IntPtr)p, (uint)bytes_read);
                }

                audioDecoder.Done(true);
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


