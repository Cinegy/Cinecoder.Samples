using System;
using System.IO;
using Cinecoder.Interop;

namespace SimpleAudioDecoder
{
    public class AudioWriterCallback : ICC_DataReadyCallback
    {
        BinaryWriter _file;
        public AudioWriterCallback(string filename)
        {
            _file = new BinaryWriter(new FileStream(filename, FileMode.Create));
        }
        public unsafe void DataReady(object pDataProducer)
        {
            var audioSource = pDataProducer as ICC_AudioProducer;

            var streaminfo = audioSource.GetAudioStreamInfo();
            var frameInfo = audioSource.GetAudioFrameInfo();

            if (_file.BaseStream.Position == 0)
                Console.WriteLine($"Audio stream: freq={streaminfo.SampleRate}, num_ch={streaminfo.NumChannels}, bitdepth={streaminfo.BitsPerSample}, bitrate={streaminfo.BitRate / 1000} Kbps");

            int bufsize = (int)(frameInfo.NumSamples * streaminfo.NumChannels * 2);

            byte[] buffer = new byte[bufsize];
            fixed (byte* p = buffer)
                audioSource.GetAudio(CC_AUDIO_FMT.CAF_PCM16, (IntPtr)p, (uint)bufsize);

            _file.Write(buffer);
        }
    }
}
