using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using Cinecoder.Interop;

namespace Daniel2.MXFTranscoder
{
    class VideoFileReader
    {
        ICC_MvxFile IndexFile;
        BinaryReader InputFile;

        public enum OpenResult
        {
            OK = 0,
            FILE_NOT_FOUND,
            FILE_OPEN_ERROR,
            NOT_VIDEO_FILE,
        };

        public OpenResult Open(string filename)
        {
            try
            {
                InputFile = new BinaryReader(new FileStream(filename, FileMode.Open, FileAccess.Read, FileShare.Read));
            }
            catch(FileNotFoundException)
            {
                return OpenResult.FILE_NOT_FOUND;
            }
            catch(Exception)
            {
                return OpenResult.FILE_OPEN_ERROR;
            }

            try
            {
                IndexFile = Program.Factory.CreateInstanceByName("MvxFile") as ICC_MvxFile;
                IndexFile.Open(filename);
            }
            catch(Exception)
            {
                return OpenResult.NOT_VIDEO_FILE;
            }

            return OpenResult.OK;
        }

        private unsafe byte[] ReadFrameInternal(long frame_no)
        {
            CC_MVX_ENTRY entry = IndexFile.FindEntryByCodingNumber((uint)frame_no);
            InputFile.BaseStream.Position = (long)entry.offset;
            var coded_frame = InputFile.ReadBytes((int)entry.size);

            if (entry.Type != 1) // add header for I-frames only
                return coded_frame;

            if (!(IndexFile is ICC_CodedStreamHeaderProp))
                return coded_frame;

            var CodedStreamHeaderGetter = IndexFile as ICC_CodedStreamHeaderProp;

            var hdr_size = CodedStreamHeaderGetter.GetCodedStreamHeader(IntPtr.Zero, 0);

            if (hdr_size == 0)
                return coded_frame;

            var coded_frame_ext = new byte[hdr_size + coded_frame.Length];

            fixed (byte* p = coded_frame_ext)
                CodedStreamHeaderGetter.GetCodedStreamHeader((IntPtr)p, hdr_size);

            Array.Copy(coded_frame, 0, coded_frame_ext, hdr_size, coded_frame.Length);

            return coded_frame_ext;
        }

        public unsafe byte[] ReadFrame(long frame_no)
        {
        	var coded_frame = ReadFrameInternal(frame_no);

            fixed (byte* p = coded_frame)
            {
                var new_size = IndexFile.UnwrapFrame((IntPtr)p, (uint)coded_frame.Length, 0);
                Array.Resize(ref coded_frame, (int)new_size);
            }

            return coded_frame;
        }

        public long Length
        {
            get { return (long)IndexFile.Length; }
        }

        public CC_ELEMENTARY_STREAM_TYPE StreamType
        {
            get { return IndexFile.StreamType; }
        }

    }
}
