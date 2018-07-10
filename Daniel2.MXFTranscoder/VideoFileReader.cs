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

        public byte[] ReadFrame(long frame_no)
        {
            CC_MVX_ENTRY entry = IndexFile.FindEntryByCodingNumber((uint)frame_no);
            InputFile.BaseStream.Position = (long)entry.offset;
            return InputFile.ReadBytes((int)entry.size);
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
