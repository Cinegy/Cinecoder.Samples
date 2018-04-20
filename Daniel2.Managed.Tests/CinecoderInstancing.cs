using System;
using System.IO;
using System.Runtime.InteropServices;
using NUnit.Framework;
using Cinecoder.Interop;

namespace Daniel2.Managed.Tests
{
    [TestFixture]
    public class CinecoderInstancing
    {
        [DllImport("Cinecoder", PreserveSig = true)]
        private static extern int Cinecoder_CreateClassFactory([MarshalAs(UnmanagedType.Interface)] out ICC_ClassFactory f);

        [DllImport("Cinecoder", PreserveSig = true)]
        private static extern CC_VERSION_INFO Cinecoder_GetVersion();

        [DllImport("Cinecoder", PreserveSig = true)]
        public static extern int Cinecoder_SetErrorHandler([MarshalAs(UnmanagedType.Interface)] ICC_ErrorHandler err,
            [MarshalAs(UnmanagedType.Interface)] out ICC_ErrorHandler old);

        [DllImport("Cinecoder", PreserveSig = true)]
        public static extern int Cinecoder_GetErrorHandler([MarshalAs(UnmanagedType.Interface)] out ICC_ErrorHandler err);
        
        [DllImport("Cinecoder", PreserveSig = false)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        public static extern string Cinecoder_GetErrorString(int error);
        
        private const string Companyname = "Cinegy";
        private const string Licensekey = "TUHSDY56CL4LUT3K5S4BFJF6643S52DE8RALW0ZKB06J54CTA1Z2MB08EDCM5U8P";

        private ICC_ClassFactory _factory;
        private ICC_Settings _settings;
        private ICC_VideoEncoder _encoder;
        
        [Test]
        public void CheckCinecoderVersion()
        {
            try
            {
                var version = Cinecoder_GetVersion();
                Console.WriteLine($"Cinecoder Version : {version.VersionHi}.{version.VersionLo}.{version.EditionNo}.{version.RevisionNo}");
            }
            catch (Exception ex)
            {
                    Assert.Fail($"Probem getting cinecoder versio: {ex.Message}");
            }
        }

        [Test]
        public void CreateCinecoderFactory()
        {
            try
            {
                var result = Cinecoder_CreateClassFactory(out _factory);

                if (result != 0)
                {
                    Assert.Fail($"Cannot create Cinecoder factory - error code: {result}");
                }
                
            }
            catch (Exception ex)
            {
                Assert.Fail($"Exception creating Cinecoder factory: {ex.Message}");
            }
        }

      
        [TestCase(Companyname, Licensekey)] //all correct sizes will pass, even if not valid
        [TestCase("FAKECOMPANY", "WRONGSIZEKEY")]
        public void AssignCinecoderLicense(string companyName, string licenseKey)
        {
            try
            {
                CreateCinecoderFactory();
                _factory.AssignLicense(companyName, licenseKey);
            }
            catch (Exception ex)
            {
                if ((uint)ex.HResult == 0x8004F4F0)
                {
                    if (licenseKey != "WRONGSIZEKEY")
                    {
                        try
                        {
                            //TODO: Fix
                            var errorString = "Broken get error";
                            //var errorString = Cinecoder_GetErrorString(ex.HResult);
                            Assert.Fail($"License with incorrect size key provided when unexpected: {errorString}");
                        }
                        catch(Exception innerEx)
                        {
                            Assert.Fail($"Nested exception looking up string: {innerEx.Message}");
                        }
                        
                    }
                        
                }
                else
                {
                    Assert.Fail($"Exception assigning Cinecoder license: {ex.Message}");
                }
            }
        }
        
        [TestCase(Companyname, Licensekey, "DanielVideoEncoder", ExpectedResult = true)] //should correctly create object
        [TestCase(Companyname, Licensekey, "DanielVideoEncoder_CUDA", ExpectedResult = true)] //should correctly create object
        [TestCase(Companyname, Licensekey, "DanielVideoEncoder_FAKE", ExpectedResult = false)] //should fail by invalid name
        [TestCase("FAKECOMPANY", Licensekey, "DanielVideoEncoder",ExpectedResult = false)] //wrong company name should fail to create object
        [TestCase(Companyname, "AAA6F6YDRHG51CEM1SC79SN1U4ZC6T3NYB4KWS54GBFTC7KPM1TJCY4HUF5CC4NG", "DanielVideoEncoder", ExpectedResult = false)] //invalid key should fail to create object
        public bool CreateEncoder(string companyName, string licenseKey, string instanceTypeName)
        {
            try
            {
                AssignCinecoderLicense( companyName, licenseKey);

                _encoder = _factory.CreateInstanceByName(instanceTypeName);
              
                Assert.IsNotNull(_encoder, "Returned instance is null");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Exception creating Cinecoder Encoder: {ex.Message}");
                return false;
            }

            return true;
        }

        [TestCase("DanielVideoEncoder" ,"DanielVideoEncoderSettings")]
        [TestCase("DanielVideoEncoder_CUDA", "DanielVideoEncoderSettings_CUDA")]
        public void CreateSettings(string instanceTypeName, string settingsTypeName)
        {
            try
            {
                CreateEncoder(Companyname,Licensekey, instanceTypeName);

                _settings = _factory.CreateInstanceByName(settingsTypeName);
                Assert.IsNotNull(_settings, "Returned settings instance is null");
            }
            catch (Exception ex)
            {
                Assert.Fail($"Exception creating Cinecoder Settings: {ex.Message}");
            }
        }
        
        [TestCase("DanielVideoEncoder", "DanielVideoEncoderSettings")]
        [TestCase("DanielVideoEncoder_CUDA", "DanielVideoEncoderSettings_CUDA")]
        public void InitEncoder(string instanceTypeName, string settingsTypeName)
        {
            try
            {
                if (instanceTypeName.Contains("CUDA"))
                {
                    //really bad test for nvidia availability - but it works
                    if (!Directory.Exists("C:\\Program Files\\NVIDIA Corporation"))
                        Assert.Ignore(
                            "Skipping CUDA tests - could not find folder C:\\Program Files\\NVIDIA Corporation.");
                }

                CreateSettings(instanceTypeName, settingsTypeName);
                var settings = _settings as ICC_DanielVideoEncoderSettings;

                Assert.IsNotNull(settings, "Cast settings object is null");
                Assert.IsNotNull(_encoder, "Encoder object is null");

                // ReSharper disable once PossibleNullReferenceException
                settings.FrameSize = new tagSIZE {cx = 7680, cy = 4320};
                settings.FrameRate = new CC_RATIONAL {denom = 1001, num = 60000};
                settings.InputColorFormat = CC_COLOR_FMT.CCF_V210;
                settings.ChromaFormat = CC_CHROMA_FORMAT.CC_CHROMA_422;
                settings.BitDepth = 10;
                settings.RateMode = CC_BITRATE_MODE.CC_CBR;
                settings.BitRate = 1000000000;
                settings.CodingMethod = CC_DANIEL2_CODING_METHOD.CC_D2_METHOD_CUDA;
                //pSettings->put_DeviceID(0);
                settings.NumSingleEncoders = 4;

                _encoder?.Init(settings);

                //
                //	if (FAILED(hr = pEncoder->Init(pSettings)))
                //		Assert::Fail(L"Encoder initialization error", LINE_INFO());
                //
                //	ENCODER_PARAMS par = {};
                //	par.pEncoder = pEncoder;
                //	//par.InputFileName = argv[1];
                //	//par.OutputFileName = argv[2];
                //	par.ColorFormat = CCF_V210;
                //	par.NumReadThreads = 4;
                //	par.QueueSize = 16;
                //
                //	CEncoderTest Test;
                //	if (FAILED(hr = Test.AssignParameters(par)))
                //		Assert::Fail(L"EncoderTest.AssignParameters error", LINE_INFO());

            }
            catch (IgnoreException)
            {
                throw;
            }
            catch (Exception ex)
            {
                Assert.Fail($"Exception initializing cinecoder encoder type {instanceTypeName}: {ex.Message}");
            }
        }
    }
}
