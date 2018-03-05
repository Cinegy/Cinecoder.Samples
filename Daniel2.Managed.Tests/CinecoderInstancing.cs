using System;
using System.CodeDom;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using NUnit.Framework;
using Cinecoder.Interop;

namespace Daniel2.Managed.Tests
{
    [TestFixture]
    public class CinecoderInstancing
    {

        private const string COMPANYNAME = "cinegy";
        private const string LICENSEKEY = "R5H6F6YDRHG51CEM1SC79SN1U4ZC6T3NYB4KWS54GBFTC7KPM1TJCY4HUF5CC4NG";

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
        public static extern string Cinecoder_GetErrorString(int Error);

        private ICC_ClassFactory _factory;
        private object _settings;
        private object _encoder;
        
        //TEST(Daniel2, CheckCinecoderVersion)
        //{
        //    CC_VERSION_INFO ver = Cinecoder_GetVersion();
        //    printf("Cinecoder version %d.%02d.%02d\n", ver.VersionHi, ver.VersionLo, ver.EditionNo);

        //}
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




        //TEST(Daniel2, CreateCinecoderFactory)
        //{
        //    HRESULT hr;
        //    EXPECT_FALSE(FAILED(hr = Cinecoder_CreateClassFactory(&pFactory)));
        //    //Assert::Fail(L"Cinecoder factory creation error", LINE_INFO());
        //}
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

        //TEST(Daniel2, AssignCinecoderLicense)
        //{
        //    HRESULT hr;

        //    EXPECT_FALSE(FAILED(hr = pFactory->AssignLicense(COMPANYNAME, LICENSEKEY)));
        //    //	Assert::Fail(L"AssignLicense error", LINE_INFO());
        //}

        [TestCase(new object[] {COMPANYNAME, LICENSEKEY})] //all correct sizes will pass, even if not valid
        [TestCase(new object[] { "FAKECOMPANY", "WRONGSIZEKEY"})]
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



        //TEST(Daniel2, CreateEncoder)
        //{
        //    HRESULT hr;

        //    EXPECT_FALSE(FAILED(hr = pFactory->CreateInstance(CLSID_CC_DanielVideoEncoder, IID_ICC_VideoEncoder, (IUnknown**)&pEncoder)));
        //    //Assert::Fail(L"Encoder creation error", LINE_INFO());

        //}
        [TestCase(new object[] { COMPANYNAME, LICENSEKEY, "DanielVideoEncoder" }, ExpectedResult = true)] //should correctly create object
        [TestCase(new object[] { COMPANYNAME, LICENSEKEY, "DanielVideoEncoder_CUDA" }, ExpectedResult = true)] //should correctly create object
        [TestCase(new object[] { COMPANYNAME, LICENSEKEY, "DanielVideoEncoder_FAKE" }, ExpectedResult = false)] //should fail by invalid name
        [TestCase(new object[] { "FAKECOMPANY", LICENSEKEY, "DanielVideoEncoder" },ExpectedResult = false)] //wrong company name should fail to create object
        [TestCase(new object[] { COMPANYNAME, "AAA6F6YDRHG51CEM1SC79SN1U4ZC6T3NYB4KWS54GBFTC7KPM1TJCY4HUF5CC4NG", "DanielVideoEncoder" }, ExpectedResult = false)] //invalid key should fail to create object
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

        //TEST(Daniel2, CreateSettings)
        //{
        //    HRESULT hr;

        //    EXPECT_FALSE(FAILED(hr = pFactory->CreateInstance(CLSID_CC_DanielVideoEncoderSettings, IID_ICC_DanielVideoEncoderSettings, (IUnknown**)&pSettings)));
        //    //Assert::Fail(L"Encoder settings creation error", LINE_INFO());

        //}

        [TestCase(new object[] {"DanielVideoEncoder" ,"DanielVideoEncoderSettings"})]
        [TestCase(new object[] { "DanielVideoEncoder_CUDA", "DanielVideoEncoderSettings_CUDA" })]
        public void CreateSettings(string instanceTypeName, string settingsTypeName)
        {
            try
            {
                CreateEncoder(COMPANYNAME,LICENSEKEY, instanceTypeName);

                _settings = _factory.CreateInstanceByName(settingsTypeName);
                Assert.IsNotNull(_settings, "Returned settings instance is null");
            }
            catch (Exception ex)
            {
                Assert.Fail($"Exception creating Cinecoder Settings: {ex.Message}");
            }
        }


        //TEST_METHOD(InitEncoder)
        //{
        //	HRESULT hr;
        //
        //	CreateSettings();
        //
        //	pSettings->put_FrameSize(MK_SIZE(7680, 4320));
        //	pSettings->put_FrameRate(MK_RATIONAL(60000, 1001));
        //	pSettings->put_InputColorFormat(CCF_V210);
        //
        //	pSettings->put_ChromaFormat(CC_CHROMA_422);
        //	pSettings->put_BitDepth(10);
        //
        //	pSettings->put_RateMode(CC_CBR);
        //	pSettings->put_BitRate(1000000000);
        //	pSettings->put_CodingMethod(CC_D2_METHOD_DEFAULT);
        //
        //	pSettings->put_DeviceID(0);
        //	pSettings->put_NumSingleEncoders(4);
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
        //
        //	LARGE_INTEGER t0, freq;
        //	QueryPerformanceFrequency(&freq);
        //	QueryPerformanceCounter(&t0);
        //
        //	//ENCODER_STATS s0 = {};
        //
        //	//Test.Run();
        //}

        [TestCase(new object[] { "DanielVideoEncoder", "DanielVideoEncoderSettings" })]
       // [TestCase(new object[] { "DanielVideoEncoder_CUDA", "DanielVideoEncoderSettings_CUDA" })]
        public void InitEncoder(string instanceTypeName, string settingsTypeName)
        {
            try
            {
                CreateSettings(instanceTypeName,settingsTypeName);
                var settings = _settings as ICC_DanielVideoEncoderSettings;
                var encoder = _encoder as ICC_DanielVideoEncoder;
                
                Assert.IsNotNull(settings,"Returned settings object is null");
                Assert.IsNotNull(encoder, "Returned encoder object is null");

                // ReSharper disable once PossibleNullReferenceException
                settings.FrameSize = new tagSIZE { cx = 7680, cy= 4320 };
                settings.FrameRate = new CC_RATIONAL { denom = 1001, num = 60000};
                settings.InputColorFormat = CC_COLOR_FMT.CCF_V210;
                settings.ChromaFormat = CC_CHROMA_FORMAT.CC_CHROMA_422;
                settings.BitDepth = 10;
                settings.RateMode = CC_BITRATE_MODE.CC_CBR;
                settings.BitRate = 1000000000;
                settings.CodingMethod = CC_DANIEL2_CODING_METHOD.CC_D2_METHOD_CUDA;
                //pSettings->put_DeviceID(0);
                settings.NumSingleEncoders = 4;

                encoder?.Init(settings);

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
            catch (Exception ex)
            {
                Assert.Fail($"Exception initializing cinecoder encoder type {instanceTypeName}: {ex.Message}");
            }
        }
    }
}
