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
using System.Diagnostics.CodeAnalysis;
using System.Runtime.InteropServices;
using Cinecoder.Interop;
using Cinegy.Marshaling;
using Cinegy.Marshaling.Extensions;
using Microsoft.Extensions.Logging;
using SimpleBenchmark.Extensions;

namespace SimpleBenchmark.Cinecoder;

[ExcludeFromCodeCoverage]
public class CinecoderFactory : IDisposable, ICC_ErrorHandler
{
    private readonly ILogger _logger;

    private ICC_ClassFactory _classFactory;
    private ICC_ErrorHandler _oldErrorHandler;

    public CinecoderFactory(ILogger<CinecoderFactory> logger)
    {
        _logger = logger;
    }

    public void ErrorHandlerFunc(int errCode, string errDescription, string fileName, int lineNo)
    {
        _logger?.LogError($"[{errCode:X}] {errDescription} at line {lineNo} in {fileName}",
            errCode,
            errDescription,
            lineNo,
            fileName);

        //allow old error handler to react
        _oldErrorHandler?.ErrorHandlerFunc(errCode, errDescription, fileName, lineNo);
    }

    public T CreateInstance<T>(Guid clsId)
    {
        _logger?.LogDebug($"Creating instance of {typeof(T).Name} [{clsId}]");

        var interfaceType = typeof(T);
        var iid = interfaceType.GUID;

        return _classFactory.CreateInstance(ref clsId, ref iid).As<T>();
    }

    public T CreateInstanceByName<T>(string clsName)
    {
        _logger?.LogDebug($"Creating instance of {typeof(T).Name} [{clsName}]");

        var instanceByName = _classFactory.CreateInstanceByName(clsName);
        var result = instanceByName.As<T>();
        instanceByName.Free();
        return result;
    }

    public void Dispose()
    {
        _classFactory.Free();
        _oldErrorHandler.Free();
    }

    public void Initialize()
    {
        _logger?.LogInformation("Initializing Cinecoder...");

        //receive all cinecoder errors
        Cinecoder_SetErrorHandler(this, out _oldErrorHandler);

        //initialize cinecoder factory
        _classFactory = Cinecoder_CreateClassFactory();
        _classFactory.AssignLicense(License.Companyname,License.Licensekey);
        //check cinecoder version
        var version = Cinecoder_GetVersion();
        _logger?.LogInformation($"Cinecoder {version.VersionHi}.{version.VersionLo}.{version.EditionNo}.{version.RevisionNo}");
    }

    #region Static members


    [DllImport("Cinecoder", PreserveSig = false)]
    [return: MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(ComMarshaler<ICC_ClassFactory>))]
    static extern ICC_ClassFactory Cinecoder_CreateClassFactory();

    [DllImport("Cinecoder", CallingConvention = CallingConvention.StdCall)]
    private static extern CC_VERSION_INFO Cinecoder_GetVersion();

    [DllImport("Cinecoder", CallingConvention = CallingConvention.StdCall)]
    private static extern void Cinecoder_SetErrorHandler(
        [MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(ComMarshaler<ICC_ErrorHandler>))] ICC_ErrorHandler errorHandler,
        [MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(ComMarshaler<ICC_ErrorHandler>))] out ICC_ErrorHandler oldHandler);

    #endregion
}