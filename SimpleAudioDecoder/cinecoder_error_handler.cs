using System;
using Cinecoder.Interop;

internal class ErrorHandler : ICC_ErrorHandler
{
    public void ErrorHandlerFunc(int ErrCode, string ErrDescription, string pFileName, int LineNo)
    {
        if (ErrCode == unchecked((int)0x80004004))  // ignore E_ABORT error
            return;

        string strErr = Cinecoder_.GetErrorString(ErrCode);

        if (ErrCode == 0)
        {
            Console.WriteLine("\nInformation from {2}({3}) : {4}", ErrCode, strErr, pFileName, LineNo, ErrDescription == null ? "<none>" : ErrDescription);
            return;
        }

        string s = "Error";
        if (ErrCode > 0)
            s = "Warning";

        Console.WriteLine("\n" + s + " {0:X}h ({1}) in {2}({3}) : {4}", ErrCode, strErr, pFileName, LineNo, ErrDescription == null ? "<none>" : ErrDescription);
    }
}
