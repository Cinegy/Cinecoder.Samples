using System;

namespace SimpleBenchmark.Extensions;

public static class ObjectExtensions
{
    #region Static members

    public static bool Free(this object @object)
    {
        if (@object is not IDisposable disposable)
        {
            return false;
        }

        disposable.Dispose();
        return true;
    }

    #endregion
}