#!/bin/bash

# Check the number of arguments passed
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 LICENSE_COMPANYNAME LICENSE_KEY"
    exit 1
fi

# Assign the passed arguments to variables
LICENSE_COMPANYNAME=$1
LICENSE_KEY=$2

# Generate the content of the cinecoder_license_string.h file
echo "#define COMPANYNAME \"$LICENSE_COMPANYNAME\"" > common/cinecoder_license_string.h
echo "#define LICENSEKEY \"$LICENSE_KEY\"" >> common/cinecoder_license_string.h

echo "License injected into cinecoder_license_string.h"

echo "internal static class License" > common/cinecoder_license_string.cs
echo "{" >> common/cinecoder_license_string.cs
echo "internal const string Companyname = \"$LICENSE_COMPANYNAME\";" >> common/cinecoder_license_string.cs
echo "internal const string Licensekey = \"$LICENSE_KEY\";" >> common/cinecoder_license_string.cs
echo "}" >> common/cinecoder_license_string.cs

echo "License injected into cinecoder_license_string.cs"
