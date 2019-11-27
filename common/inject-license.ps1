# PowerShell Cinegy Build Script
# COPYRIGHT Cinegy GmbH 2019
param([string]$CompanyName="cinegy",[string]$LicenseKey="FFFFFF")

#define regular expressions to be used when checking for #define statements
$companyNameRegex = "(\s*#define\s+COMPANYNAME\s+`")(\S+)`""
$licenseKeyRegex = "(\s*#define\s+LICENSEKEY\s+`")(\S+)`""

$FileLines = Get-Content -path ".\cinecoder_license_string_example.h"  

for($i=0;$i -lt $FileLines.Count;$i++)
{
    $FileLines[$i] = $FileLines[$i] -Replace $companyNameRegex, "`${1}$CompanyName`""
    $FileLines[$i] = $FileLines[$i] -Replace $licenseKeyRegex, "`${1}$LicenseKey`""
}

[Environment]::CurrentDirectory = (Get-Location -PSProvider FileSystem).ProviderPath
[System.IO.File]::WriteAllLines("cinecoder_license_string.h", $FileLines)


#define regular expressions to be used when checking for const statements
$companyNameRegex = "(\s*#define\s+COMPANYNAME\s+`")(\S+)`"" #todo: update for the actual CS values
$licenseKeyRegex = "(\s*#define\s+LICENSEKEY\s+`")(\S+)`""

$FileLines = Get-Content -path ".\cinecoder_license_string_example.cs"  

for($i=0;$i -lt $FileLines.Count;$i++)
{
    $FileLines[$i] = $FileLines[$i] -Replace $companyNameRegex, "`${1}$CompanyName`""
    $FileLines[$i] = $FileLines[$i] -Replace $licenseKeyRegex, "`${1}$LicenseKey`""
}

[System.IO.File]::WriteAllLines("cinecoder_license_string.cs", $FileLines)
