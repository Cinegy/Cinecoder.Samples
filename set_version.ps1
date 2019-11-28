# PowerShell Cinegy Build Script
# COPYRIGHT Cinegy GmbH 2019
param([string]$majorVer,[string]$minorVer,[string]$buildVer,[string]$sourceVer)

$SoftwareVersion = "$majorVer.$minorVer.$buildVer.$sourceVer"
$SoftwareVersionList = $SoftwareVersion.Replace('.', ',')

#make teamcity update with this new version number
Write-Host "##teamcity[buildNumber '$SoftwareVersion']"

#find C++ resource files and update versions, company info and copyright years
$CompanyNameRegex = '(\bVALUE\s+\"CompanyName\"\s*\,\s*\")([^\"]*\\\")*[^\"]*(\")'
$CopyrightRegex = '(\bVALUE\s+\"LegalCopyright\"\s*\,\s*\")([^\"]*\\\")*[^\"]*(\")'
$FileVersionStringRegex = '(\bVALUE\s+\"FileVersion\"\s*\,\s*\")([^\"]*\\\")*[^\"]*(\")'
$FileVersionListRegex = '(?s)(\bFILEVERSION\s+)([\d\*]+\,)*[\d\*]+'
$ProductVersionStringRegex = '(\bVALUE\s+\"ProductVersion\"\s*\,\s*\")([^\"]*\\\")*[^\"]*(\")'
$ProductVersionListRegex = '(?s)(\bPRODUCTVERSION\s+)([\d\*]+\,)*[\d\*]+'

Get-ChildItem -Path ComponentsCustoms.rc -Recurse | ForEach-Object {
    $fileName = $_
    Write-Host "Processing metadata changes for file: $fileName"

    $FileLines = Get-Content -path $fileName 
    
    for($i=0;$i -lt $FileLines.Count;$i++)
    {
        $FileLines[$i] = $FileLines[$i] -Replace $CompanyNameRegex, '$1Cinegy GmbH$3'
        $FileLines[$i] = $FileLines[$i] -Replace $CopyrightRegex, "`$1$([char]0xA9)$((Get-Date).year) Cinegy GmbH. All rights reserved.`$3"
        $FileLines[$i] = $FileLines[$i] -Replace $FileVersionStringRegex, "`${1}$SoftwareVersion`${3}"
        $FileLines[$i] = $FileLines[$i] -Replace $FileVersionListRegex, "`${1}$SoftwareVersionList" 
        $FileLines[$i] = $FileLines[$i] -Replace $ProductVersionStringRegex, "`${1}$SoftwareVersion`${3}"
        $FileLines[$i] = $FileLines[$i] -Replace $ProductVersionListRegex, "`${1}$SoftwareVersionList" 
    }
    
    [System.IO.File]::WriteAllLines($fileName.FullName, $FileLines)
}

#find C#/C++ AssemblyInfo files and update versions, company info and copyright years
$AssemblyCompanyRegex = '(^\s*\[\s*assembly\s*:\s*((System\s*\.)?\s*Reflection\s*\.)?\s*AssemblyCompany(Attribute)?\s*\(\s*@?\")(([^\"]*\\\")*[^\"]*)(\"\s*\)\s*\])'
$AssemblyCopyrightRegex = '(^\s*\[\s*assembly\s*:\s*((System\s*\.)?\s*Reflection\s*\.)?\s*AssemblyCopyright(Attribute)?\s*\(\s*@?\")(([^\"]*\\\")*[^\"]*)(\"\s*\)\s*\])'
$AssemblyVersionRegex = '(^\s*\[\s*assembly\s*:\s*((System\s*\.)?\s*Reflection\s*\.)?\s*AssemblyVersion(Attribute)?\s*\(\s*@?\")(([^\"]*\\\")*[^\"]*)(\"\s*\)\s*\])'
$AssemblyFileVersionRegex = '(^\s*\[\s*assembly\s*:\s*((System\s*\.)?\s*Reflection\s*\.)?\s*AssemblyFileVersion(Attribute)?\s*\(\s*@?\")(([^\"]*\\\")*[^\"]*)(\"\s*\)\s*\])'

Get-ChildItem -Path AssemblyInfo.c* -Recurse | ForEach-Object {
    $fileName = $_
    Write-Host "Processing metadata changes for file: $fileName"

    $FileLines = Get-Content -path $fileName 
    
    for($i=0;$i -lt $FileLines.Count;$i++)
    {
        $FileLines[$i] = $FileLines[$i] -Replace $AssemblyCompanyRegex, '$1Cinegy GmbH$6$7'
        $FileLines[$i] = $FileLines[$i] -Replace $AssemblyCopyrightRegex, "`$1$([char]0xA9)$((Get-Date).year) Cinegy GmbH. All rights reserved.`$6`$7"
        $FileLines[$i] = $FileLines[$i] -Replace $AssemblyVersionRegex, "`${1}$SoftwareVersion`$6`$7"
        $FileLines[$i] = $FileLines[$i] -Replace $AssemblyFileVersionRegex, "`${1}$SoftwareVersion`$6`$7"
    }

    [System.IO.File]::WriteAllLines($fileName.FullName, $FileLines)
}
