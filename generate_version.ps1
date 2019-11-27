# PowerShell Cinegy Build Script
# COPYRIGHT Cinegy GmbH 2019
param([int]$buildCounter,[string]$SourceRevisionValue="FFFFFF",[string]$OverrideMinorVersion="")

#generate major / minor versions from current year / month
$majorVer= Get-Date -UFormat "%y"
$minorVer= Get-Date -UFormat "%m" 

if($OverrideMinorVersion)
{
    $minorVer = $OverrideMinorVersion
}

$shortRev = $SourceRevisionValue.Substring(0,4)
$sourceAsDecimal = [System.Convert]::ToUInt16($shortRev, 16)

Write-Host "##teamcity[buildNumber '$majorVer.$minorVer.$buildCounter.$sourceAsDecimal']"
Write-Host "##teamcity[setParameter name='MajorVersion' value='$majorVer']"
Write-Host "##teamcity[setParameter name='MinorVersion' value='$minorVer']"
Write-Host "##teamcity[setParameter name='BuildVersion' value='$buildCounter']"
Write-Host "##teamcity[setParameter name='SourceVersion' value='$sourceAsDecimal']"