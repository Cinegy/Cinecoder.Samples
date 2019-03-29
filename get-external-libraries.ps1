#Script to grab external libraries for building these samples
#will require PS5 to unzip - if it fails, you can just unzip by hand to the toolkit directory

$FreeGlut_Package_url = "https://www.transmissionzero.co.uk/files/software/development/GLUT/freeglut-MSVC.zip"
$FreeGlutPackageName =  Split-Path -Path $FreeGlut_Package_url -Leaf

$CUDAConvertLibUrl = "https://caas-deploy.s3-eu-central-1.amazonaws.com/v1/Cinegy-CUDA-ConvertLib/cudaconvertlib.1.0.0.5.nupkg"
$CUDAConvertLibPackageName =  Split-Path -Path $CUDAConvertLibUrl -Leaf

$OpenALSoftPackageName = "OpenALSoft-1.18.2.zip"

#freeglut unpack
if (Test-Path "./_toolkits/freeglut/include/GL/freeglut.h") { 
	Write-Host "Toolkit for FreeGlut already exists" 
}
else
{
	Expand-Archive ./_toolkits/freeglut/$FreeGlutPackageName ./_toolkits/ -force
}

#OpenALSoft unpack
if (Test-Path "./_toolkits/OpenALSoft/include/al.h") { 
	Write-Host "Toolkit for OpenALSoft already exists" 
}
else
{
	Expand-Archive ./_toolkits/OpenALSoft/$OpenALSoftPackageName ./_toolkits/OpenALSoft -force
}

#temporary download of CUDAConvertLib to local nuget
 if (Test-Path "./LocalNuget/$CUDAConvertLibPackageName") { 
	Write-Host "NuGet package for CUDAConvertLib already exists" 
 }
 else
 {
	Write-Host "Downloading NuGet with CUDAConvertLib package... please be patient"
	New-Item -Path ./LocalNuget -ItemType Directory -Force
	Invoke-WebRequest -ContentType "application/octet-stream" -Uri $CUDAConvertLibUrl -OutFile ./LocalNuget/$CUDAConvertLibPackageName
 }
