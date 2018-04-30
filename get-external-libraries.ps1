#Script to grab external libraries for building these samples
#will require PS5 to unzip - if it fails, you can just unzip by hand to the toolkit directory

$FreeGlut_Package_url = "https://www.transmissionzero.co.uk/files/software/development/GLUT/freeglut-MSVC.zip"

$FreeGlutPackageName =  Split-Path -Path $FreeGlut_Package_url -Leaf

if (Test-Path "./_toolkits/freeglut/include/GL/freeglut.h") { 
	Write-Host "Toolkit for FreeGlut already exists"
	exit 
}
else
{
	Write-Host "Downloading ZIP with FreeGlut package... please be patient"
	md -Force ./_toolkits/freeglut
	iwr -ContentType "application/octet-stream" -Uri $FreeGlut_Package_url -OutFile ./_toolkits/freeglut/$FreeGlutPackageName
	Expand-Archive ./_toolkits/freeglut/$FreeGlutPackageName ./_toolkits/ -force
}