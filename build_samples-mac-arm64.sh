#!/bin/bash

set -e
set -o pipefail

#temporarily remove nuget presence test while fighting docker
#command -v nuget >/dev/null 2>&1 || { echo >&2 "NuGet isn't found. Please install it at first."; exit 1; }

if [ -f "/usr/local/bin/nuget.exe" ]; then
	mono /usr/local/bin/nuget.exe restore
else
	echo "⚠️  NuGet not found, skipping dependency recovery"
fi

[ -d _build.tmp ] && rm -rf _build.tmp
mkdir _build.tmp
cd _build.tmp

BUILDTYPE=${1:-Debug}

echo "Building Cinecoder Samples in mode:" $BUILDTYPE

cmake -DCMAKE_BUILD_TYPE=$BUILDTYPE -DCMAKE_OSX_ARCHITECTURES=arm64 ..

make -j8

cd ..
