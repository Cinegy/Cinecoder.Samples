#!/bin/bash

#temporarily remove nuget presence test while fighting docker
#command -v nuget >/dev/null 2>&1 || { echo >&2 "NuGet isn't found. Please install it at first."; exit 1; }
mono /usr/local/bin/nuget.exe restore

mkdir _build.tmp
cd "$_"
cmake .. && make -j8
cd ..
