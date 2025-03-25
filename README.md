# Cinecoder Samples

Projects demonstrating how to use Cinecoder to encode and decode, including the Daniel2 codec.

## Build Status

We auto-build this using AppVeyor - here is how we are doing right now:

[![Build status](https://ci.appveyor.com/api/projects/status/cbhe9hx8mne2yuej?svg=true)](https://ci.appveyor.com/project/cinegy/cinecoder-samples)

You can check out the latest compiled binary from our branches here:

[AppVeyor Cinecoder.Samples Project Builder](https://ci.appveyor.com/project/cinegy/cinecoder-samples)

## Project Overview

To build locally, develop custom apps or to integrate into other projects, please contact sales@cinegy.com to access a development trial license or agree commercial terms for use.

Example license files are visible in the /common folder of the project, although these license example strings are not valid.

The build output artifacts from the sample build should not be used for any production scenarios and are provided for testing or demo purposes only, and will expire a period of time after compilation.

Building for Windows requires Visual Studio 2022, and the samples include the required solution files.

Building for Linux uses cmake, and a dockerfile is included within the repository that is used by Cinegy CI systems to compile the Linux samples. You can build this dockerfile and then run the 'build_samples.sh' script to execute the compilation.

There are a few different projects included now within these samples, which help to show off some critical features of Cinecoder and the Daniel2 codec.

The most important are:

### Daniel2.SimplePlayerGL

An example cross-platform player, showing core techniques for decoding, color-space conversion and rendering to screen.

### Daniel2.DPXEncoder

This project is designed to show off the raw speed of Daniel2 encoding, using C++. Clarity has been sacrificed in favour of targetting maximum performance, so if you are just starting out with Daniel2 maybe take a look at the Simple Video Decoder and Encoder managed code projects first.

Our approach with this project is to take an input file(s) and read directly into CUDA pinned
memory, and then upload into the GPU for encoding. It also uses several reading threads and no reading cache to get the maximum possible reading bandwidth. On top-end cards, the PCI bus speed becomes the major limiting factor of the encoding process.

### Daniel2.Benchmark

This project is a light-weight cross-platform console benchmark written in C++, showing off the maximal potential performance of Cinecoder codecs on different platforms.

It has several goals:

educational:
- to show the approach of using Cinecoder encoders and decoders at their maximal speed with minimal overhead

practical:
- to show the maximal potential performance of a particular Cinecoder codec on a particular platform
- to discover any hidden pitfalls in pipelines - for example, why the processing is slow if you use not-pinned memory with CUDA codecs

Sample profiles and footages you can download from our cloud: https://files.cinegy.com/index.php/s/HSX48LBe8KwzNYb

### Dependencies

We rely on a few packages we publish on NuGet for these samples:

* [Cinecoder](https://www.nuget.org/packages/Cinecoder/) - core library for encoding / decoding
* [Cinecoder Multiplexers](https://www.nuget.org/packages/Cinecoder.Plugin.Multiplexers/) - extension library for MXF manipulation
* [Cinecoder GPU Plugins](https://www.nuget.org/packages/Cinecoder.Plugin.GpuCodecs/) - extension library for enabling NVENC / NVDEC and Intel QuickSync accelerated operations
