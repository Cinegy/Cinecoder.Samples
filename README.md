# Cinecoder Samples
Projects demonstrating how to use Cinecoder to encode and decode, including the Daniel2 codec

## Build Status
We auto-build this using AppVeyor - here is how we are doing right now:

[![Build status](https://ci.appveyor.com/api/projects/status/cbhe9hx8mne2yuej?svg=true)](https://ci.appveyor.com/project/cinegy/cinecoder-samples)

You can check out the latest compiled binary from our branches here:

[AppVeyor Cinecoder.Samples Project Builder](https://ci.appveyor.com/project/cinegy/cinecoder-samples)

## Project Overview

There are a few different projects included now within these samples, which help to show off some critical features of Cinecoder and the Daniel2 codec.

The most important are:

### Daniel2.SimplePlayerGL



### Daniel2.DPXEncoder

This project is designed to show off the raw speed of Daniel2 encoding, using C++. Clarity has been sacrificed in favour of targetting maximum performance, so if you are just starting out with Daniel2 maybe take a look at the Simple Video Decoder and Encoder managed code projects first.

Our approach with this project is to take an input file(s) and read directly into CUDA pinned
memory, and then upload into the GPU for encoding. It also uses several reading threads and no reading cache to get the maximum possible reading bandwidth. On top-end cards, the PCI bus speed becomes the major limiting factor of the encoding process.

### Dependencies

We rely on a few packages we publish on NuGet for these samples:
[Cinecoder](https://www.nuget.org/packages/Cinecoder/) - core library for encoding / decoding
[Cinecoder Multiplexers](https://www.nuget.org/packages/Cinecoder.Plugin.Multiplexers/) - extension library for MXF manipulation
[Cinecoder GPU Plugins](https://www.nuget.org/packages/Cinecoder.Plugin.GpuCodecs/) - extension library for enabling NVENC / NVDEC and Intel QuickSync accelerated operations
