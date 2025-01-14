cmake_minimum_required(VERSION 3.12)
project(Cinecoder.Samples)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_scripts")
find_package(Cinecoder REQUIRED)
find_package(Cinecoder.Plugin.Multiplexers REQUIRED)

if(UNIX AND NOT APPLE)
find_package(CUDAConvertLib REQUIRED)
endif()

if(APPLE)
    set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/_bin/macosx")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/_bin/linux.aarch64")
    else()
        set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/_bin/linux")
    endif()
endif()

#add_custom_command(
#	TARGET Cinecoder.Samples
#	POST_BUILD 
#	COMMAND ${CMAKE_COMMAND} -E copy
#	"${Cinecoder_LIBRARY_DIRS}/*"
#	"${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
#)

file(MAKE_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})

if(APPLE)
configure_file(${Cinecoder_LIBRARY_DIRS}/libCinecoder.dylib ${EXECUTABLE_OUTPUT_PATH} COPYONLY)
#configure_file(${Cinecoder.Plugin.Multiplexers_LIBRARY_DIRS}/libMXFToolkit_portable.dylib  ${EXECUTABLE_OUTPUT_PATH} COPYONLY)
configure_file(${Cinecoder.Plugin.Multiplexers_LIBRARY_DIRS}/libCinecoder.Plugin.Multiplexers.dylib ${EXECUTABLE_OUTPUT_PATH} COPYONLY)
else()
configure_file(${Cinecoder_LIBRARY_DIRS}/libCinecoder.so ${EXECUTABLE_OUTPUT_PATH} COPYONLY)

if(EXISTS "${Cinecoder_LIBRARY_DIRS}/libd2cudalib.so")
  configure_file(${Cinecoder_LIBRARY_DIRS}/libd2cudalib.so ${EXECUTABLE_OUTPUT_PATH} COPYONLY)
endif()

if(EXISTS "${Cinecoder_LIBRARY_DIRS}/libCinecoder.d2cudalibv2.so")
  configure_file(${Cinecoder_LIBRARY_DIRS}/libCinecoder.d2cudalibv2.so ${EXECUTABLE_OUTPUT_PATH} COPYONLY)
endif()

if(EXISTS "${Cinecoder_LIBRARY_DIRS}/libCinecoder.d2ocllibv2.so")
  configure_file(${Cinecoder_LIBRARY_DIRS}/libCinecoder.d2ocllibv2.so ${EXECUTABLE_OUTPUT_PATH} COPYONLY)
endif()

#configure_file(${Cinecoder.Plugin.Multiplexers_LIBRARY_DIRS}/libMXFToolkit_portable.so  ${EXECUTABLE_OUTPUT_PATH} COPYONLY)
configure_file(${Cinecoder.Plugin.Multiplexers_LIBRARY_DIRS}/libCinecoder.Plugin.Multiplexers.so ${EXECUTABLE_OUTPUT_PATH} COPYONLY)
configure_file(${CUDAConvertLib_LIBRARY_DIRS}/libcudaconvertlib.so ${EXECUTABLE_OUTPUT_PATH} COPYONLY)
endif()

add_subdirectory(SimpleVideoDecoder)
add_subdirectory(SimpleVideoEncoder)
if(NOT ANDROID)
add_subdirectory(Daniel2.SimplePlayerGL)
endif()
add_subdirectory(Daniel2.DPXEncoder)
add_subdirectory(Daniel2.Benchmark)
