cmake_minimum_required(VERSION 3.12)
project(Cinecoder.Samples)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_scripts")
find_package(Cinecoder REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/_bin/linux")

#add_custom_command(
#	TARGET Cinecoder.Samples
#	POST_BUILD 
#	COMMAND ${CMAKE_COMMAND} -E copy
#	"${Cinecoder_LIBRARY_DIRS}/*"
#	"${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
#)

configure_file(${Cinecoder_LIBRARY_DIRS}/libCinecoder.so ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} COPYONLY)
configure_file(${Cinecoder_LIBRARY_DIRS}/libd2cudalib.so ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} COPYONLY)

add_subdirectory(SimpleVideoDecoder)
add_subdirectory(SimpleVideoEncoder)
add_subdirectory(Daniel2.SimplePlayerGL)
add_subdirectory(Daniel2.DPXEncoder)
