project (SimpleVideoDecoder)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES
	SimpleVideoDecoder.cpp 
)

set(HEADERS
	ppm_writer.h 
)

include_directories("../common")
include_directories("${Cinecoder_INCLUDE_DIRS}")
link_directories("${Cinecoder_LIBRARY_DIRS}")

add_compile_options("-D__CINECODER_DISABLE_PRINT_VERSION__")

if(UNIX AND NOT APPLE)
set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} "-Wl,-rpath='$ORIGIN'")
endif()

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME} ${Cinecoder_LIBRARIES})
