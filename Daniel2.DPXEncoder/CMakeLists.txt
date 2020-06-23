project (Daniel2_DPXEncoder)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES
	stdafx.cpp
	CEncoderTest.cpp 
	Daniel2.DPXEncoder.cpp
	CudaMemAlloc.cpp
)

set(HEADERS
	stdafx.h
	CEncoderTest.h
	dpx_file.h
)

include_directories("../common")
include_directories("${Cinecoder_INCLUDE_DIRS}")
include_directories("../packages/Cinecoder.Plugin.Multiplexers.1.2.3/sources")
link_directories("${Cinecoder_LIBRARY_DIRS}")

add_compile_options("-D__CINECODER_DISABLE_PRINT_VERSION__")
#add_compile_options("-Wl,-rpath='$ORIGIN'")
if(UNIX AND NOT APPLE)
set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} "-Wl,-rpath='$ORIGIN'")
endif()

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${Cinecoder_LIBRARIES})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE pthread)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE dl)
