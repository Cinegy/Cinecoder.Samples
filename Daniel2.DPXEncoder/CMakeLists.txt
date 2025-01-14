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
include_directories("${Cinecoder.Plugin.Multiplexers_INCLUDE_DIRS}")
link_directories("${Cinecoder_LIBRARY_DIRS}")

add_compile_options("-D__CINECODER_DISABLE_PRINT_VERSION__")

if(APPLE)
    set(CMAKE_INSTALL_RPATH "@executable_path")
else()
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${Cinecoder_LIBRARIES})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE pthread)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE dl)
