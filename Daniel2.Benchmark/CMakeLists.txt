project (Daniel2.Benchmark)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-ggdb)

set(SOURCES
	Daniel2.Benchmark.cpp

	../common/cuda_dyn/cuda_dyn_load.cpp
	../external/cuda_drvapi_dyn_load/src/cuda_drvapi_dyn_load.cpp
    ../external/opencl_dyn_load/src/opencl_dyn_load.cpp
)

#set(HEADERS
#)

include_directories("../external/CUDART")
include_directories("../external/OpenCL")
include_directories("../common")
include_directories("${Cinecoder_INCLUDE_DIRS}")
link_directories("${Cinecoder_LIBRARY_DIRS}")

add_compile_options("-D__CINECODER_DISABLE_PRINT_VERSION__")

if(UNIX AND NOT APPLE)
set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} "-Wl,-rpath='$ORIGIN'")
endif()

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${Cinecoder_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PRIVATE dl)
