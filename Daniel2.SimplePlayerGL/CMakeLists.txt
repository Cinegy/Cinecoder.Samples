project (Daniel2_SimplePlayerGL)

cmake_minimum_required(VERSION 3.10)

find_package(OpenAL REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT   REQUIRED)
#find_package(GLEW   REQUIRED)
#find_package(OpenCL REQUIRED)

if(APPLE)
find_library(APPLICATION_SERVICES_LIBRARY ApplicationServices)
#find_library(CORE_GRAPHICS_LIBRARY CoreGraphics)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES
	stdafx.cpp
	AudioSource.cpp
	Block.cpp
	DecodeDaniel2.cpp
	ReadFileDN2.cpp
	Daniel2.SimplePlayerGL.cpp

	../common/cuda_dyn/cuda_dyn_load.cpp
	../external/opencl_dyn_load/src/opencl_dyn_load.cpp

	../external/GLEW/src/glew.c
)

set(HEADERS
	stdafx.h
	AudioSource.h
	Block.h
	CinecoderErrorHandler.h
	DecodeDaniel2.h
	ReadFileDN2.h
)

include_directories("../common")
include_directories("../external/OpenCL")
include_directories("../external/GLEW/include")

include_directories("${Cinecoder_INCLUDE_DIRS}")
link_directories("${Cinecoder_LIBRARY_DIRS}")

if(CUDAConvertLib_FOUND)
include_directories("${CUDAConvertLib_INCLUDE_DIRS}")
link_directories("${CUDAConvertLib_LIBRARY_DIRS}")
endif()

#if(NOT "${GLEW_NO_GLEW_CMAKE}" STREQUAL "")
#    include_directories(${GLEW_INCLUDE_DIRS})
#    link_libraries(${GLEW_LIBRARIES})
#endif()

if (OpenCL_FOUND)
    include_directories(${OPENCL_INCLUDE_DIRS})
endif()

add_compile_options("-D__CINECODER_DISABLE_PRINT_VERSION__")

if(APPLE)
    set(CMAKE_INSTALL_RPATH "@executable_path")
else()
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

#target_link_libraries(${PROJECT_NAME} ${Cinecoder_LIBRARIES} ${OPENGL_LIBRARY} ${OPENAL_LIBRARY} ${GLUT_LIBRARY} pthread dl)

if(NOT APPLE)
target_link_libraries(${PROJECT_NAME} ${Cinecoder_LIBRARIES} ${OPENGL_LIBRARY} ${OPENAL_LIBRARY} GLUT::GLUT pthread dl)
else()
#target_link_libraries(${PROJECT_NAME} ${Cinecoder_LIBRARIES} ${OPENGL_LIBRARY} ${OPENAL_LIBRARY} GLUT::GLUT pthread dl ${APPLICATION_SERVICES_LIBRARY} ${CORE_GRAPHICS_LIBRARY})
target_link_libraries(${PROJECT_NAME} ${Cinecoder_LIBRARIES} ${OPENGL_LIBRARY} ${OPENAL_LIBRARY} GLUT::GLUT pthread dl ${APPLICATION_SERVICES_LIBRARY})
endif()

if(CUDAConvertLib_FOUND)
target_link_libraries(${PROJECT_NAME} ${CUDAConvertLib_LIBRARIES})
endif()

if(UNIX AND NOT APPLE)
target_link_libraries(${PROJECT_NAME} X11)
endif()

#if("${GLEW_NO_GLEW_CMAKE}" STREQUAL "")
#     target_link_libraries(${PROJECT_NAME} GLEW::GLEW)
#endif()

if (OpenCL_FOUND)
    target_link_libraries(${PROJECT_NAME} OpenCL)
endif()
