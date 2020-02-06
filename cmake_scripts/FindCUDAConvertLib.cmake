function(GetNPackageVersion NPACKAGE_NAME)
	if(EXISTS "${PROJECT_SOURCE_DIR}/packages.config")
		file(STRINGS "${PROJECT_SOURCE_DIR}/packages.config" PACKAGES REGEX ".+${NPACKAGE_NAME}\".+")
	endif()
	if(PACKAGES)
		list(GET PACKAGES 0 ITEM)
		string(REGEX MATCH ".+([0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+).+" SMATCH ${ITEM})
		if(SMATCH)
			set(NPACKAGE_VERSION ${CMAKE_MATCH_1} PARENT_SCOPE)
		endif(SMATCH)
	endif(PACKAGES)
endfunction()

function(GetNToolkitDir)
	if(EXISTS "${CMAKE_SOURCE_DIR}/NuGet.config")
		file(STRINGS "${CMAKE_SOURCE_DIR}/NuGet.config" REPOS REGEX ".+repositoryPath.+")
	endif()
	if(REPOS)
		list(GET REPOS 0 ITEM)
		string(REGEX MATCH "value=\"(.+)\"" SMATCH ${ITEM})
		if(SMATCH)
			set(NREPOSITORY_PATH ${CMAKE_MATCH_1} PARENT_SCOPE)
		endif(SMATCH)
	elseif(NOT REPOS)
		set(NREPOSITORY_PATH "packages" PARENT_SCOPE)
	endif(REPOS)
endfunction()

GetNToolkitDir()
GetNPackageVersion("CUDAConvertLib")

if(NREPOSITORY_PATH AND NPACKAGE_VERSION)
	set(CUDAConvertLib_VERSION_STRING "${NPACKAGE_VERSION}")
	set(CUDAConvertLib_ROOT_DIR "${CMAKE_SOURCE_DIR}/${NREPOSITORY_PATH}/CUDAConvertLib.${NPACKAGE_VERSION}")
	set(CUDAConvertLib_INCLUDE_DIRS
		"${CUDAConvertLib_ROOT_DIR}/sources/"
	)

	if(APPLE)
		set(CUDAConvertLib_LIBRARY_DIRS
			"${CUDAConvertLib_ROOT_DIR}/runtimes/osx-x64/native/release/"
		)
	elseif(UNIX)
		set(CUDAConvertLib_LIBRARY_DIRS
			"${CUDAConvertLib_ROOT_DIR}/runtimes/linux-x64/native/release/"
		)
#	set(CUDAConvertLib_LIBRARIES
#		"${CUDAConvertLib_ROOT_DIR}/runtimes/linux-x64/native/release/"
#	)
	endif()

	find_library(CUDAConvertLib_LIBRARIES NAME  d2cudaconvert
	                                 PATHS ${CUDAConvertLib_LIBRARY_DIRS}
	)

#	set(CUDAConvertLib_LIBRARIES
#		"${CUDAConvertLib_ROOT_DIR}/runtimes/linux-x64/native/release/"
#	)

endif()

unset(NPACKAGE_VERSION)
unset(NREPOSITORY_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDAConvertLib REQUIRED_VARS   CUDAConvertLib_INCLUDE_DIRS
                                                            CUDAConvertLib_LIBRARY_DIRS
                                                            CUDAConvertLib_LIBRARIES
                                            VERSION_VAR     CUDAConvertLib_VERSION_STRING
                                            FAIL_MESSAGE    "CUDAConvertLib package was not found!"
                                  )

mark_as_advanced(CUDAConvertLib_ROOT_DIR
                 CUDAConvertLib_VERSION_STRING
                 CUDAConvertLib_INCLUDE_DIRS
                 CUDAConvertLib_LIBRARY_DIRS
                )
