function(GetNPackageVersion NPACKAGE_NAME)
	if(EXISTS "${PROJECT_SOURCE_DIR}/packages.config")
		file(STRINGS "${PROJECT_SOURCE_DIR}/packages.config" PACKAGES REGEX ".+${NPACKAGE_NAME}\".+")
	endif()
	if(PACKAGES)
		list(GET PACKAGES 0 ITEM)
		string(REGEX MATCH ".+([0-9]+\\.[0-9]+\\.[0-9]+).+" SMATCH ${ITEM})
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
GetNPackageVersion("Cinecoder.Plugin.Multiplexers")

if(NREPOSITORY_PATH AND NPACKAGE_VERSION)
	set(Cinecoder.Plugin.Multiplexers_VERSION_STRING ${NPACKAGE_VERSION})
	set(Cinecoder.Plugin.Multiplexers_ROOT_DIR "${CMAKE_SOURCE_DIR}/${NREPOSITORY_PATH}/Cinecoder.Plugin.Multiplexers.${NPACKAGE_VERSION}")
	set(Cinecoder.Plugin.Multiplexers_INCLUDE_DIRS
		"${Cinecoder.Plugin.Multiplexers_ROOT_DIR}/sources/"
	)

	if(APPLE)
		set(Cinecoder.Plugin.Multiplexers_LIBRARY_DIRS
			"${Cinecoder.Plugin.Multiplexers_ROOT_DIR}/runtimes/osx-x64/native/"
		)
	elseif(UNIX AND (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64"))
		set(Cinecoder.Plugin.Multiplexers_LIBRARY_DIRS
			"${Cinecoder.Plugin.Multiplexers_ROOT_DIR}/runtimes/linux-arm64/native/"
		)
	elseif(UNIX)
		set(Cinecoder.Plugin.Multiplexers_LIBRARY_DIRS
			"${Cinecoder.Plugin.Multiplexers_ROOT_DIR}/runtimes/linux-x64/native/"
		)
	elseif(WIN32 AND CMAKE_CXX_SIZEOF_DATA_PTR EQUAL 8)
		set(Cinecoder.Plugin.Multiplexers_LIBRARY_DIRS
			"${Cinecoder.Plugin.Multiplexers_ROOT_DIR}/runtimes/win-x64/native/"
		)
	elseif(WIN32 AND CMAKE_CXX_SIZEOF_DATA_PTR EQUAL 4)
		set(Cinecoder.Plugin.Multiplexers_LIBRARY_DIRS
			"${Cinecoder.Plugin.Multiplexers_ROOT_DIR}/runtimes/win-x86/native/"
		)
	endif()

	find_library(Cinecoder.Plugin.Multiplexers_LIBRARIES NAME  Cinecoder.Plugin.Multiplexers
	                                                     PATHS ${Cinecoder.Plugin.Multiplexers_LIBRARY_DIRS}
	)

#	if(WIN32)
#		find_library(Cinecoder.Plugin.Multiplexers_MXFToolkitPortable_LIBRARIES NAME  MXFToolkitPortable
#	                                                                            PATHS ${Cinecoder.Plugin.Multiplexers_LIBRARY_DIRS}
#		)
#	else()
#		find_library(Cinecoder.Plugin.Multiplexers_MXFToolkitPortable_LIBRARIES NAME  MXFToolkit_portable
#																				PATHS ${Cinecoder.Plugin.Multiplexers_LIBRARY_DIRS}
#		)
#	endif()
endif()

unset(NPACKAGE_VERSION)
unset(NREPOSITORY_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Cinecoder.Plugin.Multiplexers REQUIRED_VARS   Cinecoder.Plugin.Multiplexers_INCLUDE_DIRS
                                                                                Cinecoder.Plugin.Multiplexers_LIBRARY_DIRS
                                                                                Cinecoder.Plugin.Multiplexers_LIBRARIES
                                                                                #Cinecoder.Plugin.Multiplexers_MXFToolkitPortable_LIBRARIES
                                                                VERSION_VAR     Cinecoder.Plugin.Multiplexers_VERSION_STRING
                                                                FAIL_MESSAGE    "Cinecoder.Plugin.Multiplexers package was not found!"
                                  )

mark_as_advanced(Cinecoder.Plugin.Multiplexers_ROOT_DIR
                 Cinecoder.Plugin.Multiplexers_VERSION_STRING
                 Cinecoder.Plugin.Multiplexers_INCLUDE_DIRS
                 Cinecoder.Plugin.Multiplexers_LIBRARY_DIRS
                )

