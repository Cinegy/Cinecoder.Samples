function(GetNPackageVersion NPACKAGE_NAME)
	if(EXISTS "${PROJECT_SOURCE_DIR}/packages.config")
		file(STRINGS "${PROJECT_SOURCE_DIR}/packages.config" PACKAGES REGEX ".+${NPACKAGE_NAME}\".+")
	endif()
	if(PACKAGES)
		list(GET PACKAGES 0 ITEM)
		string(REGEX MATCH ".+([0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+-*[a-z]*).+" SMATCH ${ITEM})
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
GetNPackageVersion("Cinecoder")

if(NREPOSITORY_PATH AND NPACKAGE_VERSION)
	set(Cinecoder_VERSION_STRING "${NPACKAGE_VERSION}")
	set(Cinecoder_ROOT_DIR "${CMAKE_SOURCE_DIR}/${NREPOSITORY_PATH}/Cinecoder.${NPACKAGE_VERSION}")
	set(Cinecoder_INCLUDE_DIRS
		"${Cinecoder_ROOT_DIR}/sources/"
		"${Cinecoder_ROOT_DIR}/sources/winclude/"
	)

	if(APPLE)
		set(Cinecoder_LIBRARY_DIRS
			"${Cinecoder_ROOT_DIR}/runtimes/osx-x64/native/release/"
		)
	elseif(UNIX)
		set(Cinecoder_LIBRARY_DIRS
			"${Cinecoder_ROOT_DIR}/runtimes/linux-x64/native/release/"
		)
	endif()

	find_library(Cinecoder_LIBRARIES NAME  Cinecoder
	                                 PATHS ${Cinecoder_LIBRARY_DIRS}
	)

	find_library(Cinecoder_D2cudalib_LIBRARY NAME  d2cudalib
	                                         PATHS ${Cinecoder_LIBRARY_DIRS}
	)
endif()

unset(NPACKAGE_VERSION)
unset(NREPOSITORY_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Cinecoder REQUIRED_VARS   Cinecoder_INCLUDE_DIRS
                                                            Cinecoder_LIBRARY_DIRS
                                                            Cinecoder_LIBRARIES
                                            VERSION_VAR     Cinecoder_VERSION_STRING
                                            FAIL_MESSAGE    "Cinecoder package was not found!"
                                  )

mark_as_advanced(Cinecoder_ROOT_DIR
                 Cinecoder_VERSION_STRING
                 Cinecoder_INCLUDE_DIRS
                 Cinecoder_LIBRARIY_DIRS
                 Cinecoder_LIBRARY_DIRS
                 Cinecoder_D2cudalib_LIBRARY
                )
