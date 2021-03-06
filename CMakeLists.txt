cmake_minimum_required(VERSION 3.9)

##############################
###        googletest      ###
##############################
SET(BUILD_GMOCK ON CACHE INTERNAL "")
SET(BUILD_GTEST ON CACHE INTERNAL "")
SET(INSTALL_GMOCK OFF CACHE INTERNAL "")
SET(INSTALL_GTEST OFF CACHE INTERNAL "")
SET(gmock_build_tests OFF CACHE INTERNAL "")
SET(gtest_build_samples OFF CACHE INTERNAL "")
SET(gtest_build_tests OFF CACHE INTERNAL "")
SET(gtest_disable_pthreads OFF CACHE INTERNAL "")
SET(gtest_hide_internal_symbols OFF CACHE INTERNAL "")
SET(gtest_force_shared_crt OFF CACHE INTERNAL "")

add_subdirectory(googletest)

##############################
###        benchmarks      ###
##############################
SET(BENCHMARK_ENABLE_INSTALL OFF CACHE INTERNAL "")
SET(BENCHMARK_ENABLE_RELEASE OFF CACHE INTERNAL "")
SET(BENCHMARK_ENABLE_EXCEPTIONS OFF CACHE INTERNAL "")
SET(BENCHMARK_ENABLE_TESTING OFF CACHE INTERNAL "")
SET(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE INTERNAL "")

add_subdirectory(benchmark)

##############################
###        Project         ###
##############################

project(rlc VERSION 1.0 LANGUAGES CXX)
enable_testing()
include("CPack")
include(CTest)
set(CMAKE_DEBUG_POSTFIX "-d")

##############################
###        Globals         ###
##############################
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable -Wno-unused-parameter")

set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

set_tests_properties(${noArgsTests}   PROPERTIES TIMEOUT 10)
include(macros.cmake)

find_package(LLVM REQUIRED CONFIG)
add_definitions(${LLVM_DEFINITIONS})
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include_directories(${LLVM_INCLUDE_DIRS})

##############################
###       Subfolders       ###
##############################
add_subdirectory(lib)
add_subdirectory(tool)

##############################
###        DOXYGEN         ###
##############################
find_package(Doxygen)
if (DOXYGEN_FOUND)
    # set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    # note the option ALL which allows to build the docs together with the application
    add_custom_target(doc 
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )
else (DOXYGEN_FOUND)
  message("Doxygen need to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)

##############################
###        COVERAGE        ###
##############################
if (${CMAKE_CXX_FLAGS} MATCHES ".*--coverage.*")
	find_program(LCOV NAMES lcov)
	find_program(GENHTML NAMES genhtml)

	add_custom_command(
		OUTPUT base.info
		COMMAND ${LCOV} --capture --initial --directory . --output-file ./base.info
		VERBATIM
		)

	add_custom_command(
		OUTPUT filtered.info 
		COMMAND ${LCOV} --remove ./total.info "/usr/*" "*/googletest/*" "*/build/*" "*/benchmark/*" --output-file ./filtered.info
		DEPENDS total.info
		VERBATIM)

	add_custom_command(
		OUTPUT total.info 
		COMMAND ${LCOV} --add-tracefile ./base.info --add-tracefile ./test.info --output-file ./total.info
		DEPENDS test.info base.info
		VERBATIM)

	add_custom_command(
		OUTPUT test.info 
		COMMAND ${LCOV} --capture --directory . --output-file ./test.info
		VERBATIM)

	add_custom_target(genCoverage
		COMMAND ${GENHTML} ./filtered.info --output-directory ./Coverage
		DEPENDS filtered.info
		BYPRODUCTS ./Coverage
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
		COMMENT "Generate coverage"
		VERBATIM )

	add_custom_target(coverage COMMAND rm test.info filtered.info base.info total.info)
	add_dependencies(coverage genCoverage)
	add_dependencies(coverage test)
elseif(TRUE)
	add_custom_target(coverage
		COMMAND echo could not calculate coverage run cmake with -DCMAKE_CXX_FLAGS=--coverage
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
		COMMENT "Coverage not aviable"
		VERBATIM)
endif()


