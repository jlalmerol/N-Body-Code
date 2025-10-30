cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_COMPILER mpicxx)
set(CMAKE_C_COMPILER mpicc)

project(NBODY VERSION 1.0 LANGUAGES CXX C)

set(NBODY_BUILD_MODE_OPTIONS
    "DEFAULT"
    "_F32"
    "_WH_MULTIHOST"
    "_WH_MULTICHIP"
    "_WH_MESH"
)
set(NBODY_BUILD_MODE "DEFAULT" CACHE STRING "Select the desired build flavor (e.g., DEFAULT, _F32, _WH_MULTIHOST, _WH_MULTICHIP, _WH_MESH).")
set_property(CACHE NBODY_BUILD_MODE PROPERTY STRINGS ${NBODY_BUILD_MODE_OPTIONS})

if ("${NBODY_BUILD_MODE}" STREQUAL "_WH_MULTIHOST" OR "${NBODY_BUILD_MODE}" STREQUAL "_WH_MULTICHIP" OR "${NBODY_BUILD_MODE}" STREQUAL "_WH_MESH")
    set(CMAKE_CXX_STANDARD 20)
else()
    set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(nbody_app
    src/main.cpp
    src/nbody.cpp
    src/utils.cpp
    src/kernel.cpp
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(MPI REQUIRED)
if (MPI_FOUND)
    message(STATUS "MPI found: ${MPI_CXX_COMPILER}")
    target_include_directories(nbody_app PRIVATE ${MPI_CXX_INCLUDE_DIRS})
    target_link_libraries(nbody_app PRIVATE ${MPI_CXX_LIBRARIES})
else()
    message(FATAL_ERROR "MPI not found. Please ensure MPI is installed and configured correctly.")
endif()

option(USE_OPENMP "Enable OpenMP for the force calculation kernel" ON)
if (USE_OPENMP)
    find_package(OpenMP REQUIRED)
    if (OpenMP_FOUND)
        message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
        target_compile_options(nbody_app PRIVATE ${OpenMP_CXX_FLAGS})
        target_link_libraries(nbody_app PRIVATE ${OpenMP_CXX_LIBRARIES})
    else()
        message(FATAL_ERROR "OpenMP requested but not found. Please install OpenMP libraries.")
    endif()
endif()

find_library(LIBNUMA numa REQUIRED)
if (LIBNUMA)
    message(STATUS "libnuma found: ${LIBNUMA}")
    target_link_libraries(nbody_app PRIVATE ${LIBNUMA})
else()
    message(FATAL_ERROR "libnuma not found. Please install libnuma-dev.")
endif()

if ("${NBODY_BUILD_MODE}" STREQUAL "_WH_MULTIHOST" OR "${NBODY_BUILD_MODE}" STREQUAL "_WH_MULTICHIP" OR "${NBODY_BUILD_MODE}" STREQUAL "_WH_MESH")
    set(ARCH_NAME "wormhole_b0")
endif()

if ("${NBODY_BUILD_MODE}" STREQUAL "_WH_MULTIHOST")
    add_compile_definitions(nbody_app "_WH_MULTIHOST")
elseif ("${NBODY_BUILD_MODE}" STREQUAL "_WH_MULTICHIP")
    add_compile_definitions(nbody_app "_WH_MULTICHIP")
elseif ("${NBODY_BUILD_MODE}" STREQUAL "_WH_MESH")
    add_compile_definitions(nbody_app "_WH_MESH")
elseif ("${NBODY_BUILD_MODE}" STREQUAL "_F32")
    add_compile_definitions(nbody_app "_F32")
endif()
    
target_compile_options(nbody_app PRIVATE -O3)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(nbody_app PRIVATE _DEBUG)
endif()

target_include_directories(nbody_app PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_options(nbody_app PRIVATE -Wno-deprecated-declarations)

if ("${NBODY_BUILD_MODE}" STREQUAL "_WH" 
 OR "${NBODY_BUILD_MODE}" STREQUAL "_GS"
 OR "${NBODY_BUILD_MODE}" STREQUAL "_WH_MULTIHOST"
 OR "${NBODY_BUILD_MODE}" STREQUAL "_WH_MULTICHIP"
 OR "${NBODY_BUILD_MODE}" STREQUAL "_WH_MESH")
    set(DEVICE_SPECIFIC_LIBS "")

    add_subdirectory(../ tt-metal-build)

    target_link_libraries(
      nbody_app
      PRIVATE
        Metalium::Metal
        TT::Metalium::HostDevCommon
        TT::Metalium::Common
        ${DEVICE_SPECIFIC_LIBS}
    )

    set(TT_METAL_INCLUDE_DIRS
    	${CMAKE_CURRENT_SOURCE_DIR}/../tt_metal/api
	${CMAKE_CURRENT_SOURCE_DIR}/../tt_metal/api/tt-metalium
    	${CMAKE_CURRENT_SOURCE_DIR}/../tt_metal
    	${CMAKE_CURRENT_SOURCE_DIR}/../tt_metal/third_party/umd
	${CMAKE_CURRENT_SOURCE_DIR}/../tt_metal/third_party/umd/common
    	${CMAKE_CURRENT_SOURCE_DIR}/../tt_metal/hw/inc
    	${CMAKE_CURRENT_SOURCE_DIR}/../tt_metal/third_party/umd/src/firmware/riscv
    	${CMAKE_CURRENT_SOURCE_DIR}/../tt_metal/impl
    	${CMAKE_CURRENT_SOURCE_DIR}/../tt_metal/impl/device
	${CMAKE_CURRENT_SOURCE_DIR}/../tt_metal/impl/context
    	${CMAKE_CURRENT_SOURCE_DIR}/../tt_metal/common
    )

    if (NOT "${ARCH_NAME}" STREQUAL "")
        list(APPEND TT_METAL_INCLUDE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/../tt-metal/tt_metal/hw/inc/${ARCH_NAME}
            ${CMAKE_CURRENT_SOURCE_DIR}/../tt-metal/tt_metal/third_party/umd/src/firmware/riscv/${ARCH_NAME}
        )
    endif()

    target_include_directories(nbody_app PRIVATE
      ${TT_METAL_INCLUDE_DIRS}
    )    

    target_sources(nbody_app PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../tt_metal/tools/mem_bench/host_utils.cpp
    )	


    # --- Add profiler tools support ---
    set(PROFILER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../tt-metal/tt_metal/tools/profiler)
    target_include_directories(nbody_app PRIVATE ${PROFILER_DIR})
    file(GLOB PROFILER_SRC ${PROFILER_DIR}/*.cpp ${PROFILER_DIR}/*.cc)
    if(PROFILER_SRC)
        target_sources(nbody_app PRIVATE ${PROFILER_SRC})
    endif()
endif()

