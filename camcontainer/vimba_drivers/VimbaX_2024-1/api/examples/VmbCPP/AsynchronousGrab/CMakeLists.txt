cmake_minimum_required(VERSION 3.0)

project(AsynchronousGrab LANGUAGES CXX)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/vmb_cmake_prefix_paths.cmake")
    # read hardcoded package location information, if the example is still located in the original install location
    include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/vmb_cmake_prefix_paths.cmake)
endif()

find_package(Vmb REQUIRED COMPONENTS CPP NAMES Vmb VmbC VmbCPP VmbImageTransform)

add_executable(AsynchronousGrab_VmbCPP
    main.cpp
    AcquisitionHelper.cpp
    AcquisitionHelper.h   
)
target_link_libraries(AsynchronousGrab_VmbCPP PRIVATE Vmb::CPP)

set_target_properties(AsynchronousGrab_VmbCPP PROPERTIES
    CXX_STANDARD 11
    VS_DEBUGGER_ENVIRONMENT "PATH=${VMB_BINARY_DIRS};$ENV{PATH}"
)
