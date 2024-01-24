cmake_minimum_required(VERSION 3.8.0)
project(libPlatform VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

#//https://stackoverflow.com/questions/13703647/how-to-properly-add-include-directories-with-cmake

set(PROJECT_NAME libPlatform)

set(Header_Files
    "EnumStringConversions.h"
    "Percent.h"
    "StringUtils.h"
    "Subprocess.h"
    "SubprocessExceptions.h"
    "Subtree.h"
    "SubtreeExceptions.h"
    "PlatformException.h"
    "TimeUtils.h"
    "InterfaceDeclaration.h"
    "loguru.hpp"
)
source_group("Header Files" FILES ${Header_Files})

set(Source_Files
    "Percent.cpp"
    "StringUtils.cpp"
    "Subprocess.cpp"
    "Subtree.cpp"
    "PlatformException.cpp"
    "TimeUtils.cpp"
    "loguru.cpp"
)
source_group("Source Files" FILES ${Source_Files})

set(ALL_FILES
    ${Header_Files}
    ${Source_Files}
)


include_directories(../extern)


add_library(${PROJECT_NAME} STATIC ${ALL_FILES})

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

