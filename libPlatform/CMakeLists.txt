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
)
source_group("Header Files" FILES ${Header_Files})

set(Source_Files
    "Percent.cpp"
    "StringUtils.cpp"
    "Subprocess.cpp"
    "Subtree.cpp"
    "PlatformException.cpp"
)
source_group("Source Files" FILES ${Source_Files})

set(ALL_FILES
    ${Header_Files}
    ${Source_Files}
)

# include(../cmake/CPM.cmake)
# CPMAddPackage(
#   NAME Boost
#   VERSION 1.66.0
#   GITHUB_REPOSITORY "boostorg/boost"
#   GIT_TAG "boost-1.66.0"
# )

# find_package(Boost 1.82.0 REQUIRED)
# if(Boost_FOUND)
#   target_include_directories(${PROJECT_NAME} PRIVATE ${Boost_INCLUDE_DIRS})
# endif(Boost_FOUND)

# set(BOOST_ROOT "../extern/")

set(BOOST_ROOT "../extern/boost.1.66.0.0/lib/native/include")

find_package(Boost 1.66 REQUIRED)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
else()
  message(STATUS "Boost NOT Found !")
endif(Boost_FOUND)

add_library(${PROJECT_NAME} STATIC ${ALL_FILES})

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

