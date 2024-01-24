cmake_minimum_required(VERSION 3.8.0)
project(libStrategies VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()


set(PROJECT_NAME libStrategies)

################################################################################
# Source groups
################################################################################
set(Header_Files
  "FileSinkStrategy.h"
  "StrategiesExceptions.h"
  "DiskFile.h"
)
source_group("Header Files" FILES ${Header_Files})

set(Source_Files
  "FileSinkStrategy.cpp"
  "DiskFile.cpp"
)
source_group("Source Files" FILES ${Source_Files})

set(ALL_FILES
    ${Header_Files}
    ${Source_Files}
)

include_directories(../)
include_directories(../extern/gsl/include)

################################################################################
# Target
################################################################################
add_library(${PROJECT_NAME} STATIC ${ALL_FILES})




target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

target_link_libraries(${PROJECT_NAME} PUBLIC
	libPlatform
)

add_compile_definitions(_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)

#use_package(${PROJECT_NAME} Microsoft.Gsl 0.1.2.2)#
