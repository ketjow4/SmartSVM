cmake_minimum_required(VERSION 3.17)
project(TestApp LANGUAGES CXX)
set(PROJECT_NAME TestApp)

################################################################################
# Source groups
################################################################################
set(Headers
    "bin.h"
    "ConfigGeneration.h"
    "ConfigParser.h"
    "LastRegionsScores.h"
    "PythonPath.h"
    "ReRuns.h"
    "RunAlgorithm.h"
    "../AppUtils/AppUtils.h"
    "../AppUtils/framework.h"
    "../AppUtils/pch.h"
    "../AppUtils/PythonFeatureSelection.h"
    "../extern/cxxopts/cxxopts.hpp"
)
source_group("Headers" FILES ${Headers})

set(Headers__Experiments
    "ClashExperiment.h"
    "Commons.h"
    "CustomKernelExperiments.h"
    "Gecco2019.h"
    "ManualMetricTests.h"
    "ManualRbfLiniear.h"
)
source_group("Headers\\Experiments" FILES ${Headers__Experiments})


set(Source
    "ConfigGeneration.cpp"
    "main.cpp"
    "ReRuns.cpp"
    "RunAlgorithm.cpp"
    "../AppUtils/AppUtils.cpp"
    "../AppUtils/pch.cpp"
    "../AppUtils/PythonFeatureSelection.cpp"
)
source_group("Source" FILES ${Source})

set(ALL_FILES
    ${Headers}
    ${Headers__Experiments}
    ${Source}
)

include_directories(../)
include_directories(../extern)
include_directories(../extern/gsl/include)

################################################################################
# Target
################################################################################
add_executable(${PROJECT_NAME} ${ALL_FILES})


add_compile_definitions(_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)


# ONLY MSVC
set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "/INCREMENTAL")
#set_target_properties(${PROJECT_NAME} PROPERTIES "/MP")
add_compile_options(/MP)
set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "/debug:fastlink")

################################################################################
# Dependencies
################################################################################
add_subdirectory(../libPlatform  ${CMAKE_CURRENT_BINARY_DIR}/libPlatform)
add_subdirectory(../libRandom ${CMAKE_CURRENT_BINARY_DIR}/libRandom)
add_subdirectory(../libDataset ${CMAKE_CURRENT_BINARY_DIR}/libDataset)
add_subdirectory(../libStrategies ${CMAKE_CURRENT_BINARY_DIR}/libStrategies)
add_subdirectory(../libGeneticComponents ${CMAKE_CURRENT_BINARY_DIR}/libGeneticComponents)
add_subdirectory(../libGeneticSvm ${CMAKE_CURRENT_BINARY_DIR}/libGeneticSvm)
add_subdirectory(../libSvmComponents ${CMAKE_CURRENT_BINARY_DIR}/libSvmComponents)
add_subdirectory(../libSvmStrategies ${CMAKE_CURRENT_BINARY_DIR}/libSvmStrategies)
add_subdirectory(../SvmLib ${CMAKE_CURRENT_BINARY_DIR}/SvmLib)


# Link with other targets.
target_link_libraries(${PROJECT_NAME} PRIVATE
    libGeneticComponents
    libGeneticSvm
    libDataset
    libPlatform
    libRandom
    libStrategies
    libSvmComponents
    libSvmStrategies
    SvmLib
    )
