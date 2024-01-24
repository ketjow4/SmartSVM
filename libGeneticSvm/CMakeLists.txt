cmake_minimum_required(VERSION 3.16)
set(CMAKE_TOOLCHAIN_FILE ../vcpkg/scripts/buildsystems/vcpkg.cmake)
set(PROJECT_NAME libGeneticSvm)

################################################################################
# Source groups
################################################################################
file(GLOB ALL_FILES CONFIGURE_DEPENDS "*.h" "*.cpp")


include_directories(../)
include_directories(../extern/gsl/include)


add_library(${PROJECT_NAME} STATIC ${ALL_FILES})


target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

add_compile_definitions(_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)


################################################################################
# Dependencies
################################################################################
# Link with other targets.
target_link_libraries(${PROJECT_NAME} PUBLIC
    libDataset
    libPlatform
    libRandom
    libStrategies
    libSvmComponents
    libSvmStrategies
    SvmLib
    libGeneticComponents
)