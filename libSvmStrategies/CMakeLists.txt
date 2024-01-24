cmake_minimum_required(VERSION 3.16.0)
set(CMAKE_TOOLCHAIN_FILE ../vcpkg/scripts/buildsystems/vcpkg.cmake)
project(libSvmStrategies VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()


set(PROJECT_NAME libSvmStrategies)

################################################################################
# Source groups
################################################################################
file(GLOB ALL_FILES CONFIGURE_DEPENDS "*.h" "*.cpp")


include_directories(../)
include_directories(../extern/gsl/include)


################################################################################
# Target
################################################################################
add_library(${PROJECT_NAME} STATIC ${ALL_FILES})

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

target_link_libraries(${PROJECT_NAME} PUBLIC
	libPlatform
	libDataset
	SvmLib
  libSvmComponents
  ${OpenCV_LIBS}
)

add_compile_definitions(_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)

#use_package(${PROJECT_NAME} Microsoft.Gsl 0.1.2.2)#
