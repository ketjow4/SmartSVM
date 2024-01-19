cmake_minimum_required(VERSION 3.16)
set(CMAKE_TOOLCHAIN_FILE ../vcpkg/scripts/buildsystems/vcpkg.cmake)
set(PROJECT_NAME libGeneticSvm)

################################################################################
# Source groups
################################################################################
file(GLOB ALL_FILES CONFIGURE_DEPENDS "*.h" "*.cpp")



set(OpenCV_DIR "../vcpkg/packages/opencv3_x64-windows/share/opencv3")
find_package(OpenCV REQUIRED)

if( OpenCV_FOUND )
  # Additional Include Directories
  include_directories( ${OpenCV_INCLUDE_DIRS} )

  # Additional Library Directories
  link_directories( ${OpenCV_LIB_DIR} )

  # Additional Dependencies
  # target_link_libraries( ${PROJECT_NAME} )
endif()

include_directories(../)
include_directories(../extern/gsl/include)

set(BOOST_ROOT "../extern/boost.1.66.0.0/lib/native/include")

find_package(Boost 1.66 REQUIRED)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
else()
  message(STATUS "Boost NOT Found !")
endif(Boost_FOUND)


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

#use_package(${PROJECT_NAME} OpenCV 3.3.1)
#use_package(${PROJECT_NAME} Microsoft.Gsl 0.1.2.2)
#use_package(${PROJECT_NAME} boost 1.66.0.0)
