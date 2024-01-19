cmake_minimum_required(VERSION 3.17)
set(CMAKE_TOOLCHAIN_FILE ../vcpkg/scripts/buildsystems/vcpkg.cmake)
project(DeevaPythonModule LANGUAGES CXX)
set(PROJECT_NAME DeevaPythonModule)

set(PY_VERSION_SUFFIX "")
set(PY_FULL_VERSION ${PROJECT_VERSION}${PY_VERSION_SUFFIX})


# Make sure that the Python and CMake versions match
if (DEFINED PY_BUILD_CMAKE_PACKAGE_VERSION)
    if (NOT "${PY_BUILD_CMAKE_PACKAGE_VERSION}" MATCHES "^${PY_FULL_VERSION}$")
        message(FATAL_ERROR "Version number does not match "
                             "(${PY_BUILD_CMAKE_PACKAGE_VERSION} - ${PY_FULL_VERSION}).")
    endif()
endif()


################################################################################
# Source groups
################################################################################
set(no_group_source_files
    "pyproject.toml"
    "setup.cfg"
    "setup.py"
)
source_group("" FILES ${no_group_source_files})

set(Header_Files
    "DatasetLoader.h"
)
source_group("Header Files" FILES ${Header_Files})

set(Python
    "DeevaPackage/__init__.py"
    "DeevaPackage/classifiers.py"
    "DeevaPackage/data_loader.py"
)
source_group("Python" FILES ${Python})

set(Source_Files
    "DatasetLoader.cpp"
    "module.cpp"
)
source_group("Source Files" FILES ${Source_Files})

set(ALL_FILES
    ${no_group_source_files}
    ${Header_Files}
    ${Python}
    ${Source_Files}
)



set(OpenCV_DIR "../vcpkg/packages/opencv3_x64-windows/share/opencv3")
find_package(OpenCV REQUIRED)

include_directories(../)
include_directories(../extern/gsl/include)

set(BOOST_ROOT "../extern/boost.1.66.0.0/lib/native/include")

find_package(Boost 1.66 REQUIRED)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
else()
  message(STATUS "Boost NOT Found !")
endif(Boost_FOUND)

if( OpenCV_FOUND )
  # Additional Include Directories
  include_directories( ${OpenCV_INCLUDE_DIRS} )

  # Additional Library Directories
  link_directories( ${OpenCV_LIB_DIR} )

  # Additional Dependencies
  # target_link_libraries( ${PROJECT_NAME} )
endif()

#target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

message(STATUS "Python:")
message(STATUS "    Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")
message(STATUS "    Python_LIBRARIES: ${Python_LIBRARIES}")
message(STATUS "    Python_LIBRARY_DIRS: ${Python_LIBRARY_DIRS}")

set(pybind11_DIR "../vcpkg/packages/pybind11_x64-windows/share/pybind11")
find_package(pybind11 CONFIG REQUIRED)

# pybind11 method:
pybind11_add_module(${PROJECT_NAME} module.cpp DatasetLoader.cpp DatasetLoader.h)

# set_target_properties(${PROJECT_NAME}  PROPERTIES
#                       SOURCES  ${CMAKE_CURRENT_SOURCE_DIR}/DeevaPackage/__init__.py)

# set_target_properties(${PROJECT_NAME}  PROPERTIES
#                       SOURCES  ${CMAKE_CURRENT_SOURCE_DIR}/DeevaPackage/data_loader.py)



add_compile_definitions(_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)

# Python method:
# Python_add_library(${PROJECT_NAME} module.cpp)
# target_link_libraries(${PROJECT_NAME} pybind11::headers)
# set_target_properties(${PROJECT_NAME} PROPERTIES
#     INTERPROCEDURAL_OPTIMIZATION ON
#     CXX_VISIBILITY_PRESET ON
#     VISIBILITY_INLINES_HIDDEN ON
# )

# add_library(${PROJECT_NAME} MODULE module.cpp)
# target_link_libraries(${PROJECT_NAME} PUBLIC pybind11::module)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)






add_subdirectory(../libPlatform  ${CMAKE_CURRENT_BINARY_DIR}/libPlatform)
add_subdirectory(../libRandom ${CMAKE_CURRENT_BINARY_DIR}/libRandom)
add_subdirectory(../libDataset ${CMAKE_CURRENT_BINARY_DIR}/libDataset)
add_subdirectory(../libStrategies ${CMAKE_CURRENT_BINARY_DIR}/libStrategies)
add_subdirectory(../libGeneticComponents ${CMAKE_CURRENT_BINARY_DIR}/libGeneticComponents)

# add_subdirectory(../libGeneticStrategies ${CMAKE_CURRENT_BINARY_DIR}/libGeneticStrategies)
add_subdirectory(../libGeneticSvm ${CMAKE_CURRENT_BINARY_DIR}/libGeneticSvm)
add_subdirectory(../libSvmComponents ${CMAKE_CURRENT_BINARY_DIR}/libSvmComponents)
add_subdirectory(../libSvmStrategies ${CMAKE_CURRENT_BINARY_DIR}/libSvmStrategies)
add_subdirectory(../SvmLib ${CMAKE_CURRENT_BINARY_DIR}/SvmLib)


################################################################################
# Dependencies
################################################################################
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

# use_package(${PROJECT_NAME} Microsoft.Gsl 0.1.2.2)
# use_package(${PROJECT_NAME} boost 1.66.0.0)
# use_package(${PROJECT_NAME} OpenCV 3.3.1)
# use_package(${PROJECT_NAME} OpenCV-Ml 3.3.1)
# use_package(${PROJECT_NAME} OpenCV-Imgproc 3.3.1)
# use_package(${PROJECT_NAME} OpenCV-Imgcodecs 3.3.1)
# use_package(${PROJECT_NAME} OpenCV-Core 3.3.1)
# use_package(${PROJECT_NAME} OpenCV-Highgui 3.3.1)
# use_package(${PROJECT_NAME} boost_program_options-vc141 1.66.0.0)
