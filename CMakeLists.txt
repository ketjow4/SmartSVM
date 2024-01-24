cmake_minimum_required(VERSION 3.17)
set(CMAKE_TOOLCHAIN_FILE ./vcpkg/scripts/buildsystems/vcpkg.cmake)
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


include_directories(./)
include_directories(./extern)
include_directories(./extern/gsl/include)


find_package(Python COMPONENTS Interpreter Development REQUIRED)

message(STATUS "Python:")
message(STATUS "    Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")
message(STATUS "    Python_LIBRARIES: ${Python_LIBRARIES}")
message(STATUS "    Python_LIBRARY_DIRS: ${Python_LIBRARY_DIRS}")

set(pybind11_DIR "./vcpkg/packages/pybind11_x64-windows/share/pybind11")
find_package(pybind11 CONFIG REQUIRED)

# pybind11 method:
pybind11_add_module(${PROJECT_NAME} module.cpp DatasetLoader.cpp DatasetLoader.h)


add_compile_definitions(_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)



add_subdirectory(./libPlatform  ${CMAKE_CURRENT_BINARY_DIR}/libPlatform)
add_subdirectory(./libRandom ${CMAKE_CURRENT_BINARY_DIR}/libRandom)
add_subdirectory(./libDataset ${CMAKE_CURRENT_BINARY_DIR}/libDataset)
add_subdirectory(./libStrategies ${CMAKE_CURRENT_BINARY_DIR}/libStrategies)
add_subdirectory(./libGeneticComponents ${CMAKE_CURRENT_BINARY_DIR}/libGeneticComponents)
add_subdirectory(./libGeneticSvm ${CMAKE_CURRENT_BINARY_DIR}/libGeneticSvm)
add_subdirectory(./libSvmComponents ${CMAKE_CURRENT_BINARY_DIR}/libSvmComponents)
add_subdirectory(./libSvmStrategies ${CMAKE_CURRENT_BINARY_DIR}/libSvmStrategies)
add_subdirectory(./SvmLib ${CMAKE_CURRENT_BINARY_DIR}/SvmLib)


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