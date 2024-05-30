cmake_minimum_required(VERSION 3.17)

project(DeevaPythonPackage LANGUAGES CXX)
set(PROJECT_NAME DeevaPythonPackage)

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


# Use the currently active Conda environment
# list(APPEND CMAKE_PREFIX_PATH "${PYTHON_EXECUTABLE}")

# set(CMAKE_PREFIX_PATH "C:/Users/Wojtek/Anaconda3/envs/Deeva_python_310")
# set(PYTHON_EXECUTABLE "C:/Users/Wojtek/Anaconda3/envs/Deeva_python_310/python.exe")

# set(Python_FIND_VIRTUALENV ONLY)

# message(STATUS "Before find Python: ${PYTHON_EXECUTABLE}")

find_package(Python COMPONENTS Interpreter Development REQUIRED)

# message(STATUS "After find Python:  ${PYTHON_EXECUTABLE}")

message(STATUS "Python:")
message(STATUS "    Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")
message(STATUS "    Python_LIBRARIES: ${Python_LIBRARIES}")
message(STATUS "    Python_LIBRARY_DIRS: ${Python_LIBRARY_DIRS}")

# set(pybind11_DIR "./extern/pybind11")
# find_package(pybind11 CONFIG REQUIRED)
add_subdirectory("./extern/pybind11")

# pybind11 method:
pybind11_add_module(${PROJECT_NAME} MODULE module.cpp DatasetLoader.cpp)


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

# install(TARGETS ${PROJECT_NAME}
#         COMPONENT python
#         LIBRARY DESTINATION "lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
#         ARCHIVE DESTINATION "lib"
#         RUNTIME DESTINATION "bin")

# install(TARGETS ${PROJECT_NAME}
#   COMPONENT python
#   LIBRARY DESTINATION "${PYTHON_LIBRARY_DIR}"
#   )



# Check if VERSION_INFO is set, and if so, define it as a compile definition
if(DEFINED VERSION_INFO)
    target_compile_definitions(${PROJECT_NAME} PRIVATE VERSION_INFO=\"${VERSION_INFO}\")
else()
    message(WARNING "VERSION_INFO is not defined. Using default value.")
    target_compile_definitions(${PROJECT_NAME} PRIVATE VERSION_INFO=\"dev\")
endif()




# Use stubgen to create .pyi files to sit alongside the just-built python module
# set(Stubgen_Executable "stubgen")
# if(WIN32)
#     set(Stubgen_Executable "${CMAKE_SOURCE_DIR}/venv/Scripts/stubgen.exe")
# endif()
# add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
#     COMMAND ${Stubgen_Executable} -m ${PROJECT_NAME} -o .
#     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#     COMMENT "Use stubgen to create .pyi for statement completion")



# add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
#     COMMAND pybind11-stubgen ${PROJECT_NAME} -o DeevaPythonPackage-stubs
#     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#     COMMENT "Use pybind11-stubgen to create .pyi for statement completion")

     