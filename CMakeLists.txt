cmake_minimum_required(VERSION 3.8.0)
project(SmartSVM VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

add_subdirectory(libPlatform)

set(BOOST_ROOT "./extern/boost.1.66.0.0/lib/native/include")

find_package(Boost 1.66 REQUIRED)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
else()
  message(STATUS "Boost NOT Found !")
endif(Boost_FOUND)


target_link_libraries(libPlatform)

add_executable(SmartSVM main.cpp)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)


  
add_compile_options(_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)