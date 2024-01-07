cmake_minimum_required(VERSION 3.8.0)
project(SmartSVM VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

set(PROJECT_NAME SmartSVM)

add_subdirectory(libPlatform)
add_subdirectory(libRandom)
add_subdirectory(libDataset)
add_subdirectory(libStrategies)


include_directories(./extern/gsl/include)
set(BOOST_ROOT "./extern/boost.1.66.0.0/lib/native/include")

find_package(Boost 1.66 REQUIRED)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
else()
  message(STATUS "Boost NOT Found !")
endif(Boost_FOUND)




# set_target_properties(libPlatform PROPERTIES IMPORTED_LOCATION "./libPlatform/build/Debug/libPlatform.lib") # Adjust the path accordingly
# set_target_properties(libRandom PROPERTIES IMPORTED_LOCATION "./libRandom/build/Debug/libRandom.lib") # Adjust the path accordingly


add_executable(${PROJECT_NAME} main.cpp)


target_link_libraries(${PROJECT_NAME} PRIVATE libPlatform)
target_link_libraries(${PROJECT_NAME} PRIVATE libRandom)
target_link_libraries(${PROJECT_NAME} PRIVATE libDataset)
target_link_libraries(${PROJECT_NAME} PRIVATE libStrategies)


target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)


add_compile_definitions(_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)
