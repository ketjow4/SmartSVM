cmake_minimum_required(VERSION 3.8.0)
project(libRandom VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()


set(PROJECT_NAME libRandom)

set(Header_Files
  "MersenneTwister64Rng.h"
  "IRandomNumberGenerator.h"
  "RandomExceptions.h"
  "RandomNumberGeneratorFactory.h"

)
source_group("Header Files" FILES ${Header_Files})

set(Source_Files
  "MersenneTwister64Rng.cpp"
  "RandomExceptions.cpp"
  "RandomNumberGeneratorFactory.cpp"

)
source_group("Source Files" FILES ${Source_Files})

set(ALL_FILES
    ${Header_Files}
    ${Source_Files}
)

include_directories(../)

set(BOOST_ROOT "../extern/boost.1.66.0.0/lib/native/include")

find_package(Boost 1.66 REQUIRED)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
else()
  message(STATUS "Boost NOT Found !")
endif(Boost_FOUND)

add_library(${PROJECT_NAME} STATIC ${ALL_FILES})

#add_dependencies (libRandom libPlatform)

# Assuming libPlatform is also built as a static library
# add_library(libPlatform STATIC IMPORTED)
# set_target_properties(libPlatform PROPERTIES IMPORTED_LOCATION "../libPlatform/build/Debug/libPlatform.lib") # Adjust the path accordingly


# Link libRandom with libPlatform
target_link_libraries(${PROJECT_NAME} libPlatform)



target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

