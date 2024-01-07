cmake_minimum_required(VERSION 3.8.0)
project(SvmLib VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()


set(PROJECT_NAME SvmLib)

################################################################################
# Source groups
################################################################################
set(Header_Files
	"denseLibSvm.h"
	"EnsembleListSvm.h"
	"EnsembleSvm.h"
	"EnumsTranslations.h"
	"ExtraTreeWrapper.h"
	"ISvm.h"
	"libSvmImplementation.h"
	"libSvmInternal.h"
	"OpenCvSvm.h"
	"SvmExceptions.h"
	"SvmFactory.h"
	"SvmSigmoid.h"
	"VotingEnsemble.h"
	"Feature.h"
)
source_group("Header Files" FILES ${Header_Files})

set(Source_Files
	"denseLibSvm.cpp"
	"EnsembleListSvm.cpp"
	"EnumsTranslations.cpp"
	"ExtraTreeWrapper.cpp"
	"libSvmImplementation.cpp"
	"libSvmInternal.cpp"
	"OpenCvSvm.cpp"
	"SvmExceptions.cpp"
	"SvmFactory.cpp"
	"SvmSigmoid.cpp"
)
source_group("Source Files" FILES ${Source_Files})

set(ALL_FILES
    ${Header_Files}
    ${Source_Files}
)

include_directories(../)
include_directories(../extern/gsl/include)

set(BOOST_ROOT "../extern/boost.1.66.0.0/lib/native/include")

find_package(Boost 1.66 REQUIRED)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
else()
  message(STATUS "Boost NOT Found !")
endif(Boost_FOUND)

################################################################################
# Target
################################################################################
add_library(${PROJECT_NAME} STATIC ${ALL_FILES})




target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

target_link_libraries(${PROJECT_NAME} PUBLIC
	libPlatform
	libDataset
)

add_compile_definitions(_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)

#use_package(${PROJECT_NAME} Microsoft.Gsl 0.1.2.2)#
