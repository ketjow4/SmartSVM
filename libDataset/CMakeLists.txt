cmake_minimum_required(VERSION 3.8.0)
project(libRandom VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()


set(PROJECT_NAME libDataset)

################################################################################
# Source groups
################################################################################
set(Header_Files
    "CsvReader.h"
    "Dataset.h"
    "DatasetBase.h"
    "DatasetExceptions.h"
    "DatasetView.h"
    "DatasetViewBase.h"
    "IDataset.h"
    "IReadOnlyDataset.h"
    "TabularDataExceptions.h"
    "TabularDataProvider.h"
)
source_group("Header Files" FILES ${Header_Files})

set(Source_Files
    "CsvReader.cpp"
    "DatasetExceptions.cpp"
    "TabularDataExceptions.cpp"
)
source_group("Source Files" FILES ${Source_Files})

set(ALL_FILES
    ${Header_Files}
    ${Source_Files}
)

include_directories(../)
include_directories(../extern/gsl/include)

################################################################################
# Target
################################################################################
add_library(${PROJECT_NAME} STATIC ${ALL_FILES})




target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

target_link_libraries(${PROJECT_NAME} PUBLIC
	libPlatform
)

add_compile_definitions(_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)
