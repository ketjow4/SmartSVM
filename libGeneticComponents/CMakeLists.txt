cmake_minimum_required(VERSION 3.16)
project(SvmLib VERSION 0.1.0 LANGUAGES C CXX)

set(PROJECT_NAME libGeneticComponents)

################################################################################
# Source groups
################################################################################
set(Header_Files
    "BaseChromosome.h"
    "BaseCrossoverOperator.h"
    "GeneticExceptions.h"
    "GeneticUtils.h"
    "HighLowFitSelection.h"
    "LocalGlobalAdaptationSelection.h"
    "Population.h"
    "resource.h"
    "TruncationSelection.h"
)
source_group("Header Files" FILES ${Header_Files})

set(Header_Files__BinaryChromosomeOperator
    "BinaryChromosomeCache.h"
    "BitFlipMutation.h"
    "FeaturesSelectionBitFlipMutation.h"
    "FeaturesSelectionOnePointCrossover.h"
    "FeaturesSelectionRandomGeneration.h"
    "OnePointCrossover.h"
    "RandomGeneration.h"
)
source_group("Header Files\\BinaryChromosomeOperator" FILES ${Header_Files__BinaryChromosomeOperator})

set(Header_Files__Factories
    "BinaryCrossoverFactory.h"
    "BinaryGenerationFactory.h"
    "BinaryMutationFactory.h"
    "CrossoverSelectionFactory.h"
    "SelectionFactory.h"
    "StopConditionFactory.h"
)
source_group("Header Files\\Factories" FILES ${Header_Files__Factories})

set(Header_Files__Interfaces
    "BinaryChromosome.h"
    "IChromosomeCache.h"
    "ICrossoverSelection.h"
    "IMutationOperator.h"
    "IOperator.h"
    "IPopulationGeneration.h"
    "ISelectionOperator.h"
    "IStopCondition.h"
)
source_group("Header Files\\Interfaces" FILES ${Header_Files__Interfaces})

set(Header_Files__StopConditions
    "BestFitnessStopCondition.h"
    "ConsecutiveIterationFitnessBase.h"
    "MeanFitnessStopCondition.h"
)
source_group("Header Files\\StopConditions" FILES ${Header_Files__StopConditions})

set(Source_Files
    "BaseChromosome.cpp"
    "FactoriesTranslations.cpp"
    "GeneticExceptions.cpp"
)
source_group("Source Files" FILES ${Source_Files})

set(ALL_FILES
    ${Header_Files}
    ${Header_Files__BinaryChromosomeOperator}
    ${Header_Files__Factories}
    ${Header_Files__Interfaces}
    ${Header_Files__StopConditions}
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




add_library(${PROJECT_NAME} STATIC ${ALL_FILES})

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

target_link_libraries(${PROJECT_NAME} PUBLIC
	libPlatform
)

add_compile_definitions(_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)

#use_package(${PROJECT_NAME} Microsoft.Gsl 0.1.2.2)
#use_package(${PROJECT_NAME} boost 1.66.0.0)
