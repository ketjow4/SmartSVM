cmake_minimum_required(VERSION 3.8.0)
set(CMAKE_TOOLCHAIN_FILE ../vcpkg/scripts/buildsystems/vcpkg.cmake)
project(libSvmComponents VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()


set(PROJECT_NAME libSvmComponents)

################################################################################
# Source groups
################################################################################
set(Header_Files
"AddingTrainingSetExamples.h"
"BaseKernelGridSearch.h"
"BaseSvmChromosome.h"
"CompensationInformation.h"
"ConfusionMatrix.h"
"ConfusionMatrixMetrics.h"
"CrossoverCompensation.h"
"CustomKernelTraining.h"
"CustomWidthGauss.h"
"DataNormalization.h"
"EducationOfTrainingSet.h"
"Feature.h"
"FoldCreator.h"
"GaSvmCrossover.h"
"GaSvmGeneration.h"
"GaSvmMutation.h"
"GaSvmRegression.h"
"GeneticAlgorithmsConfigs.h"
"GridSearchCrossValidation.h"
"HeuristicCrossover.h"
"IFeatureSelection.h"
"IGroupPropagation.h"
"ISupportVectorSelection.h"
"ISvmMetricsCalculator.h"
"ISvmTraining.h"
"LinearKernel.h"
"MemeticCrossover.h"
"MemeticFeatureCompensation.h"
"MemeticFeatureCrossover.h"
"MemeticFeatureMutation.h"
"MemeticFeaturesAdaptation.h"
"MemeticFeaturesCompensationGeneration.h"
"MemeticFeaturesEducation.h"
"MemeticFeaturesFactories.h"
"MemeticFeaturesPool.h"
"MemeticFeaturesSuperIndividualsGeneration.h"
"MemeticMutialInfoRoulleteWheelGeneration.h"
"MemeticTraningSetAdaptation.h"
"Metric.h"
"MultipleGammaInternals.h"
"MutationKernelParameters.h"
"PolyKernel.h"
"R2_regression.h"
"RandomMemeticFeaturesGeneration.h"
"RbfKernel.h"
"RbfLinearInternals.h"
"RbfPolyGlobalKernel.h"
"resource.h"
"SequentialGammaInternals.h"
"SmallerPoolExperiment.h"
"SuperIndividualsCreation.h"
"SupportVectorPool.h"
"SvmAccuracyMetric.h"
"SvmAucMetric.h"
"SvmAucprcMetric.h"
"SvmComponentsExceptions.h"
"SvmConfigStructures.h"
"SvmCrossoverFactory.h"
"SvmCustomKernelChromosome.h"
"SvmCustomKernelFeaturesSelectionChromosome.h"
"SvmFeatureSetChromosome.h"
"SvmFeatureSetMemeticChromosome.h"
"SvmGenerationPopulationFactory.h"
"SvmHyperplaneDistance.h"
"SvmKernelChromosome.h"
"SvmKernelGridGeneration.h"
"SvmKernelRandomGeneration.h"
"SvmKernelTraining.h"
"SvmMemeticFeatureSetPopulationFactory.h"
"SvmMetricFactory.h"
"SvmMutationFactory.h"
"SvmPopulationStatistics.h"
"SvmSimultaneousChromosome.h"
"SvmSubsetValidation.h"
"SvmTraining.h"
"SvmTrainingSetChromosome.h"
"SvmTrainingSetCrossoverFactory.h"
"SvmTrainingSetMutationFactory.h"
"SvmTrainingSetPopulationFactory.h"
"SvmUtils.h"
"SvmValidationFactory.h"
"SvmValidationStrategy.h"
"SvmVisualization.h"
"TraningSetCrossover.h"
"ValidationSelectionMethodFactory.h"
)
source_group("Header Files" FILES ${Header_Files})
	
set(Source_Files
"BaseKernelGridSearch.cpp"
"BaseSvmChromosome.cpp"
"CMakeLists.txt"
"CompensationInformation.cpp"
"ConfusionMatrix.cpp"
"CrossoverCompensation.cpp"
"DataNormalization.cpp"
"EducationOfTrainingSet.cpp"
"FoldCreator.cpp"
"GaSvmCrossover.cpp"
"GaSvmGeneration.cpp"
"GaSvmMutation.cpp"
"GeneticAlgorithmsConfigs.cpp"
"GridSearchCrossValidation.cpp"
"HeuristicCrossover.cpp"
"IGroupPropagation.cpp"
"LinearKernel.cpp"
"MemeticCrossover.cpp"
"MemeticFeatureCompensation.cpp"
"MemeticFeaturesCompensationGeneration.cpp"
"MemeticFeaturesFactories.cpp"
"MemeticTraningSetAdaptation.cpp"
"Metric.cpp"
"MultipleGammaInternals.cpp"
"MutationKernelParameters.cpp"
"PolyKernel.cpp"
"R2_regression.cpp"
"RbfKernel.cpp"
"RbfLinearInternals.cpp"
"RbfPolyGlobalKernel.cpp"
"SequentialGammaInternals.cpp"
"SuperIndividualsCreation.cpp"
"SupportVectorPool.cpp"
"SvmAccuracyMetric.cpp"
"SvmAucMetric.cpp"
"SvmAucprcMetric.cpp"
"SvmComponentsExceptions.cpp"
"SvmConfigStructures.cpp"
"SvmCrossoverFactory.cpp"
"SvmFeatureSetChromosome.cpp"
"SvmFeatureSetMemeticChromosome.cpp"
"SvmGenerationPopulationFactory.cpp"
"SvmHyperplaneDistance.cpp"
"SvmKernelChromosome.cpp"
"SvmKernelGridGeneration.cpp"
"SvmKernelRandomGeneration.cpp"
"SvmKernelTraining.cpp"
"SvmMemeticFeatureSetPopulationFactory.cpp"
"SvmMetricFactory.cpp"
"SvmMutationFactory.cpp"
"SvmTrainingSetChromosome.cpp"
"SvmTrainingSetCrossoverFactory.cpp"
"SvmTrainingSetMutationFactory.cpp"
"SvmTrainingSetPopulationFactory.cpp"
"SvmUtils.cpp"
"SvmValidationFactory.cpp"
"SvmVisualization.cpp"
"TraningSetCrossover.cpp"
)
source_group("Source Files" FILES ${Source_Files})
	
set(ALL_FILES
    ${Header_Files}
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

################################################################################
# Target
################################################################################
add_library(${PROJECT_NAME} STATIC ${ALL_FILES})


if( OpenCV_FOUND )
  # Additional Include Directories
  include_directories( ${OpenCV_INCLUDE_DIRS} )

  # Additional Library Directories
  link_directories( ${OpenCV_LIB_DIR} )

  # Additional Dependencies
  # target_link_libraries( ${PROJECT_NAME} )
endif()

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

target_link_libraries(${PROJECT_NAME} PUBLIC
	libPlatform
	libDataset
	SvmLib
	libGeneticComponents
  ${OpenCV_LIBS}
)

add_compile_definitions(_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)

#use_package(${PROJECT_NAME} Microsoft.Gsl 0.1.2.2)#
