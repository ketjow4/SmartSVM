#pragma once

#include <memory>

#include "AddingTrainingSetExamples.h"
#include "libGeneticComponents/IStopCondition.h"
#include "libGeneticComponents/BaseCrossoverOperator.h"
#include "libGeneticComponents/IMutationOperator.h"
#include "libGeneticComponents/ISelectionOperator.h"
#include "libGeneticComponents/IPopulationGeneration.h"
#include "libGeneticComponents/ICrossoverSelection.h"
#include "libSvmComponents/SvmKernelChromosome.h"
#include "libSvmComponents/ISvmTraining.h"
#include "libSvmComponents/ISvmMetricsCalculator.h"
#include "libSvmComponents/SvmVisualization.h"
#include "libSvmComponents/SvmTrainingSetChromosome.h"
#include "libSvmComponents/SvmFeatureSetChromosome.h"
#include "libSvmComponents/EducationOfTrainingSet.h"
#include "libSvmComponents/CrossoverCompensation.h"
#include "libSvmComponents/MemeticTraningSetAdaptation.h"
#include "libSvmComponents/SuperIndividualsCreation.h"
#include "libSvmComponents/CompensationInformation.h"
#include "libSvmComponents/SvmConfigStructures.h"
#include "libSvmComponents/SvmFeatureSetMemeticChromosome.h"

#include "libSvmComponents/MemeticFeaturesPool.h"

#include "libSvmComponents/MemeticFeatureCompensation.h"
#include "libSvmComponents/MemeticFeaturesEducation.h"
#include "libSvmComponents/MemeticFeaturesCompensationGeneration.h"
#include "libSvmComponents/MemeticFeaturesSuperIndividualsGeneration.h"
#include "libSvmComponents/MemeticFeaturesAdaptation.h"

#include "libSvmComponents/SvmCustomKernelChromosome.h"
#include "SequentialGammaInternals.h"
#include "MultipleGammaInternals.h"

#include "libSvmComponents/SvmValidationStrategy.h"
#include "RbfLinearInternals.h"


//#include "libGeneticSvm/GeneticKernelEvolutionWorkflow.h"

namespace svmComponents
{
class SvmKernelTraining;

template <class chromosome>
using MutationOperator = std::shared_ptr<geneticComponents::IMutationOperator<chromosome>>;

template <class chromosome>
using CrossoverOperator = std::shared_ptr<geneticComponents::BaseCrossoverOperator<chromosome>>;

template <class chromosome>
using SelectionOperator = std::shared_ptr<geneticComponents::ISelectionOperator<chromosome>>;

template <class chromosome>
using PopulationGeneration = std::shared_ptr<geneticComponents::IPopulationGeneration<chromosome>>;

template <class chromosome>
using StopCondition = std::shared_ptr<geneticComponents::IStopCondition<chromosome>>;

template <class chromosome>
using ParentSelection = std::shared_ptr<geneticComponents::ICrossoverSelection<chromosome>>;

template <class chromosome>
using TrainingMethod = std::shared_ptr<ISvmTraining<chromosome>>;

template <class chromosome>
using ValidationMethod = std::shared_ptr<svmStrategies::IValidationStrategy<chromosome>>;


//using FeatureSetOptimizationWorkflow = std::shared_ptr<genetic::IFeatureSelectionWorkflow<svmComponents::SvmFeatureSetMemeticChromosome>>;

struct GeneticKernelEvolutionConfiguration
{
	explicit GeneticKernelEvolutionConfiguration(const platform::Subtree& config);

	const SvmAlgorithmConfiguration m_svmConfig;
	const unsigned int m_populationSize;

	const TrainingMethod<SvmKernelChromosome> m_training;
	const StopCondition<SvmKernelChromosome> m_stopCondition;
	const CrossoverOperator<SvmKernelChromosome> m_crossover;
	const MutationOperator<SvmKernelChromosome> m_mutation;
	const SelectionOperator<SvmKernelChromosome> m_selection;
	const PopulationGeneration<SvmKernelChromosome> m_populationGeneration;
	const ParentSelection<SvmKernelChromosome> m_parentSelection;

private:
	GeneticKernelEvolutionConfiguration(const platform::Subtree& config, SvmAlgorithmConfiguration svmConfig);
};

struct GeneticTrainingSetEvolutionConfiguration
{
	explicit GeneticTrainingSetEvolutionConfiguration(const platform::Subtree& config,
	                                                  const dataset::Dataset<std::vector<float>, float>& traningSet);

	const SvmAlgorithmConfiguration m_svmConfig;
	const unsigned int m_populationSize;
	const std::vector<unsigned int> m_labelsCount;
	const unsigned int m_numberOfClassExamples;

	const std::shared_ptr<ITrainingSet> m_trainingSetInterface;
	const TrainingMethod<SvmTrainingSetChromosome> m_training;
	const StopCondition<SvmTrainingSetChromosome> m_stopCondition;
	const CrossoverOperator<SvmTrainingSetChromosome> m_crossover;
	const MutationOperator<SvmTrainingSetChromosome> m_mutation;
	const SelectionOperator<SvmTrainingSetChromosome> m_selection;
	const PopulationGeneration<SvmTrainingSetChromosome> m_populationGeneration;
	const ParentSelection<SvmTrainingSetChromosome> m_parentSelection;
	const ValidationMethod<SvmTrainingSetChromosome> m_validationMethod;

private:
	explicit GeneticTrainingSetEvolutionConfiguration(const platform::Subtree& config,
	                                                  const dataset::Dataset<std::vector<float>, float>& traningSet,
	                                                  SvmAlgorithmConfiguration svmConfig);
};

struct GeneticFeatureSetEvolutionConfiguration
{
	explicit GeneticFeatureSetEvolutionConfiguration(const platform::Subtree& config,
	                                                 const dataset::Dataset<std::vector<float>, float>& traningSet);

	const SvmAlgorithmConfiguration m_svmConfig;
	const unsigned int m_populationSize;

	const TrainingMethod<SvmFeatureSetChromosome> m_training;
	const StopCondition<SvmFeatureSetChromosome> m_stopCondition;
	const CrossoverOperator<SvmFeatureSetChromosome> m_crossover;
	const MutationOperator<SvmFeatureSetChromosome> m_mutation;
	const SelectionOperator<SvmFeatureSetChromosome> m_selection;
	const PopulationGeneration<SvmFeatureSetChromosome> m_populationGeneration;
	const ParentSelection<SvmFeatureSetChromosome> m_parentSelection;

private:
	explicit GeneticFeatureSetEvolutionConfiguration(const platform::Subtree& config,
	                                                 const dataset::Dataset<std::vector<float>, float>& traningSet,
	                                                 SvmAlgorithmConfiguration svmConfig);
};

struct MemeticTrainingSetEvolutionConfiguration
{
	explicit MemeticTrainingSetEvolutionConfiguration(const platform::Subtree& config,
	                                                  const dataset::Dataset<std::vector<float>, float>& traningSet);

	const SvmAlgorithmConfiguration m_svmConfig;
	const unsigned int m_populationSize;
	const unsigned int m_numberOfClasses;
	const std::vector<unsigned int> m_labelsCount;
	const double m_superIndividualAlpha;
	const unsigned int m_initialNumberOfClassExamples;


	const std::shared_ptr<ITrainingSet> m_trainingSetInterface;
	const TrainingMethod<SvmTrainingSetChromosome> m_training;
	const StopCondition<SvmTrainingSetChromosome> m_stopCondition;
	const CrossoverOperator<SvmTrainingSetChromosome> m_crossover;
	const MutationOperator<SvmTrainingSetChromosome> m_mutation;
	const SelectionOperator<SvmTrainingSetChromosome> m_selection;
	const PopulationGeneration<SvmTrainingSetChromosome> m_populationGeneration;
	const ParentSelection<SvmTrainingSetChromosome> m_parentSelection;
	const ValidationMethod<SvmTrainingSetChromosome> m_validationMethod;
	const std::shared_ptr<CompensationInformation> m_compensationGeneration;
	const std::shared_ptr<EducationOfTrainingSet> m_education;
	const std::shared_ptr<CrossoverCompensation> m_compensation;
	const std::shared_ptr<SupportVectorPool> m_supporVectorPool;
	const std::shared_ptr<SuperIndividualsCreation> m_superIndivudualsGeneration;
	const std::shared_ptr<MemeticTrainingSetAdaptation> m_adaptation;
	

private:
	explicit MemeticTrainingSetEvolutionConfiguration(const platform::Subtree& config,
	                                                  const dataset::Dataset<std::vector<float>, float>& traningSet,
	                                                  SvmAlgorithmConfiguration svmConfig);
};

struct MemeticFeatureSetEvolutionConfiguration
{
	explicit MemeticFeatureSetEvolutionConfiguration(const platform::Subtree& config,
	                                                 const dataset::Dataset<std::vector<float>, float>& traningSet);

	const SvmAlgorithmConfiguration m_svmConfig;
	const unsigned int m_populationSize;
	const double m_superIndividualAlpha;
	const unsigned int m_initialNumberOfClassExamples;
	const unsigned int m_numberOfClasses;
	const std::vector<unsigned int> m_labelsCount;

	const TrainingMethod<SvmFeatureSetMemeticChromosome> m_training;
	const StopCondition<SvmFeatureSetMemeticChromosome> m_stopCondition;
	const CrossoverOperator<SvmFeatureSetMemeticChromosome> m_crossover;
	const MutationOperator<SvmFeatureSetMemeticChromosome> m_mutation;
	const SelectionOperator<SvmFeatureSetMemeticChromosome> m_selection;
	const PopulationGeneration<SvmFeatureSetMemeticChromosome> m_populationGeneration;
	const ParentSelection<SvmFeatureSetMemeticChromosome> m_parentSelection;
	const std::shared_ptr<MemeticFeaturesCompensationGeneration> m_compensationGeneration;
	const std::shared_ptr<MemeticFeaturesEducation> m_education;
	const std::shared_ptr<MemeticFeatureCompensation> m_compensation;
	const std::shared_ptr<MemeticFeaturesPool> m_supporVectorPool;
	const std::shared_ptr<MemeticFeaturesSuperIndividualsGeneration> m_superIndivudualsGeneration;
	const std::shared_ptr<MemeticFeaturesAdaptation> m_adaptation;

private:
	explicit MemeticFeatureSetEvolutionConfiguration(const platform::Subtree& config,
	                                                 const dataset::Dataset<std::vector<float>, float>& traningSet,
	                                                 SvmAlgorithmConfiguration svmConfig,
	                                                 std::string trainingDataPath,
	                                                 std::string outputPath);
};

struct CustomKernelEvolutionConfiguration
{
	explicit CustomKernelEvolutionConfiguration(const platform::Subtree& config,
	                                            const dataset::Dataset<std::vector<float>, float>& traningSet);

	const SvmAlgorithmConfiguration m_svmConfig;
	const unsigned int m_populationSize;
	const std::vector<unsigned int> m_labelsCount;
	const unsigned int m_numberOfClassExamples;

	const TrainingMethod<SvmCustomKernelChromosome> m_training;
	const StopCondition<SvmCustomKernelChromosome> m_stopCondition;
	const CrossoverOperator<SvmCustomKernelChromosome> m_crossover;
	const MutationOperator<SvmCustomKernelChromosome> m_mutation;
	const SelectionOperator<SvmCustomKernelChromosome> m_selection;
	const PopulationGeneration<SvmCustomKernelChromosome> m_populationGeneration;
	const ParentSelection<SvmCustomKernelChromosome> m_parentSelection;

private:
	explicit CustomKernelEvolutionConfiguration(const platform::Subtree& config,
	                                            const dataset::Dataset<std::vector<float>, float>& traningSet,
	                                            SvmAlgorithmConfiguration svmConfig);
};

struct SequentialGammaConfig
{
	explicit SequentialGammaConfig(const platform::Subtree& config,
	                               const dataset::Dataset<std::vector<float>, float>& traningSet);

	const SvmAlgorithmConfiguration m_svmConfig;
	const unsigned int m_populationSize;
	const std::vector<unsigned int> m_labelsCount;
	const unsigned int m_numberOfClassExamples;
	const double m_superIndividualAlpha;
	const bool m_trainAlpha;
	const bool m_shrinkOnBestOnly;
	const int m_genererateEveryGeneration;
	const std::string m_helperAlgorithmName;

	const bool m_useSmallerGamma;
	const double m_logStepGamma;

	const TrainingMethod<SvmCustomKernelChromosome> m_training;
	const StopCondition<SvmCustomKernelChromosome> m_stopCondition;
	const CrossoverOperator<SvmCustomKernelChromosome> m_crossover;
	const MutationOperator<SvmCustomKernelChromosome> m_mutation;
	const SelectionOperator<SvmCustomKernelChromosome> m_selection;
	const PopulationGeneration<SvmCustomKernelChromosome> m_populationGeneration;
	const ParentSelection<SvmCustomKernelChromosome> m_parentSelection;
	const ValidationMethod<SvmCustomKernelChromosome> m_validationMethod;

	svmComponents::SupportVectorPoolGamma m_supportVectorPoolElement;
	std::shared_ptr<svmComponents::EducationOfTrainingSetGamma> m_educationElement;
	svmComponents::CrossoverCompensationGamma m_crossoverCompensationElement;
	svmComponents::MemeticTrainingSetAdaptationGamma m_adaptationElement;
	std::shared_ptr < svmComponents::SuperIndividualsCreationGamma> m_superIndividualsGenerationElement;
	svmComponents::CompensationInformationGamma m_compensationGenerationElement;


private:
	explicit SequentialGammaConfig(const platform::Subtree& config,
	                               const dataset::Dataset<std::vector<float>, float>& traningSet,
	                               SvmAlgorithmConfiguration svmConfig);
};

struct MutlipleGammaMASVMConfig
{
	explicit MutlipleGammaMASVMConfig(const platform::Subtree& config,
		const dataset::Dataset<std::vector<float>, float>& traningSet);

	const SvmAlgorithmConfiguration m_svmConfig;
	const unsigned int m_populationSize;
	const std::vector<unsigned int> m_labelsCount;
	const unsigned int m_numberOfClassExamples;
	const double m_superIndividualAlpha;

	const TrainingMethod<SvmCustomKernelChromosome> m_training;
	const StopCondition<SvmCustomKernelChromosome> m_stopCondition;
	const CrossoverOperator<SvmCustomKernelChromosome> m_crossover;
	const MutationOperator<SvmCustomKernelChromosome> m_mutation;
	const SelectionOperator<SvmCustomKernelChromosome> m_selection;
	const PopulationGeneration<SvmCustomKernelChromosome> m_populationGeneration;
	const ParentSelection<SvmCustomKernelChromosome> m_parentSelection;
	const ValidationMethod<SvmCustomKernelChromosome> m_validationMethod;

	svmComponents::MultipleGammaSupportVectorPool m_supportVectorPoolElement;
	std::shared_ptr<svmComponents::MultipleGammaEducationOfTrainingSet> m_educationElement;
	svmComponents::MultipleGammaCrossoverCompensation m_crossoverCompensationElement;
	svmComponents::MultipleGammaMemeticTrainingSetAdaptation m_adaptationElement;
	std::shared_ptr < svmComponents::MultipleGammaSuperIndividualsCreation> m_superIndividualsGenerationElement;
	svmComponents::MultipleGammaCompensationInformation m_compensationGenerationElement;

private:
	explicit MutlipleGammaMASVMConfig(const platform::Subtree& config,
		const dataset::Dataset<std::vector<float>, float>& traningSet,
		SvmAlgorithmConfiguration svmConfig);
};

struct RbfLinearConfig
{
	explicit RbfLinearConfig(const platform::Subtree& config,
	                         const dataset::Dataset<std::vector<float>, float>& traningSet);

	
	const SvmAlgorithmConfiguration m_svmConfig;
	unsigned int m_populationSize;
	const std::vector<unsigned int> m_labelsCount;
	const unsigned int m_numberOfClassExamples;
	const double m_superIndividualAlpha;
	const bool m_trainAlpha;

	const TrainingMethod<SvmCustomKernelChromosome> m_training;
	const StopCondition<SvmCustomKernelChromosome> m_stopCondition;
	const CrossoverOperator<SvmCustomKernelChromosome> m_crossover;
	const MutationOperator<SvmCustomKernelChromosome> m_mutation;
	const SelectionOperator<SvmCustomKernelChromosome> m_selection;
	const PopulationGeneration<SvmCustomKernelChromosome> m_populationGeneration;
	const ParentSelection<SvmCustomKernelChromosome> m_parentSelection;
	const ValidationMethod<SvmCustomKernelChromosome> m_validationMethod;

	svmComponents::SupportVectorPoolRbfLinear m_supportVectorPoolElement;
	std::shared_ptr<svmComponents::EducationOfTrainingSetRbfLinear> m_educationElement;
	svmComponents::CrossoverCompensationRbfLinear m_crossoverCompensationElement;
	svmComponents::MemeticTrainingSetAdaptationRbfLinear m_adaptationElement;
	std::shared_ptr<svmComponents::SuperIndividualsCreationRbfLinear> m_superIndividualsGenerationElement;
	svmComponents::CompensationInformationRbfLinear m_compensationGenerationElement;

private:
	explicit RbfLinearConfig(const platform::Subtree& config,
	                         const dataset::Dataset<std::vector<float>, float>& traningSet,
	                         SvmAlgorithmConfiguration svmConfig);
};
} // namespace svmComponents
