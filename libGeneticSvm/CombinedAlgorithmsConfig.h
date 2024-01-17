#pragma once

#include <libSvmComponents/SvmConfigStructures.h>
#include "libGeneticSvm/GeneticKernelEvolutionWorkflow.h"
#include "libSvmComponents/SvmHyperplaneDistance.h"
#include "libSvmComponents/SvmCustomKernelFeaturesSelectionChromosome.h"

//#include "libGeneticSvm/GaSvmWorkflow.h"
//#include "libSvmComponents/GeneticAlgorithmsConfigs.h"
//#include <libGeneticSvm/IDatasetLoader.h>

namespace genetic
{
using TrainingSetOptimizationWorkflow = std::shared_ptr<ITrainingSetOptimizationWorkflow<svmComponents::SvmTrainingSetChromosome>>;
using KernelOptimizationWorkflow = std::shared_ptr<IKernelOptimalizationWorkflow<svmComponents::SvmKernelChromosome>>;

//using FeatureChromosome = svmComponents::SvmFeatureSetMemeticChromosome;
//or
//using FeatureChromosome = svmComponents::SvmFeatureSetChromosome;

using FeatureSetOptimizationWorkflow = std::shared_ptr<IFeatureSelectionWorkflow<svmComponents::SvmFeatureSetMemeticChromosome>>;

inline svmComponents::SvmFeatureSetChromosome convertToOldChromosome(const svmComponents::SvmFeatureSetMemeticChromosome& chromosome,
                                                                     unsigned int featureNumber)
{
    std::vector<bool> oldRepresentation;
    oldRepresentation.resize(featureNumber, false);

    for (const auto feature : chromosome.getDataset())
    {
        oldRepresentation[feature.id] = true;
    }
    return svmComponents::SvmFeatureSetChromosome(std::move(oldRepresentation));
}

struct GeneticAlternatingEvolutionConfiguration
{
    explicit GeneticAlternatingEvolutionConfiguration(const platform::Subtree& config, IDatasetLoader& loadingWorkflow);

    const svmComponents::SvmAlgorithmConfiguration m_svmConfig;
    const TrainingSetOptimizationWorkflow m_trainingSetOptimization;
    const KernelOptimizationWorkflow m_kernelOptimization;
    const svmComponents::StopCondition<svmComponents::SvmKernelChromosome> m_stopKernel;
    const svmComponents::StopCondition<svmComponents::SvmTrainingSetChromosome> m_stopTrainingSet;
};

struct KTFGeneticEvolutionConfiguration
{
    explicit KTFGeneticEvolutionConfiguration(const platform::Subtree& config, IDatasetLoader& loadingWorkflow);

    const TrainingSetOptimizationWorkflow m_trainingSetOptimization;
    const KernelOptimizationWorkflow m_kernelOptimization;
    const FeatureSetOptimizationWorkflow m_featureSetOptimization;
    const svmComponents::StopCondition<svmComponents::SvmKernelChromosome> m_stopKernel;
    const svmComponents::StopCondition<svmComponents::SvmTrainingSetChromosome> m_stopTrainingSet;
    const svmComponents::StopCondition<svmComponents::SvmFeatureSetMemeticChromosome> m_stopFeatureSet;
};

struct TFGeneticEvolutionConfiguration
{
    explicit TFGeneticEvolutionConfiguration(const platform::Subtree& config,
                                             IDatasetLoader& loadingWorkflow,
                                             const std::string& algorithmName);

    const TrainingSetOptimizationWorkflow m_trainingSetOptimization;
    const FeatureSetOptimizationWorkflow m_featureSetOptimization;
    const svmComponents::StopCondition<svmComponents::SvmTrainingSetChromosome> m_stopTrainingSet;
    const svmComponents::StopCondition<svmComponents::SvmFeatureSetMemeticChromosome> m_stopFeatureSet;
};

struct SimultaneousWorkflowConfig
{
    explicit SimultaneousWorkflowConfig(const platform::Subtree& config, IDatasetLoader& loadingWorkflow);

    unsigned int m_populationSize;
    platform::Subtree m_config;

    const svmComponents::SvmAlgorithmConfiguration m_svmConfig;
    const TrainingSetOptimizationWorkflow m_trainingSetOptimization;
    const KernelOptimizationWorkflow m_kernelOptimization;
    const FeatureSetOptimizationWorkflow m_featureSetOptimization;
    const svmComponents::StopCondition<svmComponents::SvmSimultaneousChromosome> m_stopCondition;
    std::shared_ptr<svmComponents::ISvmTraining<svmComponents::SvmSimultaneousChromosome>> m_svmTraining;
    std::shared_ptr<geneticComponents::ISelectionOperator<svmComponents::SvmSimultaneousChromosome>> m_selectionElement;
    
};

struct RandomSearchWorkflowConfig
{
    explicit RandomSearchWorkflowConfig(const platform::Subtree& config, IDatasetLoader& loadingWorkflow);

    unsigned int m_populationSize;

    const svmComponents::SvmAlgorithmConfiguration m_svmConfig;
    const TrainingSetOptimizationWorkflow m_trainingSetOptimization;
    const KernelOptimizationWorkflow m_kernelOptimization;
    const FeatureSetOptimizationWorkflow m_featureSetOptimization;
    std::shared_ptr<svmComponents::ISvmTraining<svmComponents::SvmSimultaneousChromosome>> m_svmTraining;
};

struct RandomSearchWorkflowInitPopsConfig
{
    explicit RandomSearchWorkflowInitPopsConfig(const platform::Subtree& config, IDatasetLoader& loadingWorkflow);

    unsigned int m_populationSize;

    const svmComponents::SvmAlgorithmConfiguration m_svmConfig;
    const TrainingSetOptimizationWorkflow m_trainingSetOptimization;
    const KernelOptimizationWorkflow m_kernelOptimization;
    const FeatureSetOptimizationWorkflow m_featureSetOptimization;
    std::shared_ptr<svmComponents::ISvmTraining<svmComponents::SvmSimultaneousChromosome>> m_svmTraining;
};

struct RandomSearchWorkflowEvoHelpConfig
{
    explicit RandomSearchWorkflowEvoHelpConfig(const platform::Subtree& config, IDatasetLoader& loadingWorkflow);

    unsigned int m_populationSize;

    const svmComponents::SvmAlgorithmConfiguration m_svmConfig;
    const TrainingSetOptimizationWorkflow m_trainingSetOptimization;
    const KernelOptimizationWorkflow m_kernelOptimization;
    const FeatureSetOptimizationWorkflow m_featureSetOptimization;
    std::shared_ptr<svmComponents::ISvmTraining<svmComponents::SvmSimultaneousChromosome>> m_svmTraining;
};

enum class SvMode
{
	none,
	previousOnes,
	all
};

struct EnsembleTreeWorkflowConfig
{
    explicit EnsembleTreeWorkflowConfig(const platform::Subtree& config, IDatasetLoader& loadingWorkflow);

    unsigned int m_populationSize;
    bool m_constKernel;
    bool m_switchFitness;
    svmComponents::MetricMode m_metricMode;

    platform::Subtree m_config;
    bool m_addSvToTraining;
    SvMode m_SvMode;
    bool m_useDasvmKernel;
	
    const svmComponents::SvmAlgorithmConfiguration m_svmConfig;
    const TrainingSetOptimizationWorkflow m_trainingSetOptimization;
    const KernelOptimizationWorkflow m_kernelOptimization;
    const FeatureSetOptimizationWorkflow m_featureSetOptimization;
    const svmComponents::StopCondition<svmComponents::SvmSimultaneousChromosome> m_stopCondition;
    std::shared_ptr<svmComponents::ISvmTraining<svmComponents::SvmSimultaneousChromosome>> m_svmTraining;
    std::shared_ptr<geneticComponents::ISelectionOperator<svmComponents::SvmSimultaneousChromosome>> m_selectionElement;

};


using namespace svmComponents;
struct SequentialGammaConfigWithFeatureSelection
{
    explicit SequentialGammaConfigWithFeatureSelection(const platform::Subtree& config,
        const dataset::Dataset<std::vector<float>, float>& traningSet, IDatasetLoader& loadingWorkflow);

    const svmComponents::SvmAlgorithmConfiguration m_svmConfig;
    const unsigned int m_populationSize;
    const std::vector<unsigned int> m_labelsCount;
    const unsigned int m_numberOfClassExamples;
    const double m_superIndividualAlpha;
    const bool m_trainAlpha;
    const bool m_shrinkOnBestOnly;
    const int m_genererateEveryGeneration;

    const TrainingMethod<SvmCustomKernelFeaturesSelectionChromosome> m_training;
    const CrossoverOperator<SvmCustomKernelChromosome> m_crossover;
    const MutationOperator<SvmCustomKernelChromosome> m_mutation;
    const SelectionOperator<SvmCustomKernelFeaturesSelectionChromosome> m_selection;
    const PopulationGeneration<SvmCustomKernelChromosome> m_populationGeneration;
    const ParentSelection<SvmCustomKernelChromosome> m_parentSelection;
    const ValidationMethod<SvmCustomKernelFeaturesSelectionChromosome> m_validationMethod;

    svmComponents::SupportVectorPoolGamma m_supportVectorPoolElement;
    std::shared_ptr<svmComponents::EducationOfTrainingSetGamma> m_educationElement;
    svmComponents::CrossoverCompensationGamma m_crossoverCompensationElement;
    svmComponents::MemeticTrainingSetAdaptationGamma m_adaptationElement;
    std::shared_ptr < svmComponents::SuperIndividualsCreationGamma> m_superIndividualsGenerationElement;
    svmComponents::CompensationInformationGamma m_compensationGenerationElement;

    FeatureSetOptimizationWorkflow m_featureSetOptimization;
    const svmComponents::StopCondition<svmComponents::SvmCustomKernelFeaturesSelectionChromosome> m_stopCondition;
    //std::shared_ptr<svmComponents::ISvmTraining<svmComponents::SvmCustomKernelFeaturesSelectionChromosome>> m_svmTraining;

private:
    explicit SequentialGammaConfigWithFeatureSelection(const platform::Subtree& config,
        const dataset::Dataset<std::vector<float>, float>& traningSet,
        SvmAlgorithmConfiguration svmConfig, IDatasetLoader& loadingWorkflow);
};
} // namespace genetic
