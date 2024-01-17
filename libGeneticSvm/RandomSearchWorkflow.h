#pragma once

#include "libGeneticSvm/ISvmAlgorithm.h"
#include "libGeneticSvm/CombinedAlgorithmsConfig.h"

namespace genetic
{
//number of runs will be loaded from file, the same as average for features and traning set size

class RandomSearchWorkflow : public ISvmAlgorithm
{
public:
    RandomSearchWorkflow(const SvmWokrflowConfiguration& config,
                         RandomSearchWorkflowConfig algorithmConfig,
                         IDatasetLoader& workflow);

    std::shared_ptr<phd::svm::ISvm> run() override;

    geneticComponents::Population<svmComponents::SvmSimultaneousChromosome> initRandom(int numberOfTries);

protected:
    void log();

    std::filesystem::path m_resultFilePath;
    RandomSearchWorkflowConfig m_algorithmConfig;
    svmStrategies::SvmTrainingStrategy<svmComponents::SvmSimultaneousChromosome> m_svmTraining;
    geneticComponents::Population<svmComponents::SvmSimultaneousChromosome> m_pop;
    geneticComponents::Population<svmComponents::SvmSimultaneousChromosome> m_popTestSet;
    svmStrategies::SvmValidationStrategy<svmComponents::SvmSimultaneousChromosome> m_validation;
    svmStrategies::SvmValidationStrategy<svmComponents::SvmSimultaneousChromosome> m_validationTest;
    IDatasetLoader& m_workflow;
    SvmWokrflowConfiguration m_config;
    GeneticWorkflowResultLogger m_resultLogger;
    Timer m_timer;
    unsigned int m_generationNumber;
    dataset::Dataset<std::vector<float>, float> m_trainingSet;
    dataset::Dataset<std::vector<float>, float> m_validationSet;
    dataset::Dataset<std::vector<float>, float> m_testSet;
    static constexpr const char* m_algorithmName = "RandomSearch";
};





class RandomSearchWithInitialPopulationsWorkflow : public ISvmAlgorithm
{
public:
    RandomSearchWithInitialPopulationsWorkflow(const SvmWokrflowConfiguration& config,
                                               RandomSearchWorkflowInitPopsConfig algorithmConfig,
                                               IDatasetLoader& workflow);

    geneticComponents::Population<svmComponents::SvmSimultaneousChromosome> initRandom(int numberOfTries);
    std::shared_ptr<phd::svm::ISvm> run() override;
private:
    void log();

    TrainingSetOptimizationWorkflow m_trainingSetOptimization;
    KernelOptimizationWorkflow m_kernelOptimization;
    FeatureSetOptimizationWorkflow m_featureSetOptimization;
    
    std::filesystem::path m_resultFilePath;
    RandomSearchWorkflowInitPopsConfig m_algorithmConfig;
    svmStrategies::SvmTrainingStrategy<svmComponents::SvmSimultaneousChromosome> m_svmTraining;
    geneticComponents::Population<svmComponents::SvmSimultaneousChromosome> m_pop;
    geneticComponents::Population<svmComponents::SvmSimultaneousChromosome> m_popTestSet;
    svmStrategies::SvmValidationStrategy<svmComponents::SvmSimultaneousChromosome> m_validation;
    svmStrategies::SvmValidationStrategy<svmComponents::SvmSimultaneousChromosome> m_validationTest;
    IDatasetLoader& m_workflow;
    SvmWokrflowConfiguration m_config;
    GeneticWorkflowResultLogger m_resultLogger;
    Timer m_timer;
    unsigned int m_generationNumber;
    dataset::Dataset<std::vector<float>, float> m_trainingSet;
    dataset::Dataset<std::vector<float>, float> m_validationSet;
    dataset::Dataset<std::vector<float>, float> m_testSet;
    static constexpr const char* m_algorithmName = "RandomSearchInitialPopulations";
};

class RandomSearchWithHelpFromEvolutionWorkflow : public ISvmAlgorithm
{
public:
    RandomSearchWithHelpFromEvolutionWorkflow(const SvmWokrflowConfiguration& config,
                                              RandomSearchWorkflowEvoHelpConfig algorithmConfig,
                                              IDatasetLoader& workflow);

    std::shared_ptr<phd::svm::ISvm> run() override;

    geneticComponents::Population<svmComponents::SvmSimultaneousChromosome> initRandom(int numberOfTries);

private:
    void log();

    TrainingSetOptimizationWorkflow m_trainingSetOptimization;
    KernelOptimizationWorkflow m_kernelOptimization;
    FeatureSetOptimizationWorkflow m_featureSetOptimization;

    std::filesystem::path m_resultFilePath;
    RandomSearchWorkflowEvoHelpConfig m_algorithmConfig;
    svmStrategies::SvmTrainingStrategy<svmComponents::SvmSimultaneousChromosome> m_svmTraining;
    geneticComponents::Population<svmComponents::SvmSimultaneousChromosome> m_pop;
    geneticComponents::Population<svmComponents::SvmSimultaneousChromosome> m_popTestSet;
    svmStrategies::SvmValidationStrategy<svmComponents::SvmSimultaneousChromosome> m_validation;
    svmStrategies::SvmValidationStrategy<svmComponents::SvmSimultaneousChromosome> m_validationTest;
    IDatasetLoader& m_workflow;
    SvmWokrflowConfiguration m_config;
    GeneticWorkflowResultLogger m_resultLogger;
    Timer m_timer;
    unsigned int m_generationNumber;
    dataset::Dataset<std::vector<float>, float> m_trainingSet;
    dataset::Dataset<std::vector<float>, float> m_validationSet;
    dataset::Dataset<std::vector<float>, float> m_testSet;
    static constexpr const char* m_algorithmName = "RandomSearchEvoHelp";
};
} // namespace genetic
