#pragma once

#include "libGeneticSvm/ISvmAlgorithm.h"
#include "libGeneticSvm/CombinedAlgorithmsConfig.h"

namespace genetic
{
class SimultaneousWorkflow : public ISvmAlgorithm
{

public:
    SimultaneousWorkflow(const SvmWokrflowConfiguration& config,
                         SimultaneousWorkflowConfig algorithmConfig,
                         IDatasetLoader& workflow);
    
    void logAllModels(AllModelsLogger& logger);
    void createVisualization();

    std::shared_ptr<phd::svm::ISvm> run() override;

    

private:
    void init();

    void performEvolution();

    void evaluate();

    bool isFinished();

    void log();

    //template <class chromosome>
    //void clearlog(IGeneticWorkflow<chromosome>& workflow);



    geneticComponents::Population<svmComponents::SvmKernelChromosome> getKernelPopulation();
    geneticComponents::Population<svmComponents::SvmFeatureSetMemeticChromosome> getFeaturesPopulation();
    geneticComponents::Population<svmComponents::SvmTrainingSetChromosome> getTrainingSetPopulation();

    void setKernelPopulation(geneticComponents::Population<svmComponents::SvmKernelChromosome>& population,
                             geneticComponents::Population<svmComponents::SvmSimultaneousChromosome>& newPop);
    void setTrainingPopulation(geneticComponents::Population<svmComponents::SvmTrainingSetChromosome>& population,
                               geneticComponents::Population<svmComponents::SvmSimultaneousChromosome>& newPop);
    void setFeaturesPopulation(geneticComponents::Population<svmComponents::SvmFeatureSetMemeticChromosome>& population,
                               geneticComponents::Population<svmComponents::SvmSimultaneousChromosome>& newPop);


    TrainingSetOptimizationWorkflow m_trainingSetOptimization;
    KernelOptimizationWorkflow m_kernelOptimization;
    FeatureSetOptimizationWorkflow m_featureSetOptimization;
    std::filesystem::path m_resultFilePath;
    SimultaneousWorkflowConfig m_algorithmConfig;

    svmStrategies::SvmTrainingStrategy<svmComponents::SvmSimultaneousChromosome> m_svmTraining;
    geneticComponents::Population<svmComponents::SvmSimultaneousChromosome> m_pop;
    geneticComponents::Population<svmComponents::SvmSimultaneousChromosome> m_popTestSet;

    svmStrategies::SvmValidationStrategy<svmComponents::SvmSimultaneousChromosome> m_validation;
    svmStrategies::SvmValidationStrategy<svmComponents::SvmSimultaneousChromosome> m_validationTest;
    geneticStrategies::StopConditionStrategy<svmComponents::SvmSimultaneousChromosome> m_stopConditionElement;
    geneticStrategies::SelectionStrategy<svmComponents::SvmSimultaneousChromosome> m_selectionElement;

    svmStrategies::CreateSvmVisualizationStrategy<SvmSimultaneousChromosome> m_createVisualizationElement;

	IDatasetLoader& m_workflow;
	SvmWokrflowConfiguration m_config;

    GeneticWorkflowResultLogger m_resultLogger;
    std::shared_ptr<Timer> m_timer;
    unsigned int m_generationNumber;

    dataset::Dataset<std::vector<float>, float> m_trainingSet;
    dataset::Dataset<std::vector<float>, float> m_validationSet;
    dataset::Dataset<std::vector<float>, float> m_testSet;

    static constexpr const char* m_algorithmName = "SESVM";
};
} // namespace genetic
