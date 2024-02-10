#pragma once

#include "libSvmComponents/GeneticAlgorithmsConfigs.h"
#include "libSvmComponents/SvmKernelChromosome.h"
#include "libGeneticSvm/ISvmAlgorithm.h"
#include "libGeneticSvm/SvmWorkflowConfigStruct.h"
#include "libGeneticSvm/Timer.h"
#include "libGeneticSvm/IGeneticWorkflow.h"
#include "libGeneticSvm/GeneticWorkflowResultLogger.h"
#include "libGeneticSvm/IDatasetLoader.h"

#include "libSvmComponents/SvmValidationStrategy.h"
#include "libGeneticStrategies/StopConditionStrategy.h"
#include "libGeneticStrategies/CrossoverStrategy.h"
#include "libGeneticStrategies/MutationStrategy.h"
#include "libGeneticStrategies/SelectionStrategy.h"
#include "libGeneticStrategies/CreatePopulationStrategy.h"
#include "libStrategies/FileSinkStrategy.h"
#include "libSvmStrategies/CreateSvmVisualizationStrategy.h"
#include "libSvmStrategies/SvmTrainingStrategy.h"
#include "libSvmStrategies/MemeticEducationStrategy.h"
#include "libSvmStrategies/CrossoverCompensationStrategy.h"
#include "libGeneticStrategies/CombinePopulationsStrategy.h"
#include "libSvmStrategies/UpdateSupportVectorPoolStrategy.h"
#include "libSvmStrategies/SuperIndividualCreationStrategy.h"
#include "libSvmStrategies/MemeticAdaptationStrategy.h"
#include "libSvmStrategies/CompensationInformationStrategy.h"
#include "libGeneticStrategies/CrossoverParentSelectionStrategy.h"
#include "AllModelsLogger.h"
#include "libSvmComponents/AddingTrainingSetExamples.h"

namespace genetic
{
class MemeticTraningSetWorkflow : public ISvmAlgorithm, public ITrainingSetOptimizationWorkflow<svmComponents::SvmTrainingSetChromosome>
{
public:
    explicit MemeticTraningSetWorkflow(const SvmWokrflowConfiguration& config,
                                       svmComponents::MemeticTrainingSetEvolutionConfiguration algorithmConfig,
                                       IDatasetLoader& workflow,
									   platform::Subtree fullConfig);

    std::shared_ptr<phd::svm::ISvm> run() override;

    void runGeneticAlgorithm() override;
    void initialize() override;
    void setupKernelParameters(const svmComponents::SvmKernelChromosome& kernelParameters) override;
    GeneticWorkflowResultLogger& getResultLogger() override;
    svmComponents::SvmTrainingSetChromosome getBestChromosomeInGeneration() const override;
    geneticComponents::Population<svmComponents::SvmTrainingSetChromosome> getPopulation() const override;
    dataset::Dataset<std::vector<float>, float> getBestTrainingSet() const override;
    void setupFeaturesSet(const svmComponents::SvmFeatureSetChromosome& featureSetChromosome) override;

    geneticComponents::Population<svmComponents::SvmTrainingSetChromosome> initNoEvaluate(int popSize) override;
    void performGeneticOperations(geneticComponents::Population<svmComponents::SvmTrainingSetChromosome>& population) override;

    void setTimer(std::shared_ptr<Timer> timer) override;

    unsigned int getInitialTrainingSetSize() override;
    void setK(unsigned int k) override;

    unsigned getCurrentTrainingSetSize() override;

    phd::svm::ISvm& getClassifierWithBestDistances() override;

    svmStrategies::MemeticAdaptationStrategy& getAdaptationElement();

    geneticComponents::Population<svmComponents::SvmTrainingSetChromosome> initNoEvaluate(int popSize, int seed) override;
	
private:
    using SvmTrainingSetChromosome = svmComponents::SvmTrainingSetChromosome;

    void initializeGeneticAlgorithm();
    void logResults(const geneticComponents::Population<SvmTrainingSetChromosome>& population,
                    const geneticComponents::Population<SvmTrainingSetChromosome>& testpopulation);
    void logAllModels(geneticComponents::Population<SvmTrainingSetChromosome>& testPopulation);


private:
    platform::Subtree m_fullConfig;
	
    svmComponents::MemeticTrainingSetEvolutionConfiguration m_algorithmConfig;
    svmStrategies::SvmTrainingStrategy<SvmTrainingSetChromosome> m_trainingSvmClassifierElement;
    //svmStrategies::SvmValidationStrategy<SvmTrainingSetChromosome> m_validationElement;
	//svmStrategies::SvmValidationStrategy<SvmTrainingSetChromosome> m_validationSuperIndividualsElement;
	std::shared_ptr<svmStrategies::IValidationStrategy<SvmTrainingSetChromosome>> m_validationElement;
	std::shared_ptr<svmStrategies::IValidationStrategy<SvmTrainingSetChromosome>> m_validationSuperIndividualsElement;
    svmStrategies::SvmValidationStrategy<SvmTrainingSetChromosome> m_valdiationTestDataElement;
    geneticStrategies::StopConditionStrategy<SvmTrainingSetChromosome> m_stopConditionElement;
    geneticStrategies::CrossoverStrategy<SvmTrainingSetChromosome> m_crossoverElement;
    geneticStrategies::MutationStrategy<SvmTrainingSetChromosome> m_mutationElement;
    geneticStrategies::SelectionStrategy<SvmTrainingSetChromosome> m_selectionElement;
    geneticStrategies::CreatePopulationStrategy<SvmTrainingSetChromosome> m_createPopulationElement;
    strategies::FileSinkStrategy m_savePngElement;
    svmStrategies::CreateSvmVisualizationStrategy<SvmTrainingSetChromosome> m_createVisualizationElement;
    svmStrategies::MemeticEducationStrategy m_educationElement;
    svmStrategies::CrossoverCompensationStrategy m_crossoverCompensationElement;
    svmStrategies::MemeticAdaptationStrategy m_adaptationElement;
    svmStrategies::UpdateSupportVectorPoolStrategy m_supportVectorPoolElement;
    svmStrategies::SuperIndividualCreationStrategy m_superIndividualsGenerationElement;
    geneticStrategies::CombinePopulationsStrategy<SvmTrainingSetChromosome> m_populationCombinationElement;
    svmStrategies::SvmTrainingStrategy<SvmTrainingSetChromosome> m_trainingSuperIndividualsElement;
    geneticStrategies::CrossoverParentSelectionStrategy<SvmTrainingSetChromosome> m_parentSelectionElement;
    svmStrategies::CompensationInformationStrategy m_compensationGenerationElement;

	std::shared_ptr<svmComponents::ITrainingSet> m_trainingSetInterface;
	
	
    unsigned int m_numberOfClassExamples;
    std::vector<svmComponents::DatasetVector> m_svPool;

    dataset::Dataset<std::vector<float>, float> m_trainingSet2;
    dataset::Dataset<std::vector<float>, float> m_validationSet2;
    dataset::Dataset<std::vector<float>, float> m_testSet2;

    const dataset::Dataset<std::vector<float>, float>* m_trainingSet;
    const dataset::Dataset<std::vector<float>, float>* m_validationSet;
    const dataset::Dataset<std::vector<float>, float>* m_testSet;
	
    std::filesystem::path m_pngNameSource;
    geneticComponents::Population<SvmTrainingSetChromosome> m_population;

    bool m_needRetrain;
    IDatasetLoader& m_loadingWorkflow;
    std::shared_ptr<Timer> m_timer;
    unsigned int m_generationNumber;
    std::shared_ptr<AllModelsLogger> m_allModelsLogger;

    const SvmWokrflowConfiguration m_config;
    GeneticWorkflowResultLogger m_resultLogger;

    static constexpr const char* m_algorithmName = "Memetic";
};
} // namespace genetic
