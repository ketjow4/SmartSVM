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
#include "libSvmStrategies/CrossoverCompensationStrategy.h"
#include "libGeneticStrategies/CombinePopulationsStrategy.h"
#include "libGeneticStrategies/CrossoverParentSelectionStrategy.h"
#include "libSvmStrategies/MemeticFeaturesStrategies.h"

namespace genetic
{
class MemeticFeaturesSelection : public ISvmAlgorithm, public IFeatureSelectionWorkflow<svmComponents::SvmFeatureSetMemeticChromosome>
{
public:
    explicit MemeticFeaturesSelection(const SvmWokrflowConfiguration& config,
                                      svmComponents::MemeticFeatureSetEvolutionConfiguration algorithmConfig,
                                      IDatasetLoader& workflow);

    std::shared_ptr<phd::svm::ISvm> run() override;

    void runGeneticAlgorithm() override;
    void initialize() override;
    void setupKernelParameters(const svmComponents::SvmKernelChromosome& kernelParameters) override;
    GeneticWorkflowResultLogger& getResultLogger() override;
    svmComponents::SvmFeatureSetMemeticChromosome getBestChromosomeInGeneration() const override;
    geneticComponents::Population<svmComponents::SvmFeatureSetMemeticChromosome> getPopulation() const override;

    dataset::Dataset<std::vector<float>, float> getFilteredTraningSet() override;
    dataset::Dataset<std::vector<float>, float> getFilteredValidationSet() override;
    dataset::Dataset<std::vector<float>, float> getFilteredTestSet() override;
    void setupTrainingSet(const dataset::Dataset<std::vector<float>, float>& trainingSet) override;

    geneticComponents::Population<svmComponents::SvmFeatureSetMemeticChromosome> initNoEvaluate(int popSize) override;
    geneticComponents::Population<svmComponents::SvmFeatureSetMemeticChromosome> initNoEvaluate(int popSize, int seed) override;
    void performGeneticOperations(geneticComponents::Population<svmComponents::SvmFeatureSetMemeticChromosome>& population) override;


    std::shared_ptr<Timer> getTimer() override;
    void setTimer(std::shared_ptr<Timer> timer) override;

private:
    using SvmFeatureSetMemeticChromosome = svmComponents::SvmFeatureSetMemeticChromosome;

    void initializeGeneticAlgorithm();
    void logResults(const geneticComponents::Population<SvmFeatureSetMemeticChromosome>& population,
                    const geneticComponents::Population<SvmFeatureSetMemeticChromosome>& testpopulation);

    geneticComponents::Population<SvmFeatureSetMemeticChromosome> createPopulationWithTimeMeasurement(int popSize);

    svmComponents::MemeticFeatureSetEvolutionConfiguration m_algorithmConfig;
    svmStrategies::SvmTrainingStrategy<SvmFeatureSetMemeticChromosome> m_trainingSvmClassifierElement;
    svmStrategies::SvmValidationStrategy<SvmFeatureSetMemeticChromosome> m_valdiationElement;
    svmStrategies::SvmValidationStrategy<SvmFeatureSetMemeticChromosome> m_valdiationTestDataElement;
    geneticStrategies::StopConditionStrategy<SvmFeatureSetMemeticChromosome> m_stopConditionElement;
    geneticStrategies::CrossoverStrategy<SvmFeatureSetMemeticChromosome> m_crossoverElement;
    geneticStrategies::MutationStrategy<SvmFeatureSetMemeticChromosome> m_mutationElement;
    geneticStrategies::SelectionStrategy<SvmFeatureSetMemeticChromosome> m_selectionElement;
    geneticStrategies::CreatePopulationStrategy<SvmFeatureSetMemeticChromosome> m_createPopulationElement;
    strategies::FileSinkStrategy m_savePngElement;
    svmStrategies::CreateSvmVisualizationStrategy<SvmFeatureSetMemeticChromosome> m_createVisualizationElement;
    svmStrategies::MemeticFeaturesEducationStrategy m_educationElement;
    svmStrategies::MemeticCrossoverCompensationStrategy m_crossoverCompensationElement;
    svmStrategies::MemeticFeaturesAdaptationStrategy m_adaptationElement;
    svmStrategies::MemeticUpdateFeaturesPoolStrategy m_featuresPoolElement;
    svmStrategies::MemeticSuperIndividualCreationStrategy m_superIndividualsGenerationElement;
    geneticStrategies::CombinePopulationsStrategy<SvmFeatureSetMemeticChromosome> m_populationCombinationElement;
    svmStrategies::SvmTrainingStrategy<SvmFeatureSetMemeticChromosome> m_trainingSuperIndividualsElement;
    svmStrategies::SvmValidationStrategy<SvmFeatureSetMemeticChromosome> m_validationSuperIndividualsElement;
    geneticStrategies::CrossoverParentSelectionStrategy<SvmFeatureSetMemeticChromosome> m_parentSelectionElement;
    svmStrategies::MemeticCompensationGenerationStrategy m_compensationGenerationElement;

    unsigned int m_numberOfClassExamples;
    std::vector<svmComponents::Feature> m_featurePool;

    dataset::Dataset<std::vector<float>, float> m_trainingSet22222;

    const dataset::Dataset<std::vector<float>, float>* m_trainingSet;
    const dataset::Dataset<std::vector<float>, float>* m_validationSet;
    const dataset::Dataset<std::vector<float>, float>* m_testSet;
    std::filesystem::path m_pngNameSource;
    geneticComponents::Population<SvmFeatureSetMemeticChromosome> m_population;

    bool m_needRetrain;
    IDatasetLoader& m_loadingWorkflow;
    std::shared_ptr<Timer> m_timer;
    unsigned int m_generationNumber;

    const SvmWokrflowConfiguration m_config;
    //logger::LogFrontend m_logger;
    GeneticWorkflowResultLogger m_resultLogger;
    size_t m_featureNumberAll;

    static constexpr const char* m_algorithmName = "Mem_FS";
};
} // namespace genetic
