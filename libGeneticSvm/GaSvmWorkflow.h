#pragma once

#include <filesystem>

#include "libSvmComponents/GeneticAlgorithmsConfigs.h"
#include "libSvmComponents/SvmKernelChromosome.h"
#include "libGeneticSvm/ISvmAlgorithm.h"
#include "libGeneticSvm/SvmWorkflowConfigStruct.h"
#include "libGeneticSvm/Timer.h"
#include "libGeneticSvm/IGeneticWorkflow.h"
#include "libGeneticSvm/GeneticWorkflowResultLogger.h"
#include "libGeneticSvm/LocalFileDatasetLoader.h"
#include "libStrategies/FileSinkStrategy.h"
#include "libSvmStrategies/CreateSvmVisualizationStrategy.h"
#include "libGeneticStrategies/CreatePopulationStrategy.h"
#include "libGeneticStrategies/SelectionStrategy.h"
#include "libGeneticStrategies/MutationStrategy.h"
#include "libGeneticStrategies/CrossoverStrategy.h"
#include "libGeneticStrategies/StopConditionStrategy.h"
#include "libSvmComponents/SvmValidationStrategy.h"
#include "libSvmStrategies/SvmTrainingStrategy.h"

namespace genetic
{
class GaSvmWorkflow : public ISvmAlgorithm, public ITrainingSetOptimizationWorkflow<svmComponents::SvmTrainingSetChromosome>
{
public:
    explicit GaSvmWorkflow(const SvmWokrflowConfiguration& config,
                           svmComponents::GeneticTrainingSetEvolutionConfiguration algorithmConfig,
                           IDatasetLoader& workflow);

    std::shared_ptr<phd::svm::ISvm> run() override;

    void runGeneticAlgorithm() override;
    void initialize() override;
    void setupKernelParameters(const svmComponents::SvmKernelChromosome& kernelParameters) override;
    GeneticWorkflowResultLogger& getResultLogger() override;

    svmComponents::SvmTrainingSetChromosome getBestChromosomeInGeneration() const override;
    geneticComponents::Population<svmComponents::SvmTrainingSetChromosome> getPopulation() const override;
    dataset::Dataset<std::vector<float>, float> getBestTrainingSet() const override;
    void setupFeaturesSet(const svmComponents::SvmFeatureSetChromosome& featureSetChromosome) override;

private:
    using SvmTrainingSetChromosome = svmComponents::SvmTrainingSetChromosome;

    void logResults(const geneticComponents::Population<SvmTrainingSetChromosome>& population,
                    const geneticComponents::Population<SvmTrainingSetChromosome>& testPopulation);
    void initializeGeneticAlgorithm();

public:
    geneticComponents::Population<svmComponents::SvmTrainingSetChromosome> initNoEvaluate(int /*popSize*/) override { throw; }
    void performGeneticOperations(geneticComponents::Population<svmComponents::SvmTrainingSetChromosome>& /*population*/) override{}
    unsigned int getInitialTrainingSetSize() override;
   
    void setK(unsigned int /*k*/) override
    {
       throw std::runtime_error("Not implemented K setting in GaSvmWorkflow");
    }

    phd::svm::ISvm& getClassifierWithBestDistances() override
    {

        throw std::runtime_error("Not implemented getClassifierWithBestDistances in GaSvmWorkflow");
    }

    unsigned getCurrentTrainingSetSize() override
    {
    	//it does not change over time
        return getInitialTrainingSetSize();
    }

    geneticComponents::Population<svmComponents::SvmTrainingSetChromosome> initNoEvaluate(int /*popSize*/, int /*seed*/) override
    {
        throw std::runtime_error("Not implemented initNoEvaluate with seed in GaSvmWorkflow");
    }
	
private:
    svmComponents::GeneticTrainingSetEvolutionConfiguration m_algorithmConfig;

    svmStrategies::SvmTrainingStrategy<SvmTrainingSetChromosome> m_trainingSvmClassifierElement;
    std::shared_ptr<svmStrategies::IValidationStrategy<SvmTrainingSetChromosome>> m_validationElement;
    svmStrategies::SvmValidationStrategy<SvmTrainingSetChromosome> m_validationTestDataElement;
    geneticStrategies::StopConditionStrategy<SvmTrainingSetChromosome> m_stopConditionElement;
    geneticStrategies::CrossoverStrategy<SvmTrainingSetChromosome> m_crossoverElement;
    geneticStrategies::MutationStrategy<SvmTrainingSetChromosome> m_mutationElement;
    geneticStrategies::SelectionStrategy<SvmTrainingSetChromosome> m_selectionElement;
    geneticStrategies::CreatePopulationStrategy<SvmTrainingSetChromosome> m_createPopulationElement;
    strategies::FileSinkStrategy m_savePngElement;
    svmStrategies::CreateSvmVisualizationStrategy<SvmTrainingSetChromosome> m_createVisualizationElement;
    geneticStrategies::CrossoverParentSelectionStrategy<SvmTrainingSetChromosome> m_crossoverParentSelectionElement;


    std::shared_ptr<svmComponents::ITrainingSet> m_trainingSetInterface;
	
    dataset::Dataset<std::vector<float>, float> m_trainingSet2;
    dataset::Dataset<std::vector<float>, float> m_validationSet2;
    dataset::Dataset<std::vector<float>, float> m_testSet2;

    const dataset::Dataset<std::vector<float>, float>* m_trainingSet;
    const dataset::Dataset<std::vector<float>, float>* m_validationSet;
    const dataset::Dataset<std::vector<float>, float>* m_testSet;
    geneticComponents::Population<SvmTrainingSetChromosome> m_population;
    std::filesystem::path m_pngNameSource;

    bool m_needRetrain;
    IDatasetLoader& m_loadingWorkflow;
    Timer m_timer;
    unsigned int m_generationNumber;
    
    const SvmWokrflowConfiguration m_config;
    GeneticWorkflowResultLogger m_resultLogger;

    static constexpr const char* m_algorithmName = "GASVM ";
};
} // namespace genetic
