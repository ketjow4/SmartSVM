#pragma once

#include "libSvmComponents/SvmFeatureSetChromosome.h"
#include "LibGeneticComponents/BinaryChromosomeCache.h"
#include "libGeneticSvm/ISvmAlgorithm.h"
#include "libGeneticSvm/CombinedAlgorithmsConfig.h"

#include "libGeneticStrategies/AddToBinaryCacheStrategy.h"
#include "libGeneticStrategies/UseBinaryCacheStrategy.h"
#include "libGeneticStrategies/CombinePopulationsStrategy.h"

namespace genetic
{
class FeatureSelectionWorkflow : public ISvmAlgorithm, public IFeatureSelectionWorkflow<svmComponents::SvmFeatureSetChromosome>
{
public:
    FeatureSelectionWorkflow(const SvmWokrflowConfiguration& config,
                             svmComponents::GeneticFeatureSetEvolutionConfiguration algorithmConfig,
                             IDatasetLoader& loadingWorkflow);

    std::shared_ptr<phd::svm::ISvm> run() override;
    void initialize() override;

    void runGeneticAlgorithm() override;
    svmComponents::SvmFeatureSetChromosome getBestChromosomeInGeneration() const override;
    geneticComponents::Population<svmComponents::SvmFeatureSetChromosome> getPopulation() const override;
    GeneticWorkflowResultLogger& getResultLogger() override;

    void setupKernelParameters(const svmComponents::SvmKernelChromosome& kernelParameters) override;
    dataset::Dataset<std::vector<float>, float> getFilteredTraningSet() override;
    dataset::Dataset<std::vector<float>, float> getFilteredValidationSet() override;
    dataset::Dataset<std::vector<float>, float> getFilteredTestSet() override;
    void setupTrainingSet(const dataset::Dataset<std::vector<float>, float>& trainingSet) override;

private:
    using SvmFeatureSetChromosome = svmComponents::SvmFeatureSetChromosome;

    void initializeGeneticAlgorithm();
    void logResult(const geneticComponents::Population<SvmFeatureSetChromosome>& population,
                   const geneticComponents::Population<SvmFeatureSetChromosome>& testpopulation);

public:
    geneticComponents::Population<svmComponents::SvmFeatureSetChromosome> initNoEvaluate(int /*popSize*/) override
    {
        throw;
    }
    void performGeneticOperations(geneticComponents::Population<svmComponents::SvmFeatureSetChromosome>& /*population*/) override
    {
        
    }

    geneticComponents::Population<svmComponents::SvmFeatureSetChromosome> initNoEvaluate(int /*popSize*/, int /*seed*/) override
    {
        throw;
    }
private:
    svmComponents::GeneticFeatureSetEvolutionConfiguration m_algorithmConfig;

    svmStrategies::SvmTrainingStrategy<SvmFeatureSetChromosome> m_trainingSvmClassifierElement;
    svmStrategies::SvmValidationStrategy<SvmFeatureSetChromosome> m_validationElement;
    svmStrategies::SvmValidationStrategy<SvmFeatureSetChromosome> m_validationTestDataElement;
    geneticStrategies::StopConditionStrategy<SvmFeatureSetChromosome> m_stopConditionElement;
    geneticStrategies::CrossoverStrategy<SvmFeatureSetChromosome> m_crossoverElement;
    geneticStrategies::MutationStrategy<SvmFeatureSetChromosome> m_mutationElement;
    geneticStrategies::SelectionStrategy<SvmFeatureSetChromosome> m_selectionElement;
    geneticStrategies::CreatePopulationStrategy<SvmFeatureSetChromosome> m_createPopulationElement;
    geneticStrategies::CrossoverParentSelectionStrategy<SvmFeatureSetChromosome> m_crossoverParentSelectionElement;

    geneticStrategies::AddToBinaryCacheStrategy<SvmFeatureSetChromosome> m_addToCacheElement;
    geneticStrategies::UseToBinaryCacheStrategy<SvmFeatureSetChromosome> m_useCacheElement;
    geneticStrategies::CombinePopulationsStrategy<SvmFeatureSetChromosome> m_combinePopulationElement;

    //std::shared_ptr<framework::Element> m_addToCacheTestElement;
    //std::shared_ptr<framework::Element> m_useCacheTestElement;
    //std::shared_ptr<framework::Element> m_combinePopulationTestElement;

    dataset::Dataset<std::vector<float>, float> m_trainingSet22222;

    const dataset::Dataset<std::vector<float>, float>* m_trainingSet;
    const dataset::Dataset<std::vector<float>, float>* m_validationSet;
    const dataset::Dataset<std::vector<float>, float>* m_testSet;
    std::filesystem::path m_pngNameSource;
    geneticComponents::Population<SvmFeatureSetChromosome> m_population;

    geneticComponents::BinaryChromosomeCache<svmComponents::SvmFeatureSetChromosome> m_cache;
    geneticComponents::BinaryChromosomeCache<svmComponents::SvmFeatureSetChromosome> m_cacheTestSet;
    bool m_needRetrain;
    IDatasetLoader& m_loadingWorkflow;
    Timer m_timer;
    unsigned int m_generationNumber;

    const SvmWokrflowConfiguration m_config;
    //logger::LogFrontend m_logger;
    GeneticWorkflowResultLogger m_resultLogger;

    static constexpr const char* m_algorithmName = "Feature";
};
} // namespace genetic
