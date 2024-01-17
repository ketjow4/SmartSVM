#pragma once

#include <filesystem>
#include <libSvmComponents/SvmConfigStructures.h>
#include "libGeneticSvm/ISvmAlgorithm.h"
#include "libGeneticSvm/SvmWorkflowConfigStruct.h"
#include "IDatasetLoader.h"
#include "Timer.h"
#include "GeneticWorkflowResultLogger.h"

#include "libStrategies/FileSinkStrategy.h"

#include "libGeneticComponents/IPopulationGeneration.h"
#include "libSvmStrategies/CreateSvmVisualizationStrategy.h"
#include "libSvmComponents/SvmValidationStrategy.h"
#include "libSvmStrategies/GridSearchPopulationGenerationStrategy.h"
#include "libSvmStrategies/GridSearchCrossValidationStrategy.h"

namespace genetic
{
class GridSearchWorkflow : public ISvmAlgorithm
{
public:
    explicit GridSearchWorkflow(const SvmWokrflowConfiguration& config,
                                svmComponents::GridSearchConfiguration&& algorithmConfig,
                                IDatasetLoader& workflow);

    std::shared_ptr<phd::svm::ISvm> run() override;

	std::vector<double>& getC() { return m_goodC; }
	std::vector<double>& getGammas() { return m_goodGammas; }

    std::shared_ptr<phd::svm::ISvm> runWithGeneration(geneticComponents::IPopulationGeneration<svmComponents::SvmTrainingSetChromosome>& populationGeneration);

    void switchMetric(std::shared_ptr<svmComponents::ISvmMetricsCalculator> metric);
	
private:
    void internalVisualization_Debugging(geneticComponents::Population<svmComponents::SvmKernelChromosome> pop);
    void getValuesForCustomKernel(geneticComponents::Population<svmComponents::SvmKernelChromosome> populationValidationFitness);
    void doGridSearch(const dataset::Dataset<std::vector<float>, float>& trainingSubset);
    void runGridSearch();

    svmComponents::GridSearchConfiguration m_algorithmConfig;

    std::shared_ptr < svmStrategies::SvmValidationStrategy<svmComponents::SvmKernelChromosome>> m_valdiationElement;
    std::shared_ptr < svmStrategies::SvmValidationStrategy<svmComponents::SvmKernelChromosome>> m_valdiationTestDataElement;
    svmStrategies::GridSearchCrossValidationStrategy m_gridSearchCrossValidationElement;
    svmStrategies::GridSearchPopulationGenerationStrategy m_createGridPopulationElement;
    strategies::FileSinkStrategy m_savePngElement;
    svmStrategies::CreateSvmVisualizationStrategy<svmComponents::SvmKernelChromosome> m_createVisualizationElement;

    const dataset::Dataset<std::vector<float>, float>* m_trainingSet;
    const dataset::Dataset<std::vector<float>, float>* m_validationSet;
    const dataset::Dataset<std::vector<float>, float>* m_testSet;
    geneticComponents::Population<svmComponents::SvmKernelChromosome> m_population;
    std::filesystem::path m_pngNameSource;

    IDatasetLoader& m_loadingWorkflow;
    Timer m_timer;
    unsigned int m_iterationNumber;
    const SvmWokrflowConfiguration m_config;
    //logger::LogFrontend m_logger;
    GeneticWorkflowResultLogger m_resultLogger;

	std::vector<double> m_goodGammas;
	std::vector<double> m_goodC;

    static constexpr const char* m_algorithmName = "GridSearch";
};
} // namespace genetic
