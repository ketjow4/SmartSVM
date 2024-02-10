#pragma once

#include "libSvmComponents/GeneticAlgorithmsConfigs.h"
#include "libSvmComponents/SvmKernelChromosome.h"
#include "libGeneticSvm/ISvmAlgorithm.h"
#include "libGeneticSvm/SvmWorkflowConfigStruct.h"
#include "libGeneticSvm/Timer.h"
#include "libGeneticSvm/IGeneticWorkflow.h"
#include "libGeneticSvm/GeneticWorkflowResultLogger.h"
#include "libGeneticSvm/IDatasetLoader.h"

#include "libStrategies/FileSinkStrategy.h"
#include "libGeneticStrategies/CreatePopulationStrategy.h"
#include "libSvmComponents/SvmValidationStrategy.h"
#include "libGeneticStrategies/StopConditionStrategy.h"
#include "libGeneticStrategies/CrossoverStrategy.h"
#include "libGeneticStrategies/MutationStrategy.h"
#include "libGeneticStrategies/SelectionStrategy.h"
#include "libSvmStrategies/SvmTrainingStrategy.h"
#include "libSvmStrategies/CreateSvmVisualizationStrategy.h"
#include "libGeneticStrategies/CrossoverParentSelectionStrategy.h"
#include "AllModelsLogger.h"

namespace genetic
{
class GeneticKernelEvolutionWorkflow : public ISvmAlgorithm, public IKernelOptimalizationWorkflow<svmComponents::SvmKernelChromosome>
{
public:
	explicit GeneticKernelEvolutionWorkflow(const SvmWokrflowConfiguration& config,
	                                        svmComponents::GeneticKernelEvolutionConfiguration algorithmConfig,
	                                        IDatasetLoader& workflow,
	                                        platform::Subtree fullConfig);

	std::shared_ptr<phd::svm::ISvm> run() override;
	void logAllModels(geneticComponents::Population<svmComponents::SvmKernelChromosome>& population,
	                  geneticComponents::Population<svmComponents::SvmKernelChromosome>& testPopulation);

	void runGeneticAlgorithm() override;
	void initialize() override;
	void setupTrainingSet(const dataset::Dataset<std::vector<float>, float>& trainingSet) override;
	GeneticWorkflowResultLogger& getResultLogger() override;

	svmComponents::SvmKernelChromosome getBestChromosomeInGeneration() const override;
	geneticComponents::Population<svmComponents::SvmKernelChromosome> getPopulation() const override;

	void setDatasets(const dataset::Dataset<std::vector<float>, float>& trainingSet,
	                 const dataset::Dataset<std::vector<float>, float>& validationSet,
	                 const dataset::Dataset<std::vector<float>, float>& testSet) override;

	geneticComponents::Population<svmComponents::SvmKernelChromosome> initNoEvaluate(int popSize) override;
	void performGeneticOperations(geneticComponents::Population<svmComponents::SvmKernelChromosome>& population) override;

	void setTimer(std::shared_ptr<Timer> timer) override;

	geneticComponents::Population<svmComponents::SvmKernelChromosome> initNoEvaluate(int popSize, int seed) override;

private:
	using SvmKernelChromosome = svmComponents::SvmKernelChromosome;

	void initializeGeneticAlgorithm();
	void logResults(const geneticComponents::Population<SvmKernelChromosome>& population,
	                const geneticComponents::Population<SvmKernelChromosome>& testPopulation);

private:
	platform::Subtree m_fullConfig;

	svmComponents::GeneticKernelEvolutionConfiguration m_algorithmConfig;
	svmStrategies::SvmTrainingStrategy<SvmKernelChromosome> m_trainingSvmClassifierElement;
	svmStrategies::SvmValidationStrategy<SvmKernelChromosome> m_valdiationElement;
	svmStrategies::SvmValidationStrategy<SvmKernelChromosome> m_valdiationTestDataElement;
	geneticStrategies::StopConditionStrategy<SvmKernelChromosome> m_stopConditionElement;
	geneticStrategies::CrossoverStrategy<SvmKernelChromosome> m_crossoverElement;
	geneticStrategies::MutationStrategy<SvmKernelChromosome> m_mutationElement;
	geneticStrategies::SelectionStrategy<SvmKernelChromosome> m_selectionElement;
	geneticStrategies::CreatePopulationStrategy<SvmKernelChromosome> m_createPopulationElement;
	strategies::FileSinkStrategy m_savePngElement;
	svmStrategies::CreateSvmVisualizationStrategy<SvmKernelChromosome> m_createVisualizationElement;
	geneticStrategies::CrossoverParentSelectionStrategy<SvmKernelChromosome> m_crossoverParentSelectionElement;

	dataset::Dataset<std::vector<float>, float> m_trainingSet222222;
	dataset::Dataset<std::vector<float>, float> m_validationSet2;
	dataset::Dataset<std::vector<float>, float> m_testSet2;

	const dataset::Dataset<std::vector<float>, float>* m_trainingSet;
	const dataset::Dataset<std::vector<float>, float>* m_validationSet;
	const dataset::Dataset<std::vector<float>, float>* m_testSet;
	std::filesystem::path m_pngNameSource;
	geneticComponents::Population<SvmKernelChromosome> m_population;

	bool m_needRetrain;
	IDatasetLoader& m_loadingWorkflow;
	std::shared_ptr<Timer> m_timer;
	unsigned int m_generationNumber;

	const SvmWokrflowConfiguration m_config;
	GeneticWorkflowResultLogger m_resultLogger;
	std::shared_ptr<AllModelsLogger> m_allModelsLogger;

	static constexpr const char* m_algorithmName = "Kernel";
};
} // namespace genetic
