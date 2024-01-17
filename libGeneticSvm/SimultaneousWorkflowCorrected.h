#pragma once

#include "libGeneticSvm/ISvmAlgorithm.h"
#include "libGeneticSvm/CombinedAlgorithmsConfig.h"

namespace genetic
{
class SimultaneousWorkflowCorrected : public ISvmAlgorithm
{
public:
	SimultaneousWorkflowCorrected(const SvmWokrflowConfiguration& config,
	                              SimultaneousWorkflowConfig algorithmConfig,
	                              IDatasetLoader& workflow);

	void logAllModels(AllModelsLogger& logger);

	std::shared_ptr<phd::svm::ISvm> run() override;

private:
	void init();

	void performEvolution();

	void evaluate();

	bool isFinished();

	void log();

	void fixKernelPop(const geneticComponents::Population<svmComponents::SvmSimultaneousChromosome>& currentPop,
		geneticComponents::Population<svmComponents::SvmSimultaneousChromosome>& newPop);

	Population<SvmKernelChromosome> getKernelPopulation();
	Population<SvmFeatureSetMemeticChromosome> getFeaturesPopulation();
	Population<SvmTrainingSetChromosome> getTrainingSetPopulation();

	void setKernelPopulation(Population<SvmKernelChromosome>& population,
	                         Population<SvmSimultaneousChromosome>& newPop);
	void setTrainingPopulation(Population<SvmTrainingSetChromosome>& population,
	                           Population<SvmSimultaneousChromosome>& newPop);
	void setFeaturesPopulation(Population<SvmFeatureSetMemeticChromosome>& population,
	                           Population<SvmSimultaneousChromosome>& newPop);

	TrainingSetOptimizationWorkflow m_trainingSetOptimization;
	KernelOptimizationWorkflow m_kernelOptimization;
	FeatureSetOptimizationWorkflow m_featureSetOptimization;
	std::filesystem::path m_resultFilePath;
	SimultaneousWorkflowConfig m_algorithmConfig;

	svmStrategies::SvmTrainingStrategy<SvmSimultaneousChromosome> m_svmTraining;
	Population<SvmSimultaneousChromosome> m_pop;
	Population<SvmSimultaneousChromosome> m_popTestSet;

	svmStrategies::SvmValidationStrategy<SvmSimultaneousChromosome> m_validation;
	svmStrategies::SvmValidationStrategy<SvmSimultaneousChromosome> m_validationTest;
	geneticStrategies::StopConditionStrategy<SvmSimultaneousChromosome> m_stopConditionElement;
	geneticStrategies::SelectionStrategy<SvmSimultaneousChromosome> m_selectionElement;

	IDatasetLoader& m_workflow;
	SvmWokrflowConfiguration m_config;

	GeneticWorkflowResultLogger m_resultLogger;
	std::shared_ptr<Timer> m_timer;
	unsigned int m_generationNumber;

	dataset::Dataset<std::vector<float>, float> m_trainingSet;
	dataset::Dataset<std::vector<float>, float> m_validationSet;
	dataset::Dataset<std::vector<float>, float> m_testSet;

	static constexpr const char* m_algorithmName = "SSVM"; //TODO change name in here
};
} // namespace genetic
