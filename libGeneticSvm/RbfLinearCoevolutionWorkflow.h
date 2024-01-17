#pragma once

#include "libGeneticSvm/ISvmAlgorithm.h"
#include "libGeneticSvm/CombinedAlgorithmsConfig.h"
#include "libSvmComponents/RbfLinearInternals.h"
#include "libGeneticStrategies/CombinePopulationsStrategy.h"

namespace genetic
{
class CoevolutionHelper;
using SvmCustomKernelChromosome = svmComponents::SvmCustomKernelChromosome;


class RbfLinearCoevolutionWorkflow final : public ISvmAlgorithm
{
public:
	RbfLinearCoevolutionWorkflow(const SvmWokrflowConfiguration& config,
	                             svmComponents::RbfLinearConfig algorithmConfig,
	                             IDatasetLoader& workflow,
	                             platform::Subtree subtreeConfig);

	std::shared_ptr<phd::svm::ISvm> run() override;

private:
	std::vector<std::shared_ptr<CoevolutionHelper>> m_population;

	SvmWokrflowConfiguration m_config;
	svmComponents::RbfLinearConfig m_algorithmConfig;
	IDatasetLoader& m_workflow;
	platform::Subtree m_subtreeConfig;

	static constexpr const char* m_algorithmName = "Rbf Linear Coevolution";
};

class CoevolutionHelper
{
public:
	CoevolutionHelper(const SvmWokrflowConfiguration& config,
	                  svmComponents::RbfLinearConfig algorithmConfig,
	                  IDatasetLoader& workflow, std::string algorithmName, platform::Subtree subtreeConfig);
	void logToFile();

	std::shared_ptr<phd::svm::ISvm> run();

	void setC(double C);

	void setGamma(double gamma);

	std::shared_ptr<phd::svm::ISvm> getSolution()
	{
		return m_population.getBestOne().getClassifier();
	}

	SvmCustomKernelChromosome getBest()
	{
		return m_population.getBestOne();
	}

	double getAverageFitness()
	{
		return m_population.getMeanFitness();
	}
	
	void initializeGeneticAlgorithm();

	bool shrinkTrainingSet(geneticComponents::Population<SvmCustomKernelChromosome>& best_pop);
	void addToFrozenSet();
	void shrinkTrainingSetComplete();
	void getGammaRangeRbfLinear();
	bool improvementAfterInit();
	bool improvementAfeterAlgorithm();
	void setInitialBest();
	void backupForNoImprovement();
	bool earlyStopAndPreviousBackup();

	void savePreviousIterFitness();

	void getGammasFromGridSearch();

	bool gammasHasEnded()
	{
		//return m_hasEnded;
		return m_i == m_gammaRange.size();
	}

	int m_i;

	double getCurrentGamma()
	{
		if (m_i == m_gammaRange.size())
			m_hasEnded = true;
		
		return m_gammaRange[m_i++];
	}
	
	void runGeneticAlgorithm();

	//void shrinkValidationSet();

	void logResults(const geneticComponents::Population<SvmCustomKernelChromosome>& population,
	                const geneticComponents::Population<SvmCustomKernelChromosome>& testPopulation);
	void initForGamma();

	void internalVisualization_Debugging(geneticComponents::Population<svmComponents::SvmCustomKernelChromosome> pop);
	void initMemetic(bool extendOnRbfToLinear = false);

	void memeticAlgorithm();

	bool memeticAlgorithmSingleIteration();

	void switchMetric();

	int getPopulationSize();
	void setPopulationSize(int popSize);

	std::string m_algorithmName;
	
private:
	void visualizeFrozenSet(geneticComponents::Population<SvmCustomKernelChromosome>& best_pop);


	platform::Subtree m_subtreeConfig;

	std::shared_ptr<svmComponents::ISvmMetricsCalculator> m_estimationMethod;
	svmComponents::RbfLinearConfig m_algorithmConfig;

	SvmWokrflowConfiguration m_config;
	std::filesystem::path m_resultFilePath;

	IDatasetLoader& m_loadingWorkflow;
	Timer m_timer;

	dataset::Dataset<std::vector<float>, float> m_trainingSet2;
	dataset::Dataset<std::vector<float>, float> m_validationSet2;
	dataset::Dataset<std::vector<float>, float> m_testSet2;

	const dataset::Dataset<std::vector<float>, float>* m_trainingSet;
	const dataset::Dataset<std::vector<float>, float>* m_validationSet;
	const dataset::Dataset<std::vector<float>, float>* m_testSet;

	//MEMETIC
	//svmComponents::MemeticTrainingSetEvolutionConfiguration m_algorithmConfig;
	svmStrategies::SvmTrainingStrategy<SvmCustomKernelChromosome> m_trainingSvmClassifierElement;
	std::shared_ptr<svmStrategies::IValidationStrategy<SvmCustomKernelChromosome>> m_valdiationElement;
	std::shared_ptr<svmStrategies::SvmValidationStrategy<SvmCustomKernelChromosome>> m_valdiationTestDataElement;
	std::shared_ptr<svmStrategies::IValidationStrategy<SvmCustomKernelChromosome>> m_validationSuperIndividualsElement;
	geneticStrategies::StopConditionStrategy<SvmCustomKernelChromosome> m_stopConditionElement;
	geneticStrategies::CrossoverStrategy<SvmCustomKernelChromosome> m_crossoverElement;
	geneticStrategies::MutationStrategy<SvmCustomKernelChromosome> m_mutationElement;
	geneticStrategies::SelectionStrategy<SvmCustomKernelChromosome> m_selectionElement;
	geneticStrategies::CreatePopulationStrategy<SvmCustomKernelChromosome> m_createPopulationElement;
	strategies::FileSinkStrategy m_savePngElement;
	svmStrategies::CreateSvmVisualizationStrategy<SvmCustomKernelChromosome> m_createVisualizationElement;
	std::shared_ptr<svmComponents::EducationOfTrainingSetRbfLinear> m_educationElement;
	svmComponents::CrossoverCompensationRbfLinear m_crossoverCompensationElement;
	svmComponents::MemeticTrainingSetAdaptationRbfLinear m_adaptationElement;
	svmComponents::SupportVectorPoolRbfLinear m_supportVectorPoolElement;
	std::shared_ptr<svmComponents::SuperIndividualsCreationRbfLinear> m_superIndividualsGenerationElement;
	geneticStrategies::CombinePopulationsStrategy<SvmCustomKernelChromosome> m_populationCombinationElement;
	svmStrategies::SvmTrainingStrategy<SvmCustomKernelChromosome> m_trainingSuperIndividualsElement;
	geneticStrategies::CrossoverParentSelectionStrategy<SvmCustomKernelChromosome> m_parentSelectionElement;
	svmComponents::CompensationInformationRbfLinear m_compensationGenerationElement;

	std::vector<svmComponents::Gene> m_svPool;
	unsigned int m_numberOfClassExamples;
	unsigned int m_initialNumberOfClassExamples;

	GeneticWorkflowResultLogger m_resultLogger;

	std::vector<svmComponents::Gene> m_frozenSV;
	std::unordered_set<svmComponents::Gene> m_frozenSV_ids;

	std::unordered_set<uint64_t> m_forbidden_set;
	std::vector<double> m_gammaRange;
	double m_CValue;

	geneticComponents::Population<svmComponents::SvmCustomKernelChromosome> m_population;

	std::filesystem::path m_pngNameSource;

	std::once_flag m_increasAfterKernelSwitch;

	unsigned int m_generationNumber;
	bool m_shrinkTrainingSet;
	bool m_hasEnded;

	double m_previousGoodSvRatio;
	double m_currentGamma;
	double m_initialBest;

	geneticComponents::Population<svmComponents::SvmCustomKernelChromosome> m_popBackup;
	geneticComponents::Population<svmComponents::SvmCustomKernelChromosome> m_popBackupTestScore;
	double m_previousIterationFitness;


	//logger::LogFrontend m_logger;
};
} // namespace genetic
