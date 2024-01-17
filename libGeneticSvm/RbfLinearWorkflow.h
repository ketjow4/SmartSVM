#pragma once

#include "libGeneticSvm/ISvmAlgorithm.h"
#include "libGeneticSvm/CombinedAlgorithmsConfig.h"
#include "libSvmComponents/RbfLinearInternals.h"
#include "libGeneticStrategies/CombinePopulationsStrategy.h"

namespace genetic
{
	class RbfLinearWorkflow final : public ISvmAlgorithm
	{
	public:
		RbfLinearWorkflow(const SvmWokrflowConfiguration& config,
			svmComponents::RbfLinearConfig algorithmConfig,
			IDatasetLoader& workflow);

		std::shared_ptr<phd::svm::ISvm> run() override;

		void setC(double C) override;

	private:
		using SvmCustomKernelChromosome = svmComponents::SvmCustomKernelChromosome;

		void initializeGeneticAlgorithm();

		void visualizeFrozenSet(geneticComponents::Population<SvmCustomKernelChromosome>& best_pop);
		bool shrinkTrainingSet(geneticComponents::Population<SvmCustomKernelChromosome>& best_pop);
		
		void runGeneticAlgorithm();

		//void shrinkValidationSet();

		void logResults(const geneticComponents::Population<SvmCustomKernelChromosome>& population,
			const geneticComponents::Population<SvmCustomKernelChromosome>& testPopulation);

		void internalVisualization_Debugging(geneticComponents::Population<svmComponents::SvmCustomKernelChromosome> pop);
		void initMemetic();
		void memeticAlgorithm();

		void switchMetric();

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

		double m_previousGoodSvRatio;
		double m_currentGamma;

		static constexpr const char* m_algorithmName = "Rbf Linear";

		//logger::LogFrontend m_logger;
	};
} // namespace genetic
