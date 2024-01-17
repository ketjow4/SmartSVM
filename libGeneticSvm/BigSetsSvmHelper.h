#pragma once

#include "libSvmComponents/GeneticAlgorithmsConfigs.h"
#include "libGeneticSvm/ISvmAlgorithm.h"
#include "libGeneticSvm/SvmWorkflowConfigStruct.h"
#include "libGeneticSvm/Timer.h"
#include "libGeneticSvm/IDatasetLoader.h"

#include "libStrategies/FileSinkStrategy.h"
#include "AllModelsLogger.h"
#include "libGeneticSvm/CombinedAlgorithmsConfig.h"
#include "libSvmComponents/SvmHyperplaneDistance.h"

namespace genetic
{
	using namespace svmComponents;


	class BigSetsSvmHelper
	{
	public:
		BigSetsSvmHelper(const SvmWokrflowConfiguration& config,
			EnsembleTreeWorkflowConfig algorithmConfig,
			IDatasetLoader& workflow,
			bool addSvsToTraining,
			std::vector<DatasetVector>& SVs,
			IDatasetLoader& fullDatasetworkflow,
			std::vector<Gene> SvWithGamma,
			bool useDasvmKernel,
			bool debugLog,
			bool useFeatureSelection,
			platform::Subtree full_config,
			bool cascadeWideFeatureSelection,
			bool newDatasetFlow = false);

		void logAllModels(AllModelsLogger& logger);
		void VisualizeWholePopulation(unsigned& numberOfRun);

		std::shared_ptr<phd::svm::ISvm> run();

		svmComponents::SvmTrainingSetChromosome getBestOne() const;

		void addVectorsToPopulation(std::vector<DatasetVector>& SVs);

	private:
		void init();

		void performEvolution();
		void train(Population<SvmSimultaneousChromosome>& pop);

		void evaluate(Population<SvmSimultaneousChromosome>& pop);

		bool isFinished();

		void log();

		[[deprecated]]
		void switchMetric();

		void updateMetricDistance();

		//template <class chromosome>
		//void clearlog(IGeneticWorkflow<chromosome>& workflow);
		//
		void fixKernelPop2(const geneticComponents::Population<svmComponents::SvmSimultaneousChromosome>& currentPop,
			geneticComponents::Population<svmComponents::SvmSimultaneousChromosome>& newPop);

		svmComponents::SvmKernelChromosome m_constKernel;
		bool m_useConstKernel;
		bool m_addSvToTraining;
		std::vector<DatasetVector> m_svToAdd;
		std::vector<Gene> m_svFrozenPool;
		bool m_useDasvmKernel;
		bool m_debugLog;
		bool m_useFeatureSelection;
		bool m_newDatasetFlow;
		platform::Subtree m_full_config;
		bool m_cascadeWideFeatureSelection;

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
		EnsembleTreeWorkflowConfig m_algorithmConfig;

		svmStrategies::SvmTrainingStrategy<svmComponents::SvmSimultaneousChromosome> m_svmTraining;
		geneticComponents::Population<svmComponents::SvmSimultaneousChromosome> m_pop;
		geneticComponents::Population<svmComponents::SvmSimultaneousChromosome> m_popTestSet;

		/*svmStrategies::SvmValidationStrategy<svmComponents::SvmSimultaneousChromosome> m_validation;
		svmStrategies::SvmValidationStrategy<svmComponents::SvmSimultaneousChromosome> m_validationTest;*/

		std::shared_ptr<svmComponents::ISvmMetricsCalculator> m_estimationMethod;
		std::shared_ptr<svmStrategies::IValidationStrategy<svmComponents::SvmSimultaneousChromosome>> m_validation;
		std::shared_ptr<svmStrategies::SvmValidationStrategy<svmComponents::SvmSimultaneousChromosome>> m_validationTest;

		geneticStrategies::StopConditionStrategy<svmComponents::SvmSimultaneousChromosome> m_stopConditionElement;
		geneticStrategies::SelectionStrategy<svmComponents::SvmSimultaneousChromosome> m_selectionElement;

		strategies::FileSinkStrategy m_savePngElement;

		IDatasetLoader& m_workflow;
		IDatasetLoader& m_fullDatasetWorkflow;
		SvmWokrflowConfiguration m_config;

		GeneticWorkflowResultLogger m_resultLogger;
		Timer m_timer;
		unsigned int m_generationNumber;

		unsigned int m_nodeNumber;

		const dataset::Dataset<std::vector<float>, float>& m_trainingSet;
		const dataset::Dataset<std::vector<float>, float>& m_validationSet;
		const dataset::Dataset<std::vector<float>, float>& m_testSet;

		static constexpr const char* m_algorithmName = "BigSetsSvmHelper";
	};
} // namespace genetic
