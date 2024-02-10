#pragma once

#include "libGeneticSvm/ISvmAlgorithm.h"
#include "libGeneticSvm/CombinedAlgorithmsConfig.h"
#include "libSvmComponents/MultipleGammaInternals.h"
#include "libGeneticStrategies/CombinePopulationsStrategy.h"

namespace genetic
{
	class MultipleGammaMASVMWorkflow final : public ISvmAlgorithm
	{
	public:
		MultipleGammaMASVMWorkflow(const SvmWokrflowConfiguration& config,
			svmComponents::MutlipleGammaMASVMConfig algorithmConfig,
			IDatasetLoader& workflow);

		std::shared_ptr<phd::svm::ISvm> run() override;

	private:
		using SvmCustomKernelChromosome = svmComponents::SvmCustomKernelChromosome;

		void initializeGeneticAlgorithm();

		
		void runGeneticAlgorithm();

		void logResults(const geneticComponents::Population<SvmCustomKernelChromosome>& population,
			const geneticComponents::Population<SvmCustomKernelChromosome>& testPopulation);

		void initMemetic();
		void memeticAlgorithm();

		svmComponents::MutlipleGammaMASVMConfig m_algorithmConfig;

		const SvmWokrflowConfiguration m_config;
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
		svmStrategies::SvmValidationStrategy<SvmCustomKernelChromosome> m_valdiationElement;
		svmStrategies::SvmValidationStrategy<SvmCustomKernelChromosome> m_valdiationTestDataElement;
		geneticStrategies::StopConditionStrategy<SvmCustomKernelChromosome> m_stopConditionElement;
		geneticStrategies::CrossoverStrategy<SvmCustomKernelChromosome> m_crossoverElement;
		geneticStrategies::MutationStrategy<SvmCustomKernelChromosome> m_mutationElement;
		geneticStrategies::SelectionStrategy<SvmCustomKernelChromosome> m_selectionElement;
		geneticStrategies::CreatePopulationStrategy<SvmCustomKernelChromosome> m_createPopulationElement;
		strategies::FileSinkStrategy m_savePngElement;
		svmStrategies::CreateSvmVisualizationStrategy<SvmCustomKernelChromosome> m_createVisualizationElement;
		std::shared_ptr<svmComponents::MultipleGammaEducationOfTrainingSet> m_educationElement;
		svmComponents::MultipleGammaCrossoverCompensation m_crossoverCompensationElement;
		svmComponents::MultipleGammaMemeticTrainingSetAdaptation m_adaptationElement;
		svmComponents::MultipleGammaSupportVectorPool m_supportVectorPoolElement;
		std::shared_ptr<svmComponents::MultipleGammaSuperIndividualsCreation> m_superIndividualsGenerationElement;
		geneticStrategies::CombinePopulationsStrategy<SvmCustomKernelChromosome> m_populationCombinationElement;
		svmStrategies::SvmTrainingStrategy<SvmCustomKernelChromosome> m_trainingSuperIndividualsElement;
		svmStrategies::SvmValidationStrategy<SvmCustomKernelChromosome> m_validationSuperIndividualsElement;
		geneticStrategies::CrossoverParentSelectionStrategy<SvmCustomKernelChromosome> m_parentSelectionElement;
		svmComponents::MultipleGammaCompensationInformation m_compensationGenerationElement;

		std::vector<svmComponents::Gene> m_svPool;
		unsigned int m_numberOfClassExamples;		

		GeneticWorkflowResultLogger m_resultLogger;

		std::vector<svmComponents::Gene> m_frozenSV;
		std::unordered_set<svmComponents::Gene> m_frozenSV_ids;
		std::vector<double> m_gammaRange;
		double m_CValue;

		geneticComponents::Population<svmComponents::SvmCustomKernelChromosome> m_population;
		std::filesystem::path m_pngNameSource;

		unsigned int m_generationNumber;
		bool m_shrinkTrainingSet;

		static constexpr const char* m_algorithmName = "Multiple Gamma MASVM";
	};
} // namespace genetic
