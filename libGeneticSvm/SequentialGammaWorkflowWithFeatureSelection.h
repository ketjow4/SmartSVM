#pragma once

#include "libGeneticSvm/ISvmAlgorithm.h"
#include "libGeneticSvm/CombinedAlgorithmsConfig.h"
#include "libSvmComponents/SequentialGammaInternals.h"
#include "libGeneticStrategies/CombinePopulationsStrategy.h"
#include "libSvmComponents/SvmCustomKernelFeaturesSelectionChromosome.h"

namespace genetic
{
	class SequentialGammaWorkflowWithFeatureSelection final : public ISvmAlgorithm
	{
	public:
		SequentialGammaWorkflowWithFeatureSelection(const SvmWokrflowConfiguration& config,
			SequentialGammaConfigWithFeatureSelection algorithmConfig,
			IDatasetLoader& workflow);

		std::shared_ptr<phd::svm::ISvm> run() override;

	private:
		using SvmCustomKernelChromosome = svmComponents::SvmCustomKernelChromosome;

		void initializeGeneticAlgorithm();

		void visualizeFrozenSet(geneticComponents::Population<SvmCustomKernelChromosome>& best_pop);
		bool shrinkTrainingSet(geneticComponents::Population<SvmCustomKernelChromosome>& best_pop);
		void buildEnsembleFromLastGeneration();
		void runGeneticAlgorithm();

		void shrinkValidationSet();

		void logResults(const geneticComponents::Population<SvmCustomKernelFeaturesSelectionChromosome>& population,
			const geneticComponents::Population<SvmCustomKernelFeaturesSelectionChromosome>& testPopulation);

		void internalVisualization_Debugging(geneticComponents::Population<svmComponents::SvmCustomKernelChromosome> pop);
		void initMemetic();
		void memeticAlgorithm();

		void switchMetric();


		Population<SvmFeatureSetMemeticChromosome> getFullFeaturePop()
		{
			std::vector<SvmFeatureSetMemeticChromosome> fullFeaturePop;


			for(auto j =0u; j < m_algorithmConfig.m_populationSize; ++j) //TODO optimize this
			{ 
				auto featureCount = m_trainingSet->getSample(0).size();
				std::vector<svmComponents::Feature> f;
				for (auto i = 0u; i < featureCount; ++i)
				{
					f.emplace_back(i);
				}
				svmComponents::SvmFeatureSetMemeticChromosome feature(std::move(f));
				fullFeaturePop.emplace_back(feature);
				
			}
				
			return geneticComponents::Population<svmComponents::SvmFeatureSetMemeticChromosome>{ fullFeaturePop };
		}

		
		void mergePopulations()
		{
			//m_populationWithFeatures

			std::vector<SvmCustomKernelFeaturesSelectionChromosome> newPop;

			
			auto featuresPop = getFeaturePop();
			for(auto i = 0u; i < m_population.size(); ++i)
			{
				newPop.emplace_back(m_population[i], featuresPop[i]);
			}

			m_populationWithFeatures = geneticComponents::Population<svmComponents::SvmCustomKernelFeaturesSelectionChromosome>{ newPop };			
		}

		Population<SvmFeatureSetMemeticChromosome> getFeaturePop()
		{
			Population<SvmFeatureSetMemeticChromosome> featuresPop;
			
			if (m_useFeatureOptimization)
			{
				featuresPop = m_featureSetOptimization->getPopulation();

			}
			else
			{
				featuresPop = getFullFeaturePop();
			}
			return featuresPop;
		}

		geneticComponents::Population<svmComponents::SvmCustomKernelFeaturesSelectionChromosome> mergePopulations(Population<svmComponents::SvmCustomKernelChromosome>& popToMerge)
		{
			//m_populationWithFeatures

			std::vector<SvmCustomKernelFeaturesSelectionChromosome> newPop;

			auto featuresPop = getFeaturePop();
			for (auto i = 0u; i < popToMerge.size(); ++i)
			{
				newPop.emplace_back(popToMerge[i], featuresPop[i]);
			}

			return geneticComponents::Population<svmComponents::SvmCustomKernelFeaturesSelectionChromosome>{ newPop };
		}

		void getCustomKernelPopFromMerged()
		{
			std::vector<SvmCustomKernelChromosome> newPop;

			for (auto i = 0u; i < m_populationWithFeatures.size(); ++i)
			{
				newPop.emplace_back(m_populationWithFeatures[i].getKernel());
			}
			m_population = geneticComponents::Population<svmComponents::SvmCustomKernelChromosome>{ newPop };
		}

		Population<svmComponents::SvmCustomKernelChromosome> getCustomKernelPopFromMerged(Population<SvmCustomKernelFeaturesSelectionChromosome>& pop)
		{
			std::vector<SvmCustomKernelChromosome> newPop;

			for (auto i = 0u; i < pop.size(); ++i)
			{
				newPop.emplace_back(pop[i].getKernel());
			}
			return geneticComponents::Population<svmComponents::SvmCustomKernelChromosome>{ newPop };
		}

		Population<svmComponents::SvmFeatureSetMemeticChromosome> getFeaturePopFromMerged(Population<SvmCustomKernelFeaturesSelectionChromosome>& pop)
		{
			if(m_useFeatureOptimization == false)
			{
				return getFullFeaturePop();
			}
			
			std::vector<SvmFeatureSetMemeticChromosome> newPop;

			for (auto i = 0u; i < pop.size(); ++i)
			{
				newPop.emplace_back(pop[i].getFeaturesChromosome());
			}
			return geneticComponents::Population<svmComponents::SvmFeatureSetMemeticChromosome>{ newPop };
		}

		std::shared_ptr<svmComponents::ISvmMetricsCalculator> m_estimationMethod;

		SequentialGammaConfigWithFeatureSelection m_algorithmConfig;

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
		svmStrategies::SvmTrainingStrategy<SvmCustomKernelFeaturesSelectionChromosome> m_trainingSvmClassifierElement;
		std::shared_ptr<svmStrategies::IValidationStrategy<SvmCustomKernelFeaturesSelectionChromosome>> m_valdiationElement;
		std::shared_ptr<svmStrategies::SvmValidationStrategy<SvmCustomKernelFeaturesSelectionChromosome>> m_valdiationTestDataElement;
		std::shared_ptr<svmStrategies::IValidationStrategy<SvmCustomKernelFeaturesSelectionChromosome>> m_validationSuperIndividualsElement;
		geneticStrategies::StopConditionStrategy<SvmCustomKernelFeaturesSelectionChromosome> m_stopConditionElement;
		geneticStrategies::CrossoverStrategy<SvmCustomKernelChromosome> m_crossoverElement;
		geneticStrategies::MutationStrategy<SvmCustomKernelChromosome> m_mutationElement;
		geneticStrategies::SelectionStrategy<SvmCustomKernelFeaturesSelectionChromosome> m_selectionElement;
		geneticStrategies::CreatePopulationStrategy<SvmCustomKernelChromosome> m_createPopulationElement;
		strategies::FileSinkStrategy m_savePngElement;
		svmStrategies::CreateSvmVisualizationStrategy<SvmCustomKernelChromosome> m_createVisualizationElement;
		std::shared_ptr<svmComponents::EducationOfTrainingSetGamma> m_educationElement;
		svmComponents::CrossoverCompensationGamma m_crossoverCompensationElement;
		svmComponents::MemeticTrainingSetAdaptationGamma m_adaptationElement;
		svmComponents::SupportVectorPoolGamma m_supportVectorPoolElement;
		std::shared_ptr<svmComponents::SuperIndividualsCreationGamma> m_superIndividualsGenerationElement;
		geneticStrategies::CombinePopulationsStrategy<SvmCustomKernelFeaturesSelectionChromosome> m_populationCombinationElement;
		svmStrategies::SvmTrainingStrategy<SvmCustomKernelFeaturesSelectionChromosome> m_trainingSuperIndividualsElement;
		geneticStrategies::CrossoverParentSelectionStrategy<SvmCustomKernelChromosome> m_parentSelectionElement;
		svmComponents::CompensationInformationGamma m_compensationGenerationElement;

		std::vector<svmComponents::Gene> m_svPool;
		unsigned int m_numberOfClassExamples;
		unsigned int m_initialNumberOfClassExamples;

		GeneticWorkflowResultLogger m_resultLogger;

		std::vector<svmComponents::Gene> m_frozenSV;
		std::unordered_set<svmComponents::Gene> m_frozenSV_ids;
		std::vector<double> m_gammaRange;
		double m_CValue;

		bool m_useFeatureOptimization;
		
		geneticComponents::Population<svmComponents::SvmCustomKernelChromosome> m_population;

		geneticComponents::Population<svmComponents::SvmCustomKernelFeaturesSelectionChromosome> m_populationWithFeatures;
		
		std::filesystem::path m_pngNameSource;

		const FeatureSetOptimizationWorkflow m_featureSetOptimization;

		unsigned int m_generationNumber;
		bool m_shrinkTrainingSet;

		double m_previousGoodSvRatio;
		double m_currentGamma;
		bool m_buildEnsamble;
		size_t m_forbiddenSetSize;

		static constexpr const char* m_algorithmName = "Sequential Gamma FS";

		//logger::LogFrontend m_logger;
	};
} // namespace genetic
