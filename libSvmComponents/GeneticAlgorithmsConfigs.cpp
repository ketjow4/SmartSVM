#include "GeneticAlgorithmsConfigs.h"
#include "SvmMetricFactory.h"
#include "libGeneticComponents/StopConditionFactory.h"
#include "SvmCrossoverFactory.h"
#include "SvmMutationFactory.h"
#include "libGeneticComponents/SelectionFactory.h"
#include "SvmGenerationPopulationFactory.h"
#include "LibGeneticComponents/CrossoverSelectionFactory.h"
#include "libRandom/RandomNumberGeneratorFactory.h"
#include "LibGeneticComponents/BinaryCrossoverFactory.h"
#include "LibGeneticComponents/BinaryMutationFactory.h"
#include "LibGeneticComponents/BinaryGenerationFactory.h"
#include "SvmComponentsExceptions.h"
#include "SvmKernelTraining.h"
#include "SvmUtils.h"
#include "SvmTraining.h"
#include "SvmTrainingSetCrossoverFactory.h"
#include "SvmTrainingSetMutationFactory.h"
#include "SvmTrainingSetPopulationFactory.h"
#include "MemeticFeaturesFactories.h"
#include "SvmMemeticFeatureSetPopulationFactory.h"
#include "CustomWidthGauss.h"
#include "CustomKernelTraining.h"
#include "libGeneticSvm/GridSearchWorkflow.h"
#include "libPlatform/loguru.hpp"
#include "RbfLinearInternals.h"
#include "SvmValidationFactory.h"

namespace svmComponents
{
SvmKernelChromosome parseKernelParameters(const platform::Subtree& config,
                                          const phd::svm::KernelTypes kernelType, bool isRegression)
{
	switch (kernelType)
	{
	case phd::svm::KernelTypes::Rbf:
	{
		if (isRegression)
		{
			auto gamma = config.getValue<double>("Kernel.Gamma");
			auto c = config.getValue<double>("Kernel.C");
			auto eps = config.getValue<double>("Kernel.Epsilon");
			return SvmKernelChromosome(kernelType, std::vector<double>{c, gamma, eps}, isRegression);
		}
		auto gamma = config.getValue<double>("Kernel.Gamma");
		auto c = config.getValue<double>("Kernel.C");
		return SvmKernelChromosome(kernelType, std::vector<double>{c, gamma}, isRegression);
	}
	case phd::svm::KernelTypes::Linear:
	{
		if (isRegression)
		{
			auto c = config.getValue<double>("Kernel.C");
			auto eps = config.getValue<double>("Kernel.Epsilon");
			return SvmKernelChromosome(kernelType, std::vector<double>{c, eps}, isRegression);
		}
		auto c = config.getValue<double>("Kernel.C");
		return SvmKernelChromosome(kernelType, std::vector<double>{c}, isRegression);
	}

	case phd::svm::KernelTypes::Poly:
	{
		if (isRegression)
		{
			throw UnsupportedKernelTypeException(kernelType);
		}
		auto c = config.getValue<double>("Kernel.C");
		auto Coef0 = config.getValue<double>("Kernel.Coef0");
		auto Degree = config.getValue<double>("Kernel.Degree");
		return SvmKernelChromosome(kernelType, std::vector<double>{c, Degree, Coef0}, isRegression);
	}

	case phd::svm::KernelTypes::Sigmoid:
	{
		if (isRegression)
		{
			throw UnsupportedKernelTypeException(kernelType);
		}
		auto c = config.getValue<double>("Kernel.C");
		auto Coef0 = config.getValue<double>("Kernel.Coef0");
		auto gamma = config.getValue<double>("Kernel.Gamma"); //gamma is alpha scale in sigmoid equation
		return SvmKernelChromosome(kernelType, std::vector<double>{c, Coef0, gamma}, isRegression);
	}
	default:
		throw UnsupportedKernelTypeException(kernelType);
	}
}

std::vector<unsigned int> countLabels(unsigned int numberOfClasses,
                                      const dataset::Dataset<std::vector<float>, float>& dataset)
{
	std::vector<unsigned int> labelsCount(numberOfClasses);
	auto targets = dataset.getLabels();
	std::for_each(targets.begin(), targets.end(),
	              [&labelsCount](const auto& label)
	              {
		              ++labelsCount[static_cast<int>(label)];
	              });
	return labelsCount;
}


unsigned int getNumberOfClassExamples(unsigned int numberOfClassExamples, std::vector<unsigned int> labelsCount, double thresholdForMaxNumberOfClassExamples)
{
	auto minorityClassExamplesNumber = static_cast<unsigned int>(*std::min_element(labelsCount.begin(), labelsCount.end()) * thresholdForMaxNumberOfClassExamples);
	if (minorityClassExamplesNumber < numberOfClassExamples)
		return minorityClassExamplesNumber;
	return numberOfClassExamples;
}

GeneticKernelEvolutionConfiguration::GeneticKernelEvolutionConfiguration(const platform::Subtree& config)
	: GeneticKernelEvolutionConfiguration(config.getNode("Svm.GeneticKernelEvolution"), SvmAlgorithmConfiguration(config))
{
}

GeneticKernelEvolutionConfiguration::GeneticKernelEvolutionConfiguration(const platform::Subtree& config, SvmAlgorithmConfiguration svmConfig)
	: m_svmConfig(std::move(svmConfig))
	, m_populationSize(config.getValue<unsigned int>("PopulationSize"))
	, m_training(std::make_shared<SvmKernelTraining>(m_svmConfig, m_svmConfig.m_estimationType == svmMetricType::Auc))
	, m_stopCondition(geneticComponents::StopConditionFactory::create<SvmKernelChromosome>(config))
	, m_crossover(SvmKernelCrossoverFactory::create(config))
	, m_mutation(SvmMutationFactory::create(config))
	, m_selection(geneticComponents::SelectionFactory::create<SvmKernelChromosome>(config))
	, m_populationGeneration(SvmGenerationPopulationFactory::create(config, m_svmConfig.m_kernelType))
	, m_parentSelection(geneticComponents::CrossoverSelectionFactory::create<SvmKernelChromosome>(config))
{
}

GeneticTrainingSetEvolutionConfiguration::GeneticTrainingSetEvolutionConfiguration(const platform::Subtree& config,
                                                                                   const dataset::Dataset<std::vector<float>, float>& traningSet)
	: GeneticTrainingSetEvolutionConfiguration(config.getNode("Svm.GaSvm"), traningSet, SvmAlgorithmConfiguration(config))
{
}

GeneticTrainingSetEvolutionConfiguration::GeneticTrainingSetEvolutionConfiguration(const platform::Subtree& config,
                                                                                   const dataset::Dataset<std::vector<float>, float>& traningSet,
                                                                                   SvmAlgorithmConfiguration svmConfig)
	: m_svmConfig(std::move(svmConfig))
	, m_populationSize(config.getValue<unsigned int>("PopulationSize"))
	, m_labelsCount(countLabels(config.getValue<unsigned int>("NumberOfClasses"), traningSet))
	, m_numberOfClassExamples(config.getValue<unsigned int>("NumberOfClassExamples"))
	, m_trainingSetInterface([&]()
		{
			auto enhanceTrainigSet = config.getValue<bool>("EnhanceTrainingSet");
			if (enhanceTrainigSet)
			{
				return std::shared_ptr<ITrainingSet>(new SetupAdditionalVectors(traningSet));
			}
			else
			{
				return std::shared_ptr<ITrainingSet>(new FullTrainingSet(traningSet));
			}
		}())
	, m_training(std::make_shared<SvmTraining<SvmTrainingSetChromosome>>(m_svmConfig,
	                                                                     parseKernelParameters(config, m_svmConfig.m_kernelType,
	                                                                                           config.getValue<bool>("Svm.isRegression")),
	                                                                     m_svmConfig.m_estimationType == svmMetricType::Auc))
	, m_stopCondition(geneticComponents::StopConditionFactory::create<SvmTrainingSetChromosome>(config))
	, m_validationMethod(SvmValidationFactory::create<SvmTrainingSetChromosome>(config, *m_svmConfig.m_estimationMethod))
	, m_crossover(SvmTrainingSetCrossoverFactory::create(config, m_trainingSetInterface))
	, m_mutation(SvmTrainingSetMutationFactory::create(config, traningSet, m_labelsCount, m_trainingSetInterface))
	, m_selection(geneticComponents::SelectionFactory::create<SvmTrainingSetChromosome>(config))
	, m_populationGeneration(SvmTrainingSetPopulationFactory::create(config, traningSet, m_labelsCount, m_trainingSetInterface))
	, m_parentSelection(geneticComponents::CrossoverSelectionFactory::create<SvmTrainingSetChromosome>(config))
{
}

GeneticFeatureSetEvolutionConfiguration::GeneticFeatureSetEvolutionConfiguration(const platform::Subtree& config,
                                                                                 const dataset::Dataset<std::vector<float>, float>& traningSet)
	: GeneticFeatureSetEvolutionConfiguration(config.getNode("Svm.FeatureSetSelection"),
	                                          traningSet,
	                                          SvmAlgorithmConfiguration(config))
{
}

GeneticFeatureSetEvolutionConfiguration::GeneticFeatureSetEvolutionConfiguration(const platform::Subtree& config,
                                                                                 const dataset::Dataset<std::vector<float>, float>& traningSet,
                                                                                 SvmAlgorithmConfiguration svmConfig)
	: m_svmConfig(std::move(svmConfig))
	, m_populationSize(config.getValue<unsigned int>("PopulationSize"))
	, m_training(std::make_shared<SvmTraining<SvmFeatureSetChromosome>>(m_svmConfig,
	                                                                    parseKernelParameters(config, m_svmConfig.m_kernelType,
	                                                                                          config.getValue<bool>("Svm.isRegression")),
	                                                                    m_svmConfig.m_estimationType == svmMetricType::Auc))
	, m_stopCondition(geneticComponents::StopConditionFactory::create<SvmFeatureSetChromosome>(config))
	, m_crossover(geneticComponents::BinaryCrossoverFactory::create<SvmFeatureSetChromosome>(config))
	, m_mutation(geneticComponents::BinaryMutationFactory::create<SvmFeatureSetChromosome>(config))
	, m_selection(geneticComponents::SelectionFactory::create<SvmFeatureSetChromosome>(config))
	, m_populationGeneration([&]()
	{
		auto numberOfFeaturesInDataset = static_cast<unsigned int>(traningSet.getSamples()[0].size());
		return geneticComponents::BinaryGenerationFactory::create<SvmFeatureSetChromosome>(config,
		                                                                                   numberOfFeaturesInDataset);
	}())
	, m_parentSelection(geneticComponents::CrossoverSelectionFactory::create<SvmFeatureSetChromosome>(config))
{
}

MemeticTrainingSetEvolutionConfiguration::MemeticTrainingSetEvolutionConfiguration(const platform::Subtree& config,
                                                                                   const dataset::Dataset<std::vector<float>, float>& traningSet)
	: MemeticTrainingSetEvolutionConfiguration(config.getNode("Svm.MemeticTrainingSetSelection"),
	                                           traningSet,
	                                           SvmAlgorithmConfiguration(config))
{
}

MemeticTrainingSetEvolutionConfiguration::MemeticTrainingSetEvolutionConfiguration(const platform::Subtree& config,
                                                                                   const dataset::Dataset<std::vector<float>, float>& traningSet,
                                                                                   SvmAlgorithmConfiguration svmConfig)
	: m_svmConfig(std::move(svmConfig))
	, m_populationSize(config.getValue<unsigned int>("PopulationSize"))
	, m_numberOfClasses(config.getValue<unsigned int>("NumberOfClasses"))
	, m_labelsCount(std::move(countLabels(m_numberOfClasses, traningSet)))
    , m_superIndividualAlpha(config.getValue<double>("Memetic.SuperIndividualsAlpha"))
	, m_initialNumberOfClassExamples(getNumberOfClassExamples(config.getValue<unsigned int>("NumberOfClassExamples"), m_labelsCount, config.getValue<double>("Memetic.ThresholdForMaxNumberOfClassExamples")))
	, m_trainingSetInterface([&]()
{
	auto enhanceTrainigSet = config.getValue<bool>("EnhanceTrainingSet");
	if (enhanceTrainigSet)
	{
		return std::shared_ptr<ITrainingSet>(new SetupAdditionalVectors(traningSet));
	}
	else
	{
		return std::shared_ptr<ITrainingSet>(new FullTrainingSet(traningSet));
	}
}())
	, m_training(std::make_shared<SvmTraining<SvmTrainingSetChromosome>>(m_svmConfig,
	                                                                     parseKernelParameters(config, m_svmConfig.m_kernelType,
	                                                                                           config.getValue<bool>("Svm.isRegression")),
	                                                                     m_svmConfig.m_estimationType == svmMetricType::Auc))
	, m_stopCondition(geneticComponents::StopConditionFactory::create<SvmTrainingSetChromosome>(config))
	, m_crossover(SvmTrainingSetCrossoverFactory::create(config, m_trainingSetInterface))
	, m_mutation(SvmTrainingSetMutationFactory::create(config,
	                                                   traningSet,
	                                                   m_labelsCount, m_trainingSetInterface))
	, m_selection(geneticComponents::SelectionFactory::create<SvmTrainingSetChromosome>(config))
	, m_populationGeneration(SvmTrainingSetPopulationFactory::create(config,
	                                                                 traningSet,
	                                                                 m_labelsCount, m_trainingSetInterface))
	, m_parentSelection(geneticComponents::CrossoverSelectionFactory::create<SvmTrainingSetChromosome>(config))
	, m_validationMethod(SvmValidationFactory::create<SvmTrainingSetChromosome>(config, *m_svmConfig.m_estimationMethod))
	, m_compensationGeneration(std::make_shared<CompensationInformation>(random::RandomNumberGeneratorFactory::create(config),
	                                                                     m_numberOfClasses))
	, m_education(std::make_shared<EducationOfTrainingSet>(platform::Percent(config.getValue<double>("Memetic.EducationProbability")),
	                                                       m_numberOfClasses,
	                                                       random::RandomNumberGeneratorFactory::create(config),
	                                                       std::make_unique<SupportVectorPool>()))
	, m_compensation(std::make_shared<CrossoverCompensation>(traningSet,
	                                                         random::RandomNumberGeneratorFactory::create(config),
	                                                         m_numberOfClasses))
	, m_supporVectorPool(std::make_shared<SupportVectorPool>())
	, m_superIndivudualsGeneration(std::make_shared<SuperIndividualsCreation>(random::RandomNumberGeneratorFactory::create(config),
	                                                                          m_numberOfClasses))
	, m_adaptation(std::make_shared<MemeticTrainingSetAdaptation>(config.getValue<bool>("CrossoverSelection.LocalGlobalSelection.IsLocalMode"),
	                                                              m_initialNumberOfClassExamples,
	                                                              platform::Percent(config.getValue<double>("Memetic.PercentOfSupportVectorsThreshold")),
	                                                              config.getValue<unsigned int>("Memetic.IterationsBeforeModeChange"),
	                                                              m_labelsCount,
	                                                              config.getValue<double>("Memetic.ThresholdForMaxNumberOfClassExamples"),
																  config.getValue<double>("Memetic.MaxK")))
{
}

MemeticFeatureSetEvolutionConfiguration::MemeticFeatureSetEvolutionConfiguration(const platform::Subtree& config,
                                                                                 const dataset::Dataset<std::vector<float>, float>& traningSet)
	: MemeticFeatureSetEvolutionConfiguration(config.getNode("Svm.MemeticFeatureSetSelection"),
	                                          traningSet,
	                                          SvmAlgorithmConfiguration(config),
	                                          config.getValue<std::string>("Svm.TrainingData"), config.getValue<std::string>("Svm.OutputFolderPath"))
{
}

MemeticFeatureSetEvolutionConfiguration::MemeticFeatureSetEvolutionConfiguration(const platform::Subtree& config,
                                                                                 const dataset::Dataset<std::vector<float>, float>& traningSet,
                                                                                 SvmAlgorithmConfiguration svmConfig,
                                                                                 std::string trainingDataPath,
                                                                                 std::string outputPath)
	: m_svmConfig(std::move(svmConfig))
	, m_populationSize(config.getValue<unsigned int>("PopulationSize"))
	, m_superIndividualAlpha(config.getValue<double>("Memetic.SuperIndividualsAlpha"))
	, m_initialNumberOfClassExamples(config.getValue<unsigned int>("NumberOfClassExamples"))
	, m_numberOfClasses(config.getValue<unsigned int>("NumberOfClasses"))
	, m_labelsCount(std::move(countLabels(m_numberOfClasses, traningSet)))
	, m_training(std::make_shared<SvmTraining<SvmFeatureSetMemeticChromosome>>(m_svmConfig,
	                                                                           parseKernelParameters(
		                                                                           config, m_svmConfig.m_kernelType, config.getValue<bool>("Svm.isRegression")),
	                                                                           m_svmConfig.m_estimationType == svmMetricType::Auc))
	, m_stopCondition(geneticComponents::StopConditionFactory::create<SvmFeatureSetMemeticChromosome>(config))
	, m_crossover(SvmMemeticFeatureSetCrossoverFactory::create(config))
	, m_mutation(SvmMemeticFeatureSetMutationFactory::create(config,
	                                                         traningSet,
	                                                         m_labelsCount))
	, m_selection(geneticComponents::SelectionFactory::create<SvmFeatureSetMemeticChromosome>(config))
	, m_populationGeneration(SvmMemeticFeatureSetPopulationFactory::create(config,
	                                                                       traningSet,
	                                                                       trainingDataPath, outputPath))
	, m_parentSelection(geneticComponents::CrossoverSelectionFactory::create<SvmFeatureSetMemeticChromosome>(config))
	, m_compensationGeneration(std::make_shared<MemeticFeaturesCompensationGeneration>(random::RandomNumberGeneratorFactory::create(config)))
	, m_education(std::make_shared<MemeticFeaturesEducation>(platform::Percent(config.getValue<double>("Memetic.EducationProbability")),
	                                                         m_numberOfClasses,
	                                                         random::RandomNumberGeneratorFactory::create(config),
	                                                         std::make_unique<MemeticFeaturesPool>()))
	, m_compensation(std::make_shared<MemeticFeatureCompensation>(traningSet,
	                                                              random::RandomNumberGeneratorFactory::create(config),
	                                                              m_numberOfClasses))
	, m_supporVectorPool(std::make_shared<MemeticFeaturesPool>())
	, m_superIndivudualsGeneration(std::make_shared<MemeticFeaturesSuperIndividualsGeneration>(random::RandomNumberGeneratorFactory::create(config),
	                                                                                           m_numberOfClasses))
	, m_adaptation(std::make_shared<MemeticFeaturesAdaptation>(config.getValue<bool>("CrossoverSelection.LocalGlobalSelection.IsLocalMode"),
	                                                           m_initialNumberOfClassExamples,
	                                                           platform::Percent(config.getValue<double>("Memetic.PercentOfSupportVectorsThreshold")),
	                                                           config.getValue<unsigned int>("Memetic.IterationsBeforeModeChange"),
	                                                           static_cast<unsigned int>(traningSet.getSample(0).size()),
	                                                           config.getValue<double>("Memetic.ThresholdForMaxNumberOfClassExamples")))
{
}

CustomKernelEvolutionConfiguration::CustomKernelEvolutionConfiguration(const platform::Subtree& config,
                                                                       const dataset::Dataset<std::vector<float>, float>& traningSet)
	: CustomKernelEvolutionConfiguration(config.getNode("Svm.CustomKernel"),
	                                     traningSet,
	                                     SvmAlgorithmConfiguration(config))
{
}

CustomKernelEvolutionConfiguration::CustomKernelEvolutionConfiguration(const platform::Subtree& config,
                                                                       const dataset::Dataset<std::vector<float>, float>& traningSet,
                                                                       SvmAlgorithmConfiguration svmConfig)
	: m_svmConfig(std::move(svmConfig))
	, m_populationSize(config.getValue<unsigned int>("PopulationSize"))
	, m_labelsCount(std::move(countLabels(config.getValue<unsigned int>("NumberOfClasses"), traningSet)))
	, m_numberOfClassExamples(config.getValue<unsigned int>("NumberOfClassExamples"))
	, m_training(std::make_shared<SvmTrainingCustomKernel>(m_svmConfig, m_svmConfig.m_estimationType == svmMetricType::Auc, 
		config.getValue<std::string>("KernelType"),
		config.getValue<bool>("TrainAlpha")))
	, m_stopCondition(geneticComponents::StopConditionFactory::create<SvmCustomKernelChromosome>(config))
	, m_crossover(std::make_shared<CrossoverCustomGauss>(random::RandomNumberGeneratorFactory::create(config), config.getValue<unsigned int>("NumberOfClasses")))
	, m_mutation(std::make_shared<MutationCustomGauss>(random::RandomNumberGeneratorFactory::create(config),
	                                                   platform::Percent(config.getValue<double>("Mutation.GaSvm.ExchangePercent")),
	                                                   platform::Percent(config.getValue<double>("Mutation.GaSvm.MutationProbability")),
	                                                   traningSet,
	                                                   m_labelsCount))
	, m_selection(geneticComponents::SelectionFactory::create<SvmCustomKernelChromosome>(config))
	, m_populationGeneration(std::make_shared<CusomKernelGeneration>(traningSet, random::RandomNumberGeneratorFactory::create(config), m_numberOfClassExamples,
	                                                                 m_labelsCount))
	, m_parentSelection(geneticComponents::CrossoverSelectionFactory::create<SvmCustomKernelChromosome>(config))
{
}

SequentialGammaConfig::SequentialGammaConfig(const platform::Subtree& config,
                                             const dataset::Dataset<std::vector<float>, float>& traningSet)
	: SequentialGammaConfig(config.getNode("Svm.SequentialGamma"),
	                        traningSet,
	                        SvmAlgorithmConfiguration(config))
{
}

SequentialGammaConfig::SequentialGammaConfig(const platform::Subtree& config,
	const dataset::Dataset<std::vector<float>, float>& traningSet,
	SvmAlgorithmConfiguration svmConfig)
	: m_svmConfig(std::move(svmConfig))
	, m_useSmallerGamma(config.getValue<bool>("UseSmallerGamma"))
	, m_logStepGamma(config.getValue<double>("GammaLogStep"))
	, m_helperAlgorithmName(config.getValue<std::string>("HelperAlgorithmName"))
	, m_populationSize(config.getValue<unsigned int>("PopulationSize"))
	, m_labelsCount(std::move(countLabels(config.getValue<unsigned int>("NumberOfClasses"), traningSet)))
	, m_numberOfClassExamples(getNumberOfClassExamples(config.getValue<unsigned int>("NumberOfClassExamples"), m_labelsCount,
		config.getValue<double>("ThresholdForMaxNumberOfClassExamples")))
	, m_superIndividualAlpha(config.getValue<double>("SuperIndividualsAlpha"))
	, m_trainAlpha(config.getValue<bool>("TrainAlpha"))
	, m_shrinkOnBestOnly(config.getValue<bool>("ShrinkOnBestOnly"))
	, m_genererateEveryGeneration([&]() -> int
{
			try
			{
				auto value = config.getValue<int>("Validation.Generation");
				return value;

			}
			catch (std::exception&)
			{
				return 100;
			}
}())
	, m_training(std::make_shared<SvmTrainingCustomKernel>(m_svmConfig, m_svmConfig.m_estimationType == svmMetricType::Auc,
	                                                       config.getValue<std::string>("KernelType"),
	                                                       config.getValue<bool>("TrainAlpha")))
	, m_stopCondition(geneticComponents::StopConditionFactory::create<SvmCustomKernelChromosome>(config))
	, m_crossover(
		std::make_shared<CrossoverCustomGauss>(random::RandomNumberGeneratorFactory::create(config), config.getValue<unsigned int>("NumberOfClasses")))
	, m_mutation(std::make_shared<MutationCustomGaussSequential>(random::RandomNumberGeneratorFactory::create(config),
	                                                             platform::Percent(config.getValue<double>("Mutation.GaSvm.ExchangePercent")),
	                                                             platform::Percent(config.getValue<double>("Mutation.GaSvm.MutationProbability")),
	                                                             traningSet,
	                                                             m_labelsCount))
	, m_selection(geneticComponents::SelectionFactory::create<SvmCustomKernelChromosome>(config))
	, m_populationGeneration([&]() -> PopulationGeneration<SvmCustomKernelChromosome>
	{
		if (config.getValue<std::string>("Generation.Name") == "Random")
		{
			return std::make_shared<CusomKernelGenerationSequential>(traningSet, random::RandomNumberGeneratorFactory::create(config),
			                                                         m_numberOfClassExamples,
			                                                         m_labelsCount);
		}
		else if (config.getValue<std::string>("Generation.Name") == "NewKernel")
		{
			return std::make_shared<CusomKernelGenerationRbfLinearSequential>(traningSet, random::RandomNumberGeneratorFactory::create(config),
			                                                                  m_numberOfClassExamples,
			                                                                  m_labelsCount);
		}

		LOG_F(ERROR, "Error: Used Random generation in SequentialGamma Algorithm as specified one was not found, Specified: %s",
		      config.getValue<std::string>("Generation.Name").c_str());
		return std::make_shared<CusomKernelGenerationSequential>(traningSet, random::RandomNumberGeneratorFactory::create(config),
		                                                         m_numberOfClassExamples,
		                                                         m_labelsCount);
	}())
	, m_parentSelection(geneticComponents::CrossoverSelectionFactory::create<SvmCustomKernelChromosome>(config))
	, m_validationMethod(SvmValidationFactory::create<SvmCustomKernelChromosome>(config, *m_svmConfig.m_estimationMethod))
	, m_educationElement(std::make_shared<EducationOfTrainingSetGamma>(platform::Percent(config.getValue<double>("EducationProbability")),
	                                                                   config.getValue<unsigned int>("NumberOfClasses"),
	                                                                   random::RandomNumberGeneratorFactory::create(config),
	                                                                   std::make_unique<SupportVectorPoolGamma>()))
	, m_crossoverCompensationElement(traningSet, random::RandomNumberGeneratorFactory::create(config), config.getValue<unsigned int>("NumberOfClasses"))
	, m_adaptationElement(false,
	                      m_numberOfClassExamples,
	                      platform::Percent(config.getValue<double>("PercentOfSupportVectorsThreshold")),
	                      config.getValue<unsigned int>("IterationsBeforeModeChange"),
	                      m_labelsCount,
	                      config.getValue<double>("ThresholdForMaxNumberOfClassExamples"))
	, m_superIndividualsGenerationElement(
		std::make_shared<SuperIndividualsCreationGamma>(random::RandomNumberGeneratorFactory::create(config), config.getValue<unsigned int>("NumberOfClasses")))
	, m_compensationGenerationElement(random::RandomNumberGeneratorFactory::create(config), config.getValue<unsigned int>("NumberOfClasses"))
{
}

MutlipleGammaMASVMConfig::MutlipleGammaMASVMConfig(const platform::Subtree& config, const dataset::Dataset<std::vector<float>, float>& traningSet)
: MutlipleGammaMASVMConfig(config.getNode("Svm.MultipleGammaMASVM"),
	traningSet,
	SvmAlgorithmConfiguration(config))
{
}

MutlipleGammaMASVMConfig::MutlipleGammaMASVMConfig(const platform::Subtree& config, const dataset::Dataset<std::vector<float>, float>& traningSet,
	SvmAlgorithmConfiguration svmConfig)
	: m_svmConfig(std::move(svmConfig))
	, m_populationSize(config.getValue<unsigned int>("PopulationSize"))
	, m_labelsCount(std::move(countLabels(config.getValue<unsigned int>("NumberOfClasses"), traningSet)))
	, m_numberOfClassExamples(getNumberOfClassExamples(config.getValue<unsigned int>("NumberOfClassExamples"), m_labelsCount, config.getValue<double>("ThresholdForMaxNumberOfClassExamples")))
	, m_superIndividualAlpha(config.getValue<double>("SuperIndividualsAlpha"))
	, m_training(std::make_shared<SvmTrainingCustomKernel>(m_svmConfig, m_svmConfig.m_estimationType == svmMetricType::Auc,
		config.getValue<std::string>("KernelType"),
		config.getValue<bool>("TrainAlpha")))
	, m_stopCondition(geneticComponents::StopConditionFactory::create<SvmCustomKernelChromosome>(config))
	, m_crossover(std::make_shared<CrossoverCustomGauss>(random::RandomNumberGeneratorFactory::create(config), config.getValue<unsigned int>("NumberOfClasses")))
	, m_mutation(std::make_shared<MultipleGammaMutation>(random::RandomNumberGeneratorFactory::create(config),
	                                                     platform::Percent(config.getValue<double>("Mutation.GaSvm.ExchangePercent")),
	                                                     platform::Percent(config.getValue<double>("Mutation.GaSvm.MutationProbability")),
	                                                     traningSet,
	                                                     m_labelsCount))
	, m_selection(geneticComponents::SelectionFactory::create<SvmCustomKernelChromosome>(config))
	, m_populationGeneration(std::make_shared<MultipleGammaGeneration>(traningSet, random::RandomNumberGeneratorFactory::create(config),
	                                                                   m_numberOfClassExamples,
	                                                                   m_labelsCount))
	, m_parentSelection(geneticComponents::CrossoverSelectionFactory::create<SvmCustomKernelChromosome>(config))
	, m_validationMethod(SvmValidationFactory::create<SvmCustomKernelChromosome>(config, *m_svmConfig.m_estimationMethod))
	, m_educationElement(std::make_shared<MultipleGammaEducationOfTrainingSet>(platform::Percent(config.getValue<double>("EducationProbability")),
	                                                                           config.getValue<unsigned int>("NumberOfClasses"),
	                                                                           random::RandomNumberGeneratorFactory::create(config),
	                                                                           std::make_unique<SupportVectorPoolGamma>()))
	, m_crossoverCompensationElement(traningSet, random::RandomNumberGeneratorFactory::create(config), config.getValue<unsigned int>("NumberOfClasses"))
	, m_adaptationElement(false,
	                      m_numberOfClassExamples,
	                      platform::Percent(config.getValue<double>("PercentOfSupportVectorsThreshold")),
	                      config.getValue<unsigned int>("IterationsBeforeModeChange"),
	                      m_labelsCount,
	                      config.getValue<double>("ThresholdForMaxNumberOfClassExamples"))
	, m_superIndividualsGenerationElement(std::make_shared<MultipleGammaSuperIndividualsCreation>(random::RandomNumberGeneratorFactory::create(config), config.getValue<unsigned int>("NumberOfClasses")))
	, m_compensationGenerationElement(random::RandomNumberGeneratorFactory::create(config), config.getValue<unsigned int>("NumberOfClasses"))
{
}

RbfLinearConfig::RbfLinearConfig(const platform::Subtree& config,
                                 const dataset::Dataset<std::vector<float>, float>& traningSet)
	: RbfLinearConfig(config.getNode("Svm.RbfLinear"),
	                  traningSet,
	                  SvmAlgorithmConfiguration(config))
{
}

RbfLinearConfig::RbfLinearConfig(const platform::Subtree& config,
                                 const dataset::Dataset<std::vector<float>, float>& traningSet,
                                 SvmAlgorithmConfiguration svmConfig)
	: m_svmConfig(std::move(svmConfig))
	, m_populationSize(config.getValue<unsigned int>("PopulationSize"))
	, m_labelsCount(std::move(countLabels(config.getValue<unsigned int>("NumberOfClasses"), traningSet)))
	, m_numberOfClassExamples(getNumberOfClassExamples(config.getValue<unsigned int>("NumberOfClassExamples"), m_labelsCount,
	                                                   config.getValue<double>("ThresholdForMaxNumberOfClassExamples")))
	, m_superIndividualAlpha(config.getValue<double>("SuperIndividualsAlpha"))
	, m_trainAlpha(config.getValue<bool>("TrainAlpha"))
	, m_training(std::make_shared<SvmTrainingCustomKernel>(m_svmConfig, m_svmConfig.m_estimationType == svmMetricType::Auc,
	                                                       config.getValue<std::string>("KernelType"),
	                                                       config.getValue<bool>("TrainAlpha")))
	, m_stopCondition(geneticComponents::StopConditionFactory::create<SvmCustomKernelChromosome>(config))
	, m_crossover(
		std::make_shared<CrossoverCustomGauss>(random::RandomNumberGeneratorFactory::create(config), config.getValue<unsigned int>("NumberOfClasses")))
	, m_mutation(std::make_shared<MutationRbfLinear>(random::RandomNumberGeneratorFactory::create(config),
	                                                 platform::Percent(config.getValue<double>("Mutation.GaSvm.ExchangePercent")),
	                                                 platform::Percent(config.getValue<double>("Mutation.GaSvm.MutationProbability")),
	                                                 traningSet,
	                                                 m_labelsCount))
	, m_selection(geneticComponents::SelectionFactory::create<SvmCustomKernelChromosome>(config))
	, m_populationGeneration([&]() -> PopulationGeneration<SvmCustomKernelChromosome>
	{
		if (config.getValue<std::string>("Generation.Name") == "NewKernel")
		{
			return std::make_shared<CusomKernelGenerationRbfLinearSequential>(traningSet, random::RandomNumberGeneratorFactory::create(config),
			                                                                  m_numberOfClassExamples,
			                                                                  m_labelsCount);
		}

		LOG_F(ERROR, "Error: Used Random generation in RbfLinear Algorithm as specified one was not found, Specified: %s",
		      config.getValue<std::string>("Generation.Name").c_str());
		return std::make_shared<CusomKernelGenerationSequential>(traningSet, random::RandomNumberGeneratorFactory::create(config),
		                                                         m_numberOfClassExamples,
		                                                         m_labelsCount);
	}())
	, m_parentSelection(geneticComponents::CrossoverSelectionFactory::create<SvmCustomKernelChromosome>(config))
	, m_validationMethod(SvmValidationFactory::create<SvmCustomKernelChromosome>(config, *m_svmConfig.m_estimationMethod))
	, m_educationElement(std::make_shared<EducationOfTrainingSetRbfLinear>(platform::Percent(config.getValue<double>("EducationProbability")),
	                                                                       config.getValue<unsigned int>("NumberOfClasses"),
	                                                                       random::RandomNumberGeneratorFactory::create(config),
	                                                                       std::make_unique<SupportVectorPoolGamma>()))
	, m_crossoverCompensationElement(traningSet, random::RandomNumberGeneratorFactory::create(config), config.getValue<unsigned int>("NumberOfClasses"))
	, m_adaptationElement(false,
	                      m_numberOfClassExamples,
	                      platform::Percent(config.getValue<double>("PercentOfSupportVectorsThreshold")),
	                      config.getValue<unsigned int>("IterationsBeforeModeChange"),
	                      m_labelsCount,
	                      config.getValue<double>("ThresholdForMaxNumberOfClassExamples"))
	, m_superIndividualsGenerationElement(
		std::make_shared<SuperIndividualsCreationRbfLinear>(random::RandomNumberGeneratorFactory::create(config),
		                                                    config.getValue<unsigned int>("NumberOfClasses")))
	, m_compensationGenerationElement(random::RandomNumberGeneratorFactory::create(config), config.getValue<unsigned int>("NumberOfClasses"))
{
}
} // namespace svmComponents
