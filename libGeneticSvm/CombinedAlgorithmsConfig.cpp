
#include "libGeneticComponents/StopConditionFactory.h"
#include "CombinedAlgorithmsConfig.h"
#include "KernelOptimizationWorkflowFactory.h"
#include "TrainingSetOptimizationWorkflowFactory.h"
#include "FeatureSetOptimizationWorkflowFactory.h"
#include "LibGeneticComponents/CrossoverSelectionFactory.h"
#include "libSvmComponents/SvmTraining.h"
#include "LibGeneticComponents/SelectionFactory.h"
#include "libPlatform/loguru.hpp"
#include "libRandom/RandomNumberGeneratorFactory.h"
#include "libSvmComponents/CustomKernelTraining.h"
#include "libSvmComponents/CustomWidthGauss.h"
#include "libSvmComponents/SvmValidationFactory.h"

namespace genetic
{
using namespace geneticComponents;
using namespace svmComponents;

GeneticAlternatingEvolutionConfiguration::GeneticAlternatingEvolutionConfiguration(const platform::Subtree& config,
                                                                                   IDatasetLoader& loadingWorkflow)
    : m_svmConfig(config)
	, m_trainingSetOptimization(TrainingSetOptimizationWorkflowFactory::create(config, loadingWorkflow, "Svm.Alga"))
    , m_kernelOptimization(KernelOptimizationWorkflowFactory::create(config, loadingWorkflow, "Svm.Alga"))
    , m_stopKernel(StopConditionFactory::create<SvmKernelChromosome>(config.getNode("Svm." + config.getValue<std::string>("Svm.Alga.KernelOptimization.Name"))))
    , m_stopTrainingSet(
        StopConditionFactory::create<SvmTrainingSetChromosome>(config.getNode("Svm." + config.getValue<std::string>("Svm.Alga.TrainingSetOptimization.Name"))))
{
}

KTFGeneticEvolutionConfiguration::KTFGeneticEvolutionConfiguration(const platform::Subtree& config,
                                                                   IDatasetLoader& loadingWorkflow)
    : m_trainingSetOptimization(TrainingSetOptimizationWorkflowFactory::create(config, loadingWorkflow, "Svm.KTF"))
    , m_kernelOptimization(KernelOptimizationWorkflowFactory::create(config, loadingWorkflow, "Svm.KTF"))
    , m_featureSetOptimization(FeatureSetOptimizationWorkflowFactory::create(config, loadingWorkflow, "Svm.KTF"))
    , m_stopKernel(StopConditionFactory::create<SvmKernelChromosome>(config.getNode("Svm." + config.getValue<std::string>("Svm.KTF.KernelOptimization.Name"))))
    , m_stopTrainingSet(StopConditionFactory::create<SvmTrainingSetChromosome>(
        config.getNode("Svm." + config.getValue<std::string>("Svm.KTF.TrainingSetOptimization.Name"))))
    , m_stopFeatureSet(StopConditionFactory::create<SvmFeatureSetMemeticChromosome>(
        config.getNode("Svm." + config.getValue<std::string>("Svm.KTF.FeatureSetOptimization.Name"))))
{
}

TFGeneticEvolutionConfiguration::TFGeneticEvolutionConfiguration(const platform::Subtree& config,
                                                                 IDatasetLoader& loadingWorkflow,
                                                                 const std::string& algorithmName)
    : m_trainingSetOptimization(TrainingSetOptimizationWorkflowFactory::create(config, loadingWorkflow, "Svm." + algorithmName))
    , m_featureSetOptimization(FeatureSetOptimizationWorkflowFactory::create(config, loadingWorkflow, "Svm." + algorithmName))
    , m_stopTrainingSet(StopConditionFactory::create<SvmTrainingSetChromosome>(
        config.getNode("Svm." + config.getValue<std::string>("Svm." + algorithmName + ".TrainingSetOptimization.Name"))))
    , m_stopFeatureSet(StopConditionFactory::create<SvmFeatureSetMemeticChromosome>(
        config.getNode("Svm." + config.getValue<std::string>("Svm." + algorithmName + ".FeatureSetOptimization.Name"))))
{
}

SimultaneousWorkflowConfig::SimultaneousWorkflowConfig(const platform::Subtree& config, IDatasetLoader& loadingWorkflow)
    : m_populationSize(config.getValue<unsigned int>("Svm.SESVM.PopulationSize"))
	, m_config(config)
    , m_svmConfig(config)
    , m_trainingSetOptimization(TrainingSetOptimizationWorkflowFactory::create(config, loadingWorkflow, "Svm.SESVM"))
    , m_kernelOptimization(KernelOptimizationWorkflowFactory::create(config, loadingWorkflow, "Svm.SESVM"))
    , m_featureSetOptimization(FeatureSetOptimizationWorkflowFactory::create(config, loadingWorkflow, "Svm.SESVM"))
    , m_stopCondition(StopConditionFactory::create<SvmSimultaneousChromosome>(config.getNode("Svm." + config.getValue<std::string>("Svm.SESVM.KernelOptimization.Name"))))
    , m_svmTraining(std::make_shared<SvmTrainingSSVM>(m_svmConfig,
                                                      m_svmConfig.m_estimationType == svmMetricType::Auc))
    , m_selectionElement(SelectionFactory::create<SvmSimultaneousChromosome>(config.getNode("Svm.SESVM")))
{
}

RandomSearchWorkflowConfig::RandomSearchWorkflowConfig(const platform::Subtree& config, IDatasetLoader& loadingWorkflow)
    : m_populationSize(config.getValue<unsigned int>("Svm.RandomSearch.PopulationSize"))
    , m_svmConfig(config)
    , m_trainingSetOptimization(TrainingSetOptimizationWorkflowFactory::create(config, loadingWorkflow, "Svm.RandomSearch"))
    , m_kernelOptimization(KernelOptimizationWorkflowFactory::create(config, loadingWorkflow, "Svm.RandomSearch"))
    , m_featureSetOptimization(FeatureSetOptimizationWorkflowFactory::create(config, loadingWorkflow, "Svm.RandomSearch"))
    , m_svmTraining(std::make_shared<SvmTrainingSSVM>(m_svmConfig,
                                                      m_svmConfig.m_estimationType == svmMetricType::Auc))
{
}


RandomSearchWorkflowInitPopsConfig::RandomSearchWorkflowInitPopsConfig(const platform::Subtree& config, IDatasetLoader& loadingWorkflow)
    : m_populationSize(config.getValue<unsigned int>("Svm.RandomSearchInitPop.PopulationSize"))
    , m_svmConfig(config)
    , m_trainingSetOptimization(TrainingSetOptimizationWorkflowFactory::create(config, loadingWorkflow, "Svm.RandomSearchInitPop"))
    , m_kernelOptimization(KernelOptimizationWorkflowFactory::create(config, loadingWorkflow, "Svm.RandomSearchInitPop"))
    , m_featureSetOptimization(FeatureSetOptimizationWorkflowFactory::create(config, loadingWorkflow, "Svm.RandomSearchInitPop"))
    , m_svmTraining(std::make_shared<SvmTrainingSSVM>(m_svmConfig,
                                                      m_svmConfig.m_estimationType == svmMetricType::Auc))
{
}

RandomSearchWorkflowEvoHelpConfig::RandomSearchWorkflowEvoHelpConfig(const platform::Subtree& config, IDatasetLoader& loadingWorkflow)
    : m_populationSize(config.getValue<unsigned int>("Svm.RandomSearchEvoHelp.PopulationSize"))
    , m_svmConfig(config)
    , m_trainingSetOptimization(TrainingSetOptimizationWorkflowFactory::create(config, loadingWorkflow, "Svm.RandomSearchEvoHelp"))
    , m_kernelOptimization(KernelOptimizationWorkflowFactory::create(config, loadingWorkflow, "Svm.RandomSearchEvoHelp"))
    , m_featureSetOptimization(FeatureSetOptimizationWorkflowFactory::create(config, loadingWorkflow, "Svm.RandomSearchEvoHelp"))
    , m_svmTraining(std::make_shared<SvmTrainingSSVM>(m_svmConfig,
                                                      m_svmConfig.m_estimationType == svmMetricType::Auc))
{
}

EnsembleTreeWorkflowConfig::EnsembleTreeWorkflowConfig(const platform::Subtree& config, IDatasetLoader& loadingWorkflow)
    : m_populationSize(config.getValue<unsigned int>("Svm.EnsembleTree.PopulationSize"))
	, m_constKernel(config.getValue<bool>("Svm.EnsembleTree.ConstKernel"))
	, m_switchFitness(config.getValue<bool>("Svm.EnsembleTree.SwitchFitness"))
	, m_config(config)
	, m_useDasvmKernel(config.getValue<bool>("Svm.EnsembleTree.DasvmKernel"))
	, m_metricMode([&]()
{
            auto mode = config.getValue<std::string>("Svm.EnsembleTree.GrowKMode");
            if (mode == "defaultOption")
            {
                return MetricMode::defaultOption;
            }
            if (mode == "zeroOut")
            {
                return MetricMode::zeroOut;
            }
            if (mode == "nonlinearDecrease")
            {
                return MetricMode::nonlinearDecrease;
            }
            if (mode == "boundryCheck")
            {
                return MetricMode::boundryCheck;
            }
            throw std::exception("Unknown metric mode in EnsembleTreeWorkflowConfig");
}())
    , m_svmConfig(config)
	, m_addSvToTraining(config.getValue<bool>("Svm.EnsembleTree.AddSvToTraining"))
	, m_SvMode([&]()
{
            auto mode = config.getValue<std::string>("Svm.EnsembleTree.SvMode");
            if (mode == "defaultOption")
            {
                return SvMode::none;
            }
            if (mode == "previous")
            {
                return SvMode::previousOnes;
            }
            if (mode == "global")
            {
                return SvMode::all;
            }
            throw std::exception("Unknown metric mode in EnsembleTreeWorkflowConfig");
}())
    , m_trainingSetOptimization(TrainingSetOptimizationWorkflowFactory::create(config, loadingWorkflow, "Svm.EnsembleTree"))
    , m_kernelOptimization(KernelOptimizationWorkflowFactory::create(config, loadingWorkflow, "Svm.EnsembleTree"))
    , m_featureSetOptimization(FeatureSetOptimizationWorkflowFactory::create(config, loadingWorkflow, "Svm.EnsembleTree"))
    , m_stopCondition(StopConditionFactory::create<SvmSimultaneousChromosome>(config.getNode("Svm." + config.getValue<std::string>("Svm.EnsembleTree.KernelOptimization.Name"))))
    , m_svmTraining(std::make_shared<SvmTrainingSSVM>(m_svmConfig,
        m_svmConfig.m_estimationType == svmMetricType::Auc))
    , m_selectionElement(SelectionFactory::create<SvmSimultaneousChromosome>(config.getNode("Svm.EnsembleTree")))
{
}



unsigned int getNumberOfClassExamples(unsigned int numberOfClassExamples, std::vector<unsigned int> labelsCount, double thresholdForMaxNumberOfClassExamples)
{
	auto minorityClassExamplesNumber = static_cast<unsigned int>(*std::min_element(labelsCount.begin(), labelsCount.end()) * thresholdForMaxNumberOfClassExamples);
	if (minorityClassExamplesNumber < numberOfClassExamples)
		return minorityClassExamplesNumber;
	return numberOfClassExamples;
}

SequentialGammaConfigWithFeatureSelection::SequentialGammaConfigWithFeatureSelection(const platform::Subtree& config,
	const dataset::Dataset<std::vector<float>, float>& traningSet, IDatasetLoader& loadingWorkflow)
	: SequentialGammaConfigWithFeatureSelection(config.getNode("Svm.SequentialGamma"),
		traningSet,
		SvmAlgorithmConfiguration(config), loadingWorkflow)

{
	m_featureSetOptimization = FeatureSetOptimizationWorkflowFactory::create(config, loadingWorkflow, "Svm.seq");
}

SequentialGammaConfigWithFeatureSelection::SequentialGammaConfigWithFeatureSelection(const platform::Subtree& config,
	const dataset::Dataset<std::vector<float>, float>& traningSet,
	SvmAlgorithmConfiguration svmConfig, IDatasetLoader& )
	: m_svmConfig(std::move(svmConfig))
	, m_populationSize(config.getValue<unsigned int>("PopulationSize"))
	, m_labelsCount(std::move(svmUtils::countLabels(config.getValue<unsigned int>("NumberOfClasses"), traningSet)))
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
	, m_training(std::make_shared<SvmTrainingCustomKernelFS>(m_svmConfig, m_svmConfig.m_estimationType == svmMetricType::Auc,
		config.getValue<std::string>("KernelType"),
		config.getValue<bool>("TrainAlpha")))
			, m_stopCondition(geneticComponents::StopConditionFactory::create<SvmCustomKernelFeaturesSelectionChromosome>(config))
			, m_crossover(
				std::make_shared<CrossoverCustomGauss>(random::RandomNumberGeneratorFactory::create(config), config.getValue<unsigned int>("NumberOfClasses")))
			, m_mutation(std::make_shared<MutationCustomGaussSequential>(random::RandomNumberGeneratorFactory::create(config),
				platform::Percent(config.getValue<double>("Mutation.GaSvm.ExchangePercent")),
				platform::Percent(config.getValue<double>("Mutation.GaSvm.MutationProbability")),
				traningSet,
				m_labelsCount))
			, m_selection(geneticComponents::SelectionFactory::create<SvmCustomKernelFeaturesSelectionChromosome>(config))
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
					, m_validationMethod(SvmValidationFactory::create<SvmCustomKernelFeaturesSelectionChromosome>(config, *m_svmConfig.m_estimationMethod))
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







} // namespace genetic
