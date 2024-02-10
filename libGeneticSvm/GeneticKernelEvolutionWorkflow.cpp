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
#include "libSvmComponents/ConfusionMatrixMetrics.h"
#include "GeneticKernelEvolutionWorkflow.h"
#include "WorkflowUtils.h"
#include "AllModelsLogger.h"
#include "libPlatform/loguru.hpp"
#include "libRandom/MersenneTwister64Rng.h"
#include "SvmLib/EnsembleSvm.h"
#include "SvmLib/libSvmImplementation.h"
#include "libSvmComponents/SvmKernelGridGeneration.h"

namespace genetic
{
using namespace svmComponents;
using namespace geneticComponents;

GeneticKernelEvolutionWorkflow::GeneticKernelEvolutionWorkflow(const SvmWokrflowConfiguration& config,
                                                               GeneticKernelEvolutionConfiguration algorithmConfig,
                                                               IDatasetLoader& workflow,
                                                               platform::Subtree fullConfig)
	: m_algorithmConfig(std::move(algorithmConfig))
	, m_trainingSvmClassifierElement(*m_algorithmConfig.m_training)
	, m_valdiationElement(*m_algorithmConfig.m_svmConfig.m_estimationMethod, false)
	, m_valdiationTestDataElement(*m_algorithmConfig.m_svmConfig.m_estimationMethod, true)
	, m_stopConditionElement(*m_algorithmConfig.m_stopCondition)
	, m_crossoverElement(*m_algorithmConfig.m_crossover)
	, m_mutationElement(*m_algorithmConfig.m_mutation)
	, m_selectionElement(*m_algorithmConfig.m_selection)
	, m_createPopulationElement(*m_algorithmConfig.m_populationGeneration)
	, m_savePngElement()
	, m_createVisualizationElement(m_algorithmConfig.m_svmConfig)
	, m_crossoverParentSelectionElement(*m_algorithmConfig.m_parentSelection)
	, m_trainingSet(nullptr)
	, m_validationSet(nullptr)
	, m_testSet(nullptr)
	, m_needRetrain(false)
	, m_loadingWorkflow(workflow)
	, m_generationNumber(0)
	, m_config(config)
	, m_allModelsLogger(nullptr)
	, m_timer(std::make_shared<Timer>())
	, m_fullConfig(fullConfig)
{
}

std::shared_ptr<phd::svm::ISvm> GeneticKernelEvolutionWorkflow::run()
{
	static unsigned int numberOfRun = 1;
	
	if(m_config.verbosity == platform::Verbosity::All)
	{
		auto outputPath = m_config.outputFolderPath.string();
		auto logger = std::make_shared<AllModelsLogger>(numberOfRun++, outputPath, m_loadingWorkflow);
		m_allModelsLogger = logger;
	}
	initialize();
	runGeneticAlgorithm();


	if(m_config.verbosity != platform::Verbosity::None)
	{
		m_resultLogger.logToFile(std::filesystem::path(m_config.outputFolderPath.string() + m_config.txtLogFilename));
	}

	return (getBestChromosomeInGeneration().getClassifier());
}

void GeneticKernelEvolutionWorkflow::logAllModels(geneticComponents::Population<svmComponents::SvmKernelChromosome>& population,
                                                  geneticComponents::Population<svmComponents::SvmKernelChromosome>& testPopulation)
{
	auto bestOneConfustionMatrix = population.getBestOne().getConfusionMatrix().value();
	auto trainingSetSize = m_trainingSet->size();
	auto featureNumber = m_validationSet->getSamples()[0].size();
	m_allModelsLogger->log(population,
	                       testPopulation,
	                       *m_timer,
	                       m_algorithmName,
	                       m_generationNumber,
	                       Accuracy(bestOneConfustionMatrix),
	                       featureNumber,
	                       trainingSetSize,
	                       bestOneConfustionMatrix);

	auto outputPaht = m_config.outputFolderPath.string();
	m_allModelsLogger->save(outputPaht + "\\" + std::to_string(m_allModelsLogger->getNumberOfRun()) + "\\populationTextLog.txt");
}

void GeneticKernelEvolutionWorkflow::initializeGeneticAlgorithm()
{
	try
	{
		auto population = m_createPopulationElement.launch(m_algorithmConfig.m_populationSize);
		population = m_trainingSvmClassifierElement.launch(population, *m_trainingSet);
		m_population = m_valdiationElement.launch(population, *m_validationSet);
		auto testPopulation = m_valdiationTestDataElement.launch(population, *m_testSet);

		if (m_algorithmConfig.m_svmConfig.m_doVisualization)
		{
			setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, m_config, m_generationNumber);

			auto svm = m_population.getBestOne().getClassifier();
			svmComponents::SvmVisualization visualization;
			auto image = visualization.createDetailedVisualization(*svm,
			                                                       m_algorithmConfig.m_svmConfig.m_height,
			                                                       m_algorithmConfig.m_svmConfig.m_width,
			                                                       *m_trainingSet, *m_validationSet, *m_testSet);

			m_savePngElement.launch(image, m_pngNameSource);
		}

		if(m_config.verbosity == platform::Verbosity::All)
		{
			logAllModels(population, testPopulation);
		}
	
		logResults(population, testPopulation);
	}
	catch (const std::exception& exception)
	{
		LOG_F(ERROR, "Error: %s", exception.what());
	}
}

void GeneticKernelEvolutionWorkflow::logResults(const Population<SvmKernelChromosome>& population,
                                                const Population<SvmKernelChromosome>& testPopulation)
{
	auto trainingSetSize = m_trainingSet->size();
	auto bestOne = population.getBestOne();
	auto bestOneConfustionMatrix = bestOne.getConfusionMatrix().value();
	auto featureNumber = m_validationSet->getSamples()[0].size();
	auto bestOneIndex = m_population.getBestIndividualIndex();

	m_resultLogger.createLogEntry(population,
	                              testPopulation,
	                              *m_timer,
	                              m_algorithmName,
	                              m_generationNumber++,
	                              Accuracy(bestOneConfustionMatrix),
	                              featureNumber,
	                              trainingSetSize,
	                              bestOneConfustionMatrix,
	                              testPopulation[bestOneIndex].getConfusionMatrix().value());
}

void GeneticKernelEvolutionWorkflow::runGeneticAlgorithm()
{
	try
	{
		if (m_needRetrain)
		{
			retrainPopulation<SvmKernelChromosome>(*m_trainingSet,
			                                       *m_validationSet,
			                                       m_population,
			                                       *m_algorithmConfig.m_training,
			                                       *m_algorithmConfig.m_svmConfig.m_estimationMethod);
			m_needRetrain = false;
		}

		bool isStop = false;

		while (!isStop)
		{
			auto parents = m_crossoverParentSelectionElement.launch(m_population);
			auto newPopulation = m_crossoverElement.launch(parents);
			m_mutationElement.launch(newPopulation);
			m_trainingSvmClassifierElement.launch(newPopulation, *m_trainingSet);
			m_valdiationElement.launch(newPopulation, *m_validationSet);
			auto nextGeneration = m_selectionElement.launch(m_population, newPopulation);

			auto nextGeneration2 = nextGeneration;
			m_valdiationTestDataElement.launch(nextGeneration2, *m_testSet); //copy in here

			m_population = m_selectionElement.launch(m_population, nextGeneration);

			if (m_algorithmConfig.m_svmConfig.m_doVisualization)
			{
				setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, m_config, m_generationNumber);

				auto svm = m_population.getBestOne().getClassifier();
				svmComponents::SvmVisualization visualization;
				auto image = visualization.createDetailedVisualization(*svm,
				                                                       m_algorithmConfig.m_svmConfig.m_height,
				                                                       m_algorithmConfig.m_svmConfig.m_width,
				                                                       *m_trainingSet, *m_validationSet, *m_testSet);

				m_savePngElement.launch(image, m_pngNameSource);
			}


			if(m_config.verbosity == platform::Verbosity::All)
			{
				logAllModels(m_population, nextGeneration2);
			}
			logResults(m_population, nextGeneration2);
			isStop = m_stopConditionElement.launch(nextGeneration);
		}
	}
	catch (const std::exception& exception)
	{
		LOG_F(ERROR, "Error: %s", exception.what());
	}
}

void GeneticKernelEvolutionWorkflow::initialize()
{
	static unsigned int numberOfRun = 1;
	auto outputPaht = m_config.outputFolderPath.string();
	if (m_allModelsLogger == nullptr && m_config.verbosity == platform::Verbosity::All)
	{
		auto logger = std::make_shared<AllModelsLogger>(numberOfRun++, outputPaht, m_loadingWorkflow);
		m_allModelsLogger = logger;
	}

	if (m_trainingSet == nullptr)
	{
		m_trainingSet = &m_loadingWorkflow.getTraningSet();
	}
	if (m_validationSet == nullptr)
	{
		m_validationSet = &m_loadingWorkflow.getValidationSet();
	}
	if (m_testSet == nullptr)
	{
		m_testSet = &m_loadingWorkflow.getTestSet();
	}

	initializeGeneticAlgorithm();
}

SvmKernelChromosome GeneticKernelEvolutionWorkflow::getBestChromosomeInGeneration() const
{
	return m_population.getBestOne();
}

Population<SvmKernelChromosome> GeneticKernelEvolutionWorkflow::getPopulation() const
{
	return m_population;
}

void GeneticKernelEvolutionWorkflow::setDatasets(const dataset::Dataset<std::vector<float>, float>& trainingSet,
                                                 const dataset::Dataset<std::vector<float>, float>& validationSet,
                                                 const dataset::Dataset<std::vector<float>, float>& testSet)
{
	if (trainingSet.empty())
	{
		throw EmptyDatasetException(DatasetType::Training);
	}
	if (validationSet.empty())
	{
		throw EmptyDatasetException(DatasetType::Validation);
	}
	if (testSet.empty())
	{
		throw EmptyDatasetException(DatasetType::Test);
	}
	m_trainingSet222222 = trainingSet; //copy in here good to delete this
	m_validationSet2 = validationSet;
	m_testSet2 = testSet;
	m_trainingSet = &m_trainingSet222222;
	m_validationSet = &m_validationSet2;
	m_testSet = &m_testSet2;

	//retrain because of different number of features in dataset
	m_needRetrain = true;
}

geneticComponents::Population<svmComponents::SvmKernelChromosome> GeneticKernelEvolutionWorkflow::initNoEvaluate(int popSize)
{
	try
	{
		auto population = m_createPopulationElement.launch(popSize);

		return population;
	}
	catch (const std::exception& exception)
	{
		LOG_F(ERROR, "Error: %s", exception.what());
		throw;
	}
}

void GeneticKernelEvolutionWorkflow::performGeneticOperations(geneticComponents::Population<svmComponents::SvmKernelChromosome>& population)
{
	m_population = population;

	auto parents = m_crossoverParentSelectionElement.launch(m_population);

	if (m_parents && !m_parents->empty()) //for SE-SVM_Corrected version where parents are selected once for all part of chromosome
	{
		parents = *m_parents;
	}

	auto newPopulation = m_crossoverElement.launch(parents);
	m_mutationElement.launch(newPopulation);

	population = newPopulation;
}

void GeneticKernelEvolutionWorkflow::setTimer(std::shared_ptr<Timer> timer)
{
	m_timer = timer;
}

geneticComponents::Population<svmComponents::SvmKernelChromosome> GeneticKernelEvolutionWorkflow::initNoEvaluate(int popSize, int seed)
{
	try
	{
		auto initialKernelParametersRange = svmUtils::getRange("Svm.GeneticKernelEvolution.Generation.Grid", m_fullConfig);
		auto isRegression = m_fullConfig.getValue<bool>("Svm.GeneticKernelEvolution.Generation.isRegression");
		auto popGenerator = std::make_unique<SvmKernelGridGeneration>(initialKernelParametersRange,
		                                                              m_algorithmConfig.m_svmConfig.m_kernelType,
		                                                              std::make_unique<random::MersenneTwister64Rng>(seed),
		                                                              isRegression,
																		*m_trainingSet);

		auto population = popGenerator->createPopulation(popSize);

		return population;
	}
	catch (const std::exception& exception)
	{
		LOG_F(ERROR, "Error: %s", exception.what());
		throw;
	}
}

void GeneticKernelEvolutionWorkflow::setupTrainingSet(const dataset::Dataset<std::vector<float>, float>& trainingSet)
{
	m_trainingSet222222 = trainingSet; //copy in here good to delete this
	m_trainingSet = &m_trainingSet222222;
	m_needRetrain = true;
}

GeneticWorkflowResultLogger& GeneticKernelEvolutionWorkflow::getResultLogger()
{
	return m_resultLogger;
}
} // namespace genetic
