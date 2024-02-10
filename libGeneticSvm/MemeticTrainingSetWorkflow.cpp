#include "libSvmComponents/SvmValidationStrategy.h"
#include "libGeneticStrategies/StopConditionStrategy.h"
#include "libGeneticStrategies/CrossoverStrategy.h"
#include "libGeneticStrategies/MutationStrategy.h"
#include "libGeneticStrategies/SelectionStrategy.h"
#include "libGeneticStrategies/CreatePopulationStrategy.h"
#include "libStrategies/FileSinkStrategy.h"
#include "libSvmStrategies/CreateSvmVisualizationStrategy.h"
#include "libSvmStrategies/SvmTrainingStrategy.h"
#include "libSvmStrategies/MemeticEducationStrategy.h"
#include "libSvmStrategies/CrossoverCompensationStrategy.h"
#include "libGeneticStrategies/CombinePopulationsStrategy.h"
#include "libSvmStrategies/UpdateSupportVectorPoolStrategy.h"
#include "libSvmStrategies/SuperIndividualCreationStrategy.h"
#include "libSvmStrategies/MemeticAdaptationStrategy.h"
#include "libSvmStrategies/CompensationInformationStrategy.h"
#include "LibGeneticComponents/LocalGlobalAdaptationSelection.h"
#include "libSvmComponents/SvmTraining.h"
#include "libGeneticStrategies/CrossoverParentSelectionStrategy.h"
#include "libSvmComponents/ConfusionMatrixMetrics.h"
#include "SvmExceptions.h"
#include "MemeticTrainingSetWorkflow.h"

#include "WorkflowUtils.h"
#include "libPlatform/loguru.hpp"
#include "SvmLib/EnsembleSvm.h"
#include "libSvmComponents/GaSvmGeneration.h"
#include "libSvmComponents/SvmTrainingSetPopulationFactory.h"

namespace genetic
{
using namespace svmComponents;
using namespace geneticComponents;

MemeticTraningSetWorkflow::MemeticTraningSetWorkflow(const SvmWokrflowConfiguration& config,
                                                     MemeticTrainingSetEvolutionConfiguration algorithmConfig,
                                                     IDatasetLoader& workflow,
                                                     platform::Subtree fullConfig)
    : m_algorithmConfig(std::move(algorithmConfig))
    , m_trainingSvmClassifierElement(*m_algorithmConfig.m_training)
    , m_validationElement(m_algorithmConfig.m_validationMethod)
    , m_valdiationTestDataElement(*m_algorithmConfig.m_svmConfig.m_estimationMethod, true)
    , m_stopConditionElement(*m_algorithmConfig.m_stopCondition)
    , m_crossoverElement(*m_algorithmConfig.m_crossover)
    , m_mutationElement(*m_algorithmConfig.m_mutation)
    , m_selectionElement(*m_algorithmConfig.m_selection)
    , m_createPopulationElement(*m_algorithmConfig.m_populationGeneration)
    , m_savePngElement()
    , m_createVisualizationElement(m_algorithmConfig.m_svmConfig)
    , m_educationElement(*m_algorithmConfig.m_education)
    , m_crossoverCompensationElement(*m_algorithmConfig.m_compensation)
    , m_adaptationElement(*m_algorithmConfig.m_adaptation)
    , m_supportVectorPoolElement(*m_algorithmConfig.m_supporVectorPool)
    , m_superIndividualsGenerationElement(*m_algorithmConfig.m_superIndivudualsGeneration)
    , m_populationCombinationElement()
    , m_trainingSuperIndividualsElement(*m_algorithmConfig.m_training)
    , m_validationSuperIndividualsElement(m_algorithmConfig.m_validationMethod)
    , m_parentSelectionElement(*m_algorithmConfig.m_parentSelection)
    , m_compensationGenerationElement(*m_algorithmConfig.m_compensationGeneration)
    , m_numberOfClassExamples(m_algorithmConfig.m_initialNumberOfClassExamples)
    , m_svPool(std::vector<DatasetVector>())
    , m_trainingSet(nullptr)
    , m_validationSet(nullptr)
    , m_testSet(nullptr)
    , m_needRetrain(false)
    , m_loadingWorkflow(workflow)
    , m_generationNumber(0)
    , m_config(config)
    , m_allModelsLogger(nullptr)
    , m_timer(std::make_shared<Timer>())
	, m_trainingSetInterface(m_algorithmConfig.m_trainingSetInterface)
	, m_fullConfig(fullConfig)
{
}

std::shared_ptr<phd::svm::ISvm> MemeticTraningSetWorkflow::run()
{
    static unsigned int numberOfRun = 1;
    
    if(m_config.verbosity == platform::Verbosity::All)
	{
        auto outputPaht = m_config.outputFolderPath.string();
        auto  logger = std::make_shared<AllModelsLogger>( numberOfRun++, outputPaht, m_loadingWorkflow );
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

void MemeticTraningSetWorkflow::logResults(const Population<SvmTrainingSetChromosome>& population,
                                           const Population<SvmTrainingSetChromosome>& testPopulation)
{
    auto bestOneConfustionMatrix = population.getBestOne().getConfusionMatrix().value();
    auto validationDataset = *m_validationSet;
    auto featureNumber = validationDataset.getSamples()[0].size();
	auto bestOneIndex = m_population.getBestIndividualIndex();

    m_resultLogger.createLogEntry(population,
								  testPopulation,
                                  *m_timer,
                                  m_algorithmName,
                                  m_generationNumber++,
                                  Accuracy(bestOneConfustionMatrix),
                                  featureNumber,
                                  m_numberOfClassExamples * m_algorithmConfig.m_labelsCount.size(),
                                  bestOneConfustionMatrix,
								  testPopulation[bestOneIndex].getConfusionMatrix().value());
}

void MemeticTraningSetWorkflow::logAllModels(geneticComponents::Population<SvmTrainingSetChromosome>& testPopulation)
{
    auto bestOneConfustionMatrix = m_population.getBestOne().getConfusionMatrix().value();
    auto validationDataset = *m_validationSet;
    auto featureNumber = validationDataset.getSamples()[0].size();
    m_allModelsLogger->log(m_population,
                           testPopulation,
                           *m_timer,
                           m_algorithmName,
                           m_generationNumber,
                           Accuracy(bestOneConfustionMatrix),
                           featureNumber,
                           m_numberOfClassExamples * m_algorithmConfig.m_labelsCount.size(),
                           bestOneConfustionMatrix);

    auto outputPaht = m_config.outputFolderPath.string();
    m_allModelsLogger->save(outputPaht + "\\" + std::to_string(m_allModelsLogger->getNumberOfRun()) + "\\populationTextLog.txt");
}

unsigned MemeticTraningSetWorkflow::getInitialTrainingSetSize()
{
    
	
    return m_algorithmConfig.m_initialNumberOfClassExamples;
}

 unsigned MemeticTraningSetWorkflow::getCurrentTrainingSetSize()
{
	return m_algorithmConfig.m_adaptation->getNumberOfClassExamples();
}

void MemeticTraningSetWorkflow::setK(unsigned k)
{
	m_algorithmConfig.m_adaptation->setK(k);
}

phd::svm::ISvm& MemeticTraningSetWorkflow::getClassifierWithBestDistances()
{
    return *m_algorithmConfig.m_adaptation->getClassifierWithBestDistance().get();
}

svmStrategies::MemeticAdaptationStrategy& MemeticTraningSetWorkflow::getAdaptationElement()
{
	return m_adaptationElement;
}

geneticComponents::Population<svmComponents::SvmTrainingSetChromosome> MemeticTraningSetWorkflow::initNoEvaluate(int popSize, int seed)
{
    try
    {
        if (m_trainingSet == nullptr)
        {
            m_trainingSet = &m_trainingSetInterface->trainingSet();
            m_trainingSetInterface = std::make_shared<FullTrainingSet>(*m_trainingSet);
            m_validationSet = &m_loadingWorkflow.getValidationSet();
            m_testSet = &m_loadingWorkflow.getTestSet();
        }



            auto numberOfClassExamples = getNumberOfClassExamples2(m_fullConfig.getValue<unsigned int>("Svm.MemeticTrainingSetSelection.NumberOfClassExamples"),
                m_algorithmConfig.m_labelsCount); //fix for small highly imbalanced datasets

        auto popGenerator = std::make_unique<GaSvmGeneration>(*m_trainingSet,
                                                              std::make_unique<random::MersenneTwister64Rng>(seed),
                                                              numberOfClassExamples,
                                                              m_algorithmConfig.m_labelsCount);
        


    	
        auto population = popGenerator->createPopulation(popSize);


    	return population;
    }
    catch (const std::exception& exception)
    {
        LOG_F(ERROR, "Error: %s", exception.what());
        throw;
    }
	
}

void MemeticTraningSetWorkflow::runGeneticAlgorithm()
{
    try
    {
        if (m_needRetrain)
        {
            retrainPopulation<SvmTrainingSetChromosome>(*m_trainingSet,
                                                        *m_validationSet,
                                                        m_population,
                                                        *m_algorithmConfig.m_training,
                                                        *m_algorithmConfig.m_svmConfig.m_estimationMethod);
            m_needRetrain = false;
        }

        bool isStop = false;

        while (!isStop)
        {
            auto parents = m_parentSelectionElement.launch(m_population);
            auto newPopulation = m_crossoverElement.launch(parents);

            auto compensantionInfo = m_compensationGenerationElement.launch(parents, m_numberOfClassExamples);
            auto result = m_crossoverCompensationElement.launch(newPopulation, compensantionInfo);

            //auto populationEducated = m_educationElement.launch(result, m_svPool, parents, *m_trainingSet);
            auto populationEducated = m_educationElement.launch(result, m_svPool, parents, m_trainingSetInterface->trainingSet());

            populationEducated = m_mutationElement.launch(populationEducated);

            //auto poptrained = m_trainingSvmClassifierElement.launch(populationEducated, *m_trainingSet);
            auto poptrained = m_trainingSvmClassifierElement.launch(populationEducated, m_trainingSetInterface->trainingSet());
            auto afterValidtion = m_validationElement->launch(populationEducated, *m_validationSet);

            //m_svPool = m_supportVectorPoolElement.launch(poptrained, *m_trainingSet);
            m_svPool = m_supportVectorPoolElement.launch(poptrained, m_trainingSetInterface->trainingSet());

            auto superIndividualsSize = static_cast<unsigned int>(m_algorithmConfig.m_populationSize * m_algorithmConfig.m_superIndividualAlpha);
            auto superIndividualsPopulation = m_superIndividualsGenerationElement.launch(superIndividualsSize, m_svPool, m_numberOfClassExamples);

            //m_trainingSuperIndividualsElement.launch(superIndividualsPopulation, *m_trainingSet);
            m_trainingSuperIndividualsElement.launch(superIndividualsPopulation, m_trainingSetInterface->trainingSet());
            m_validationSuperIndividualsElement->launch(superIndividualsPopulation, *m_validationSet);

            auto combinedPopulation = m_populationCombinationElement.launch(afterValidtion, superIndividualsPopulation);
            m_population = m_selectionElement.launch(m_population, combinedPopulation);

            auto [IsModeLocal, NumberOfClassExamples] = m_adaptationElement.launch(m_population);
            m_numberOfClassExamples = NumberOfClassExamples;

            auto copy = m_population;
            auto testPopulation = m_valdiationTestDataElement.launch(copy, *m_testSet);

            isStop = m_stopConditionElement.launch(m_population);

            if (m_algorithmConfig.m_svmConfig.m_doVisualization)
            {
                setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, m_config, m_generationNumber);
                auto image = m_createVisualizationElement.launch(m_population, m_trainingSetInterface->trainingSet(), *m_testSet);
                m_savePngElement.launch(image, m_pngNameSource);
            }

            auto localGlobal = dynamic_cast<LocalGlobalAdaptationSelection<SvmTrainingSetChromosome>*>(m_algorithmConfig.m_parentSelection.get());
            if (localGlobal != nullptr)
            {
                localGlobal->setMode(IsModeLocal);
            }
            
            if(m_config.verbosity == platform::Verbosity::All)
            {
                logAllModels(testPopulation);
            }
            logResults(m_population, testPopulation);
        }
    }
    catch (const std::exception& exception)
    {
		LOG_F(ERROR, "Error: %s", exception.what());
    }
}

void MemeticTraningSetWorkflow::initialize()
{
    static unsigned int numberOfRun = 1;
    auto outputPaht = m_config.outputFolderPath.string();
    if (m_allModelsLogger == nullptr && m_config.verbosity == platform::Verbosity::All)
    {
        auto  logger = std::make_shared<AllModelsLogger>(numberOfRun++, outputPaht, m_loadingWorkflow);
        m_allModelsLogger = logger;
    }

    if (m_trainingSet == nullptr)
    {
        m_trainingSet = &m_loadingWorkflow.getTraningSet();
        m_validationSet = &m_loadingWorkflow.getValidationSet();
        m_testSet = &m_loadingWorkflow.getTestSet();
    }
    initializeGeneticAlgorithm();
}

void MemeticTraningSetWorkflow::setupKernelParameters(const SvmKernelChromosome& kernelParameters)
{
    auto trainingElement = dynamic_cast<SvmTraining<SvmTrainingSetChromosome>*>(m_algorithmConfig.m_training.get());
    if (trainingElement != nullptr)
    {
        trainingElement->updateParameters(kernelParameters);
        m_needRetrain = true;
        return;
    }
    throw BadTrainingElement();
}

GeneticWorkflowResultLogger& MemeticTraningSetWorkflow::getResultLogger()
{
    return m_resultLogger;
}

SvmTrainingSetChromosome MemeticTraningSetWorkflow::getBestChromosomeInGeneration() const
{
    auto population = m_population;
    return population.getBestOne();
}

Population<SvmTrainingSetChromosome> MemeticTraningSetWorkflow::getPopulation() const
{
    return m_population;
}

dataset::Dataset<std::vector<float>, float> MemeticTraningSetWorkflow::getBestTrainingSet() const
{
    auto population = m_population;
    return population.getBestOne().convertChromosome(m_loadingWorkflow.getTraningSet());
}

void MemeticTraningSetWorkflow::setupFeaturesSet(const SvmFeatureSetChromosome& featureSetChromosome)
{
	//TODO introduce the use of m_trainingSetInterface
    m_trainingSet2 = featureSetChromosome.convertChromosome((m_loadingWorkflow.getTraningSet()));
    m_trainingSet = &m_trainingSet2;

    m_trainingSetInterface = std::make_shared<FullTrainingSet>(m_trainingSet2);

    m_validationSet2 = featureSetChromosome.convertChromosome((m_loadingWorkflow.getValidationSet()));
    m_validationSet = &m_validationSet2;

    m_testSet2 = featureSetChromosome.convertChromosome((m_loadingWorkflow.getTestSet()));
    m_testSet = &m_testSet2;

    // @wdudzik retrain because of different number of features in dataset
    m_needRetrain = true;
}

geneticComponents::Population<svmComponents::SvmTrainingSetChromosome> MemeticTraningSetWorkflow::initNoEvaluate(int popSize)
{
    try
    {
        if (m_trainingSet == nullptr)
        {
			m_trainingSet = &m_trainingSetInterface->trainingSet();
            m_trainingSetInterface = std::make_shared<FullTrainingSet>(*m_trainingSet);
            m_validationSet = &m_loadingWorkflow.getValidationSet();
            m_testSet = &m_loadingWorkflow.getTestSet();
        }

        auto population = m_createPopulationElement.launch(popSize);
        return population;
    }
    catch (const std::exception& exception)
    {
		LOG_F(ERROR, "Error: %s", exception.what());
        throw;
    }
}

void MemeticTraningSetWorkflow::performGeneticOperations(geneticComponents::Population<svmComponents::SvmTrainingSetChromosome>& population)
{
    m_population = population;

   
    auto parents = m_parentSelectionElement.launch(m_population);

    if (m_parents && !m_parents->empty())  //for SE-SVM_Corrected version where parents are selected once for all part of chromosome
    {
        parents = *m_parents;
    }
	
    auto newPopulation = m_crossoverElement.launch(parents);

    auto compensantionInfo = m_compensationGenerationElement.launch(parents, m_numberOfClassExamples);
    auto result = m_crossoverCompensationElement.launch(newPopulation, compensantionInfo);

    auto populationEducated = m_educationElement.launch(result, m_svPool, parents, m_trainingSetInterface->trainingSet());

    populationEducated = m_mutationElement.launch(populationEducated);

    //TODO think about using this population is ok in here :P
    m_svPool = m_supportVectorPoolElement.launch(population, m_trainingSetInterface->trainingSet());

    Population<SvmTrainingSetChromosome> superIndividualsPopulation;
	
	if (m_svPool.empty())
	{
		//Fix for some rare situations where we obtain trivial SVMs without any SVs
        superIndividualsPopulation = Population<SvmTrainingSetChromosome>({m_population[0], m_population[1]});
	}
    else
    {
	    auto superIndividualsSize = static_cast<unsigned int>(m_algorithmConfig.m_populationSize * m_algorithmConfig.m_superIndividualAlpha);
	    superIndividualsPopulation = m_superIndividualsGenerationElement.launch(superIndividualsSize, m_svPool, m_numberOfClassExamples);

        //Fix rare problem where svPool have vectors from only one class
        if(superIndividualsPopulation[0].size() == 0)
        {
            superIndividualsPopulation = Population<SvmTrainingSetChromosome>({ m_population[0], m_population[1] });
        }
    }

	auto combinedPopulation = m_populationCombinationElement.launch(populationEducated, superIndividualsPopulation);

    auto[IsModeLocal, NumberOfClassExamples] = m_adaptationElement.launch(m_population);
    m_numberOfClassExamples = NumberOfClassExamples;

    population = combinedPopulation;

    auto localGlobal = dynamic_cast<LocalGlobalAdaptationSelection<SvmTrainingSetChromosome>*>(m_algorithmConfig.m_parentSelection.get());
    if (localGlobal != nullptr)
    {
        localGlobal->setMode(IsModeLocal);
    }
}

void MemeticTraningSetWorkflow::setTimer(std::shared_ptr<Timer> timer)
{
    m_timer = timer;
}

void MemeticTraningSetWorkflow::initializeGeneticAlgorithm()
{
    try
    {
        auto population = m_createPopulationElement.launch(m_algorithmConfig.m_populationSize);

        m_trainingSvmClassifierElement.launch(population, m_trainingSetInterface->trainingSet());
        m_population = m_validationElement->launch(population, *m_validationSet);
        auto testPopulation = m_valdiationTestDataElement.launch(population, *m_testSet);

        if (m_algorithmConfig.m_svmConfig.m_doVisualization)
        {
            setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, m_config, m_generationNumber);
            auto image = m_createVisualizationElement.launch(population, m_trainingSetInterface->trainingSet(), *m_testSet);
            m_savePngElement.launch(image, m_pngNameSource);
        }

        if(m_config.verbosity == platform::Verbosity::All)
        {
            logAllModels(testPopulation);
        }
        logResults(m_population, testPopulation);
    }
    catch (const std::exception& exception)
    {
		LOG_F(ERROR, "Error: %s", exception.what());
		throw;
    }
}
} // namespace genetic
