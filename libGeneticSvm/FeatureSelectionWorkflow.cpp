#include "libSvmStrategies/SvmTrainingStrategy.h"
#include "libSvmComponents/SvmValidationStrategy.h"
#include "libGeneticStrategies/StopConditionStrategy.h"
#include "libGeneticStrategies/CrossoverStrategy.h"
#include "libGeneticStrategies/MutationStrategy.h"
#include "libGeneticStrategies/SelectionStrategy.h"
#include "libGeneticStrategies/CreatePopulationStrategy.h"
#include "libStrategies/FileSinkStrategy.h"
#include "libGeneticStrategies/CrossoverParentSelectionStrategy.h"
#include "libSvmComponents/SvmTraining.h"
#include "libSvmComponents/ConfusionMatrixMetrics.h"
#include "FeatureSelectionWorkflow.h"
#include "WorkflowUtils.h"
#include "SvmExceptions.h"
#include "libGeneticStrategies/AddToBinaryCacheStrategy.h"
#include "libGeneticStrategies/UseBinaryCacheStrategy.h"
#include "libGeneticStrategies/CombinePopulationsStrategy.h"
#include "libPlatform/loguru.hpp"

namespace genetic
{
using namespace svmComponents;
using namespace geneticComponents;

FeatureSelectionWorkflow::FeatureSelectionWorkflow(const SvmWokrflowConfiguration& config,
                                                   GeneticFeatureSetEvolutionConfiguration algorithmConfig,
                                                   IDatasetLoader& loadingWorkflow)
    : m_algorithmConfig(std::move(algorithmConfig))
    , m_trainingSvmClassifierElement(*m_algorithmConfig.m_training)
    , m_validationElement(*m_algorithmConfig.m_svmConfig.m_estimationMethod, false)
    , m_validationTestDataElement(*m_algorithmConfig.m_svmConfig.m_estimationMethod, true)
    , m_stopConditionElement(*m_algorithmConfig.m_stopCondition)
    , m_crossoverElement(*m_algorithmConfig.m_crossover)
    , m_mutationElement(*m_algorithmConfig.m_mutation)
    , m_selectionElement(*m_algorithmConfig.m_selection)
    , m_createPopulationElement(*m_algorithmConfig.m_populationGeneration)
    , m_crossoverParentSelectionElement(*m_algorithmConfig.m_parentSelection)
    , m_addToCacheElement(m_cache)
    , m_useCacheElement(m_cache)
    , m_combinePopulationElement()
    , m_trainingSet(nullptr)
    , m_validationSet(nullptr)
    , m_testSet(nullptr)
    , m_needRetrain(false)
    , m_loadingWorkflow(loadingWorkflow)
    , m_generationNumber(0)
    , m_config(config)
{
}

void FeatureSelectionWorkflow::initialize()
{
    if (m_trainingSet == nullptr)
    {
        m_trainingSet = &m_loadingWorkflow.getTraningSet();
    }
    m_validationSet = &m_loadingWorkflow.getValidationSet();  //why only traning set
    m_testSet = &m_loadingWorkflow.getTestSet();
    
    initializeGeneticAlgorithm();
}

void FeatureSelectionWorkflow::logResult(const Population<SvmFeatureSetChromosome>& population,
                                         const Population<SvmFeatureSetChromosome>& testpopulation)
{
    auto bestOne = population.getBestOne();
    auto bestOneConfustionMatrix = bestOne.getConfusionMatrix().value();

    auto genes = bestOne.getGenes();
    auto featuresNumber = std::accumulate(std::begin(genes), std::end(genes), 0);
    auto trainingSetSize = m_trainingSet->size();
    std::stringstream genesAsString;
    std::copy(genes.begin(), genes.end(), std::ostream_iterator<int>(genesAsString, ""));

    m_resultLogger.createLogEntry(population,
                                  testpopulation,
                                  m_timer,
                                  m_algorithmName,
                                  m_generationNumber,
                                  Accuracy(bestOneConfustionMatrix),
                                  featuresNumber,
                                  trainingSetSize,
                                  bestOneConfustionMatrix,
                                  genesAsString.str());
    ++m_generationNumber;
}

void FeatureSelectionWorkflow::runGeneticAlgorithm()
{
    try
    {
        if (m_needRetrain)
        {
            m_cache.clearCache();
            retrainPopulation<SvmFeatureSetChromosome>(*m_trainingSet,
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
            newPopulation = m_mutationElement.launch(newPopulation);
            
            auto [unknownFitness, knownFittness] = m_useCacheElement.launch(newPopulation);
            if(!unknownFitness.empty())
            {
                unknownFitness = m_trainingSvmClassifierElement.launch(unknownFitness, *m_trainingSet);
                unknownFitness = m_validationElement.launch(unknownFitness, *m_validationSet);
                m_addToCacheElement.launch(unknownFitness);
            }
            auto combined = m_combinePopulationElement.launch(unknownFitness, knownFittness);

            auto nextGeneration = m_selectionElement.launch(combined, m_population);

            m_population = nextGeneration;

            m_validationTestDataElement.launch(nextGeneration, *m_testSet);

            logResult(m_population, nextGeneration);
            isStop = m_stopConditionElement.launch(m_population);
        }
    }
    catch (const std::runtime_error& exception)
    {
		LOG_F(ERROR, "Error: %s", exception.what());
        std::cout << exception.what();
    }
}

SvmFeatureSetChromosome FeatureSelectionWorkflow::getBestChromosomeInGeneration() const
{
    auto population = m_population;
    return population.getBestOne();
}

Population<SvmFeatureSetChromosome> FeatureSelectionWorkflow::getPopulation() const
{
    return m_population;
}

GeneticWorkflowResultLogger& FeatureSelectionWorkflow::getResultLogger()
{
    return m_resultLogger;
}

void FeatureSelectionWorkflow::setupKernelParameters(const svmComponents::SvmKernelChromosome& kernelParameters)
{
    auto trainingElement = dynamic_cast<SvmTraining<SvmFeatureSetChromosome>*>(m_algorithmConfig.m_training.get());
    if (trainingElement != nullptr)
    {
        trainingElement->updateParameters(kernelParameters);
        m_needRetrain = true;
        return;
    }
    throw BadTrainingElement();
}

dataset::Dataset<std::vector<float>, float> FeatureSelectionWorkflow::getFilteredTraningSet()
{
    return getBestChromosomeInGeneration().convertChromosome(*m_trainingSet);
}

dataset::Dataset<std::vector<float>, float> FeatureSelectionWorkflow::getFilteredValidationSet()
{
    return getBestChromosomeInGeneration().convertChromosome(*m_validationSet);
}

dataset::Dataset<std::vector<float>, float> FeatureSelectionWorkflow::getFilteredTestSet()
{
    return getBestChromosomeInGeneration().convertChromosome(*m_testSet);
}


void FeatureSelectionWorkflow::initializeGeneticAlgorithm()
{
    try
    {
        auto population = m_createPopulationElement.launch(m_algorithmConfig.m_populationSize);
        m_trainingSvmClassifierElement.launch(population, *m_trainingSet);
        m_population =  m_validationElement.launch(population, *m_validationSet);
        const auto testPopulation =  m_validationTestDataElement.launch(population, *m_testSet);
        m_addToCacheElement.launch(m_population);


        logResult(m_population, testPopulation);
    }
    catch (const std::runtime_error& exception)
    {
		LOG_F(ERROR, "Error: %s", exception.what());
        std::cout << exception.what();
    }
}

std::shared_ptr<phd::svm::ISvm> FeatureSelectionWorkflow::run()
{
    initialize();
    runGeneticAlgorithm();
    m_resultLogger.logToFile(std::filesystem::path(m_config.outputFolderPath.string() + m_config.txtLogFilename));

    return getBestChromosomeInGeneration().getClassifier();
}

void FeatureSelectionWorkflow::setupTrainingSet(const dataset::Dataset<std::vector<float>, float>& trainingSet)
{
    m_trainingSet22222 = trainingSet;
    m_trainingSet = &m_trainingSet22222;
    m_needRetrain = true;
}
} // namespace genetic
