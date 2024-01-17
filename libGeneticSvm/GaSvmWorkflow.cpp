//#include "libStrategies/TabularDataProviderStrategy.h"
#include "libStrategies/FileSinkStrategy.h"
#include "libSvmStrategies/CreateSvmVisualizationStrategy.h"
#include "libGeneticStrategies/CreatePopulationStrategy.h"
#include "libGeneticStrategies/SelectionStrategy.h"
#include "libGeneticStrategies/MutationStrategy.h"
#include "libGeneticStrategies/CrossoverStrategy.h"
#include "libGeneticStrategies/StopConditionStrategy.h"
#include "libSvmComponents/SvmValidationStrategy.h"
#include "libSvmStrategies/SvmTrainingStrategy.h"
#include "libSvmComponents/SvmTraining.h"
#include "libGeneticStrategies/CrossoverParentSelectionStrategy.h"
#include "libSvmComponents/ConfusionMatrixMetrics.h"
#include "WorkflowUtils.h"
#include "GaSvmWorkflow.h"
#include "SvmExceptions.h"
#include "libPlatform/loguru.hpp"
#include "SvmLib/EnsembleSvm.h"
#include "SvmLib/libSvmImplementation.h"

namespace genetic
{
using namespace svmComponents;
using namespace geneticComponents;

GaSvmWorkflow::GaSvmWorkflow(const SvmWokrflowConfiguration& config,
                             GeneticTrainingSetEvolutionConfiguration algorithmConfig,
                             IDatasetLoader& workflow)
    : m_algorithmConfig(std::move(algorithmConfig))
    , m_trainingSvmClassifierElement(*m_algorithmConfig.m_training)
    , m_validationElement(m_algorithmConfig.m_validationMethod)
    , m_validationTestDataElement(*m_algorithmConfig.m_svmConfig.m_estimationMethod, true)
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
    , m_trainingSetInterface(m_algorithmConfig.m_trainingSetInterface)
{
}

std::shared_ptr<phd::svm::ISvm> GaSvmWorkflow::run()
{
    initialize();
    runGeneticAlgorithm();
    m_resultLogger.logToFile(std::filesystem::path(m_config.outputFolderPath.string() + m_config.txtLogFilename));

    return (getBestChromosomeInGeneration().getClassifier());
}


void GaSvmWorkflow::logResults(const Population<SvmTrainingSetChromosome>& population, const Population<SvmTrainingSetChromosome>& testPopulation)
{
    auto bestOneConfustionMatrix = population.getBestOne().getConfusionMatrix().value();
    auto validationDataset = *m_validationSet;
    auto featureNumber = validationDataset.getSamples()[0].size();
    auto bestOneIndex = m_population.getBestIndividualIndex();

    m_resultLogger.createLogEntry(population,
                                  testPopulation,
                                  m_timer,
                                  m_algorithmName,
                                  m_generationNumber,
                                  Accuracy(bestOneConfustionMatrix),
                                  featureNumber,
								  m_algorithmConfig.m_numberOfClassExamples * m_algorithmConfig.m_labelsCount.size(),
                                  bestOneConfustionMatrix,
                                  testPopulation[bestOneIndex].getConfusionMatrix().value());
}

void GaSvmWorkflow::initializeGeneticAlgorithm()
{
    try
    {
        auto population = m_createPopulationElement.launch(m_algorithmConfig.m_populationSize);
        m_trainingSvmClassifierElement.launch(population, m_trainingSetInterface->trainingSet());
        m_population = m_validationElement->launch(population, *m_validationSet);
        auto testPopulation = m_validationTestDataElement.launch(population, *m_testSet);

        if (m_algorithmConfig.m_svmConfig.m_doVisualization)
        {
            setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, m_config, m_generationNumber);
           
			auto svm = m_population.getBestOne().getClassifier();
			svmComponents::SvmVisualization visualization;
			auto image = visualization.createDetailedVisualization(*svm,
				m_algorithmConfig.m_svmConfig.m_height,
				m_algorithmConfig.m_svmConfig.m_width,
                m_trainingSetInterface->trainingSet(), *m_validationSet, *m_testSet);

			m_savePngElement.launch(image, m_pngNameSource);
        }

        logResults(m_population, testPopulation);

        ++m_generationNumber;
    }
    catch (const std::exception& exception)
    {
		LOG_F(ERROR, "Error: %s", exception.what());
    }
}

unsigned int GaSvmWorkflow::getInitialTrainingSetSize()
{
    return m_algorithmConfig.m_numberOfClassExamples;
}

void GaSvmWorkflow::runGeneticAlgorithm()
{
    try
    {
        if (m_needRetrain)
        {
            retrainPopulation<SvmTrainingSetChromosome>(m_trainingSetInterface->trainingSet(),
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
            m_trainingSvmClassifierElement.launch(newPopulation, m_trainingSetInterface->trainingSet());
            m_validationElement->launch(newPopulation, *m_validationSet);
            auto nextGeneration = m_selectionElement.launch(m_population, newPopulation);

            auto nextGeneration2 = nextGeneration;
            m_validationTestDataElement.launch(nextGeneration2, *m_testSet); //tutaj ma by� kopia 

            m_population = m_selectionElement.launch(m_population, nextGeneration);

            if (m_algorithmConfig.m_svmConfig.m_doVisualization)
            {
                setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, m_config, m_generationNumber);
                
				auto svm = m_population.getBestOne().getClassifier();
				svmComponents::SvmVisualization visualization;
				auto image = visualization.createDetailedVisualization(*svm,
					m_algorithmConfig.m_svmConfig.m_height,
					m_algorithmConfig.m_svmConfig.m_width,
                    m_trainingSetInterface->trainingSet(), *m_validationSet, *m_testSet);

				m_savePngElement.launch(image, m_pngNameSource);
            }

            logResults(m_population, nextGeneration2);
            isStop = m_stopConditionElement.launch(nextGeneration);
            ++m_generationNumber;
        }



     /*   auto pop = getPopulation();
        std::vector<phd::svm::libSvmImplementation*> svms;
        for (auto& p : pop)
        {
            svms.emplace_back(reinterpret_cast<phd::svm::libSvmImplementation*>(p.getClassifier().get()));
        }

        phd::svm::EnsembleSvm ensemble(svms);

        auto& metric = *m_algorithmConfig.m_svmConfig.m_estimationMethod;
        BaseSvmChromosome ch;
        ch.updateClassifier(std::make_shared<phd::svm::EnsembleSvm>(ensemble));
        auto fitness = metric.calculateMetric(ch, *m_validationSet, false);

        if (fitness.m_fitness > m_population.getBestOne().getFitness())
        {
            std::cout << "Ensemble is better :) \n";
        }*/
    }
    catch (const std::exception& exception)
    {
		LOG_F(ERROR, "Error: %s", exception.what());
    }
}

void GaSvmWorkflow::initialize()
{
    if (m_trainingSet == nullptr)
    {
        m_trainingSet = &m_trainingSetInterface->trainingSet();
        m_validationSet = &m_loadingWorkflow.getValidationSet();
        m_testSet = &m_loadingWorkflow.getTestSet();
    }

    initializeGeneticAlgorithm();
}

SvmTrainingSetChromosome GaSvmWorkflow::getBestChromosomeInGeneration() const
{
    auto population = m_population;
    return population.getBestOne();
}

Population<SvmTrainingSetChromosome> GaSvmWorkflow::getPopulation() const
{
    return m_population;
}

void GaSvmWorkflow::setupKernelParameters(const SvmKernelChromosome& kernelParameters)
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

GeneticWorkflowResultLogger& GaSvmWorkflow::getResultLogger()
{
    return m_resultLogger;
}

dataset::Dataset<std::vector<float>, float> GaSvmWorkflow::getBestTrainingSet() const
{
    auto population = m_population;
    return population.getBestOne().convertChromosome(m_loadingWorkflow.getTraningSet());
}

void GaSvmWorkflow::setupFeaturesSet(const SvmFeatureSetChromosome& featureSetChromosome)
{
    m_trainingSet2 = featureSetChromosome.convertChromosome((m_trainingSetInterface->trainingSet()));
    m_trainingSet = &m_trainingSet2;

    m_validationSet2 = featureSetChromosome.convertChromosome((m_loadingWorkflow.getValidationSet()));
    m_validationSet = &m_validationSet2;

    m_testSet2 = featureSetChromosome.convertChromosome((m_loadingWorkflow.getTestSet()));
    m_testSet = &m_testSet2;

    // @wdudzik retrain because of different number of features in dataset
    m_needRetrain = true;
}
} // namespace genetic
