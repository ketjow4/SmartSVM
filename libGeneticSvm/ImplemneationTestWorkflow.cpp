#include "ImplemneationTestWorkflow.h"

//#include "libStrategies/TabularDataProviderStrategy.h"
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
#include "WorkflowUtils.h"
#include "AllModelsLogger.h"
#include "libPlatform/loguru.hpp"

namespace genetic
{
    using namespace svmComponents;
    using namespace geneticComponents;

    ImplementationTestnWorkflow::ImplementationTestnWorkflow(const SvmWokrflowConfiguration& config,
        GeneticKernelEvolutionConfiguration algorithmConfig,
        IDatasetLoader& workflow)
        : m_algorithmConfig(std::move(algorithmConfig))
        , m_trainingSvmClassifierElement(*m_algorithmConfig.m_training)
        , m_valdiationElement(*m_algorithmConfig.m_svmConfig.m_estimationMethod, false)
        , m_valdiationTestDataElement(*m_algorithmConfig.m_svmConfig.m_estimationMethod, true)
       // , m_stopConditionElement(*m_algorithmConfig.m_stopCondition)
       // , m_crossoverElement(*m_algorithmConfig.m_crossover)
       // , m_mutationElement(*m_algorithmConfig.m_mutation)
       // , m_selectionElement(*m_algorithmConfig.m_selection)
        , m_createPopulationElement(*m_algorithmConfig.m_populationGeneration)
       // , m_savePngElement()
       // , m_createVisualizationElement(m_algorithmConfig.m_svmConfig)
       // , m_crossoverParentSelectionElement(*m_algorithmConfig.m_parentSelection)
        , m_trainingSet(nullptr)
        , m_validationSet(nullptr)
        , m_testSet(nullptr)
        , m_needRetrain(false)
        , m_loadingWorkflow(workflow)
        , m_generationNumber(0)
        , m_config(config)
        , m_allModelsLogger(nullptr)
        , m_timer(std::make_shared<Timer>())
    {
    }

    std::shared_ptr<phd::svm::ISvm> ImplementationTestnWorkflow::run()
    {
        static unsigned int numberOfRun = 1;
        auto outputPaht = m_config.outputFolderPath.string();
        auto logger = std::make_shared<AllModelsLogger>(numberOfRun++, outputPaht, m_loadingWorkflow);
        m_allModelsLogger = logger;

        initialize();

        m_resultLogger.logToFile(std::filesystem::path(m_config.outputFolderPath.string() + m_config.txtLogFilename));

        return m_population.getBestOne().getClassifier();
    }

    void ImplementationTestnWorkflow::logAllModels(geneticComponents::Population<svmComponents::SvmKernelChromosome>& population,
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


    void ImplementationTestnWorkflow::runSingleSvm()
    {
        try
        {
            auto population = m_createPopulationElement.launch(1);
            auto C = 10.0;
            auto gamma = 10.0;
            std::vector<double> kenrelParams = {C, gamma};
            population[0].updateKernelParameters(kenrelParams);

        	
            population = m_trainingSvmClassifierElement.launch(population, *m_trainingSet);
            m_population = m_valdiationElement.launch(population, *m_validationSet);
            auto testPopulation = m_valdiationTestDataElement.launch(population, *m_testSet);

            logAllModels(m_population, testPopulation);
            logResults(m_population, testPopulation);
        }
        catch (const std::runtime_error& exception)
        {
            LOG_F(ERROR, "Error: %s", exception.what());
            std::cout << exception.what();
        }
    }

    void ImplementationTestnWorkflow::logResults(const Population<SvmKernelChromosome>& population,
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

   

    void ImplementationTestnWorkflow::initialize()
    {
        static unsigned int numberOfRun = 1;
        auto outputPaht = m_config.outputFolderPath.string();
        if (m_allModelsLogger == nullptr)
        {
            auto  logger = std::make_shared<AllModelsLogger>(numberOfRun++, outputPaht, m_loadingWorkflow);
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

        runSingleSvm();
    }
} // namespace genetic
