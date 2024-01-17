#include "libSvmComponents/SvmValidationStrategy.h"
#include "libGeneticStrategies/StopConditionStrategy.h"
#include "libGeneticStrategies/CrossoverStrategy.h"
#include "libGeneticStrategies/MutationStrategy.h"
#include "libGeneticStrategies/SelectionStrategy.h"
#include "libGeneticStrategies/CreatePopulationStrategy.h"
#include "libStrategies/FileSinkStrategy.h"
#include "libSvmStrategies/CreateSvmVisualizationStrategy.h"
#include "libSvmStrategies/SvmTrainingStrategy.h"
#include "libGeneticStrategies/CombinePopulationsStrategy.h"
#include "LibGeneticComponents/LocalGlobalAdaptationSelection.h"
#include "libSvmComponents/SvmTraining.h"
#include "libGeneticStrategies/CrossoverParentSelectionStrategy.h"
#include "libSvmComponents/ConfusionMatrixMetrics.h"
#include "SvmExceptions.h"
#include "MemeticFeaturesSelection.h"
#include "WorkflowUtils.h"
#include "CombinedAlgorithmsConfig.h"
#include "libPlatform/loguru.hpp"

namespace genetic
{
using namespace svmComponents;
using namespace geneticComponents;

MemeticFeaturesSelection::MemeticFeaturesSelection(const SvmWokrflowConfiguration& config,
                                                   MemeticFeatureSetEvolutionConfiguration algorithmConfig,
                                                   IDatasetLoader& workflow)
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
    , m_educationElement(*m_algorithmConfig.m_education)
    , m_crossoverCompensationElement(*m_algorithmConfig.m_compensation)
    , m_adaptationElement(*m_algorithmConfig.m_adaptation)
    , m_featuresPoolElement(*m_algorithmConfig.m_supporVectorPool)
    , m_superIndividualsGenerationElement(*m_algorithmConfig.m_superIndivudualsGeneration)
    , m_populationCombinationElement()
    , m_trainingSuperIndividualsElement(*m_algorithmConfig.m_training)
    , m_validationSuperIndividualsElement(*m_algorithmConfig.m_svmConfig.m_estimationMethod, false)
    , m_parentSelectionElement(*m_algorithmConfig.m_parentSelection)
    , m_compensationGenerationElement(*m_algorithmConfig.m_compensationGeneration)
    , m_numberOfClassExamples(m_algorithmConfig.m_initialNumberOfClassExamples)
    , m_featurePool(std::vector<Feature>())
    , m_trainingSet(nullptr)
    , m_validationSet(nullptr)
    , m_testSet(nullptr)
    , m_needRetrain(false)
    , m_loadingWorkflow(workflow)
    , m_generationNumber(0)
    , m_config(config)
	, m_timer(std::make_shared<Timer>())
{
}

std::shared_ptr<phd::svm::ISvm> MemeticFeaturesSelection::run()
{
    initialize();
    runGeneticAlgorithm();
    m_resultLogger.logToFile(std::filesystem::path(m_config.outputFolderPath.string() + m_config.txtLogFilename));

    return (getBestChromosomeInGeneration().getClassifier());
}

void MemeticFeaturesSelection::logResults(const Population<SvmFeatureSetMemeticChromosome>& population,
                                          const Population<SvmFeatureSetMemeticChromosome>& testpopulation)
{
    auto bestOne = population.getBestOne();
    auto bestOneConfustionMatrix = bestOne.getConfusionMatrix().value();

    auto genes = convertToOldChromosome(bestOne, static_cast<unsigned>(m_featureNumberAll)).getGenes();
    auto featuresNumber = std::accumulate(std::begin(genes), std::end(genes), 0);
    auto trainingSetSize = m_trainingSet->size();
    std::stringstream genesAsString;
    std::copy(genes.begin(), genes.end(), std::ostream_iterator<int>(genesAsString, ""));
    auto bestOneIndex = m_population.getBestIndividualIndex();

    m_resultLogger.createLogEntry(population,
                                  testpopulation,
                                  *m_timer,
                                  m_algorithmName,
                                  m_generationNumber,
                                  Accuracy(bestOneConfustionMatrix),
                                  featuresNumber,
                                  trainingSetSize,
                                  bestOneConfustionMatrix,
                                  testpopulation[bestOneIndex].getConfusionMatrix().value(),
                                  genesAsString.str());
    ++m_generationNumber;
}

Population<SvmFeatureSetMemeticChromosome> MemeticFeaturesSelection::createPopulationWithTimeMeasurement(int popSize)
{
    //std::cout << "Adding time of feature selection\n";
    m_timer->pause();
    auto population = m_createPopulationElement.launch(popSize);
    m_timer->contine();
    std::ifstream timeOfInitPython(m_config.outputFolderPath.string() + "\\timeOfEnsembleFeatures.txt");
    double time = 0.0;
    timeOfInitPython >> time;
    //std::cout << time << "\n";
    m_timer->addTime(time * 1000); //converting to miliseconds
    return population;
}

void createHistogram(const Population<SvmFeatureSetMemeticChromosome>& /*population*/)
{
    //std::vector<int> histogram(500, 0);

    //for(auto& p : population)
    //{
    //    auto features = p.getDataset();
    //    for(auto& f : features)
    //    {
    //        histogram[f.id]++;
    //    }
    //}
    //static int generation = 0;
    //std::fstream f("D:\\datasetsFolds\\hist\\histogram_gen" + std::to_string(generation) + ".txt", std::ios::out);

    //if(f.is_open())
    //{
    //    for(auto i = 0u; i < histogram.size(); ++i)
    //    {
    //        f << i << "," << histogram[i] << "\n";
    //    }

    //    f.close();
    //}
    //generation++;
}


inline std::vector<SvmFeatureSetMemeticChromosome> generate(unsigned populationSize)
{
    std::vector<SvmFeatureSetMemeticChromosome> population(populationSize);
    
    return population;
}

void MemeticFeaturesSelection::runGeneticAlgorithm()
{
    try
    {
        if (m_needRetrain)
        {
            retrainPopulation<SvmFeatureSetMemeticChromosome>(*m_trainingSet,
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

            auto populationEducated = m_educationElement.launch(result, m_featurePool, parents, *m_trainingSet);

            populationEducated = m_mutationElement.launch(populationEducated);
            auto poptrained = m_trainingSvmClassifierElement.launch(populationEducated, *m_trainingSet);
            auto afterValidtion = m_valdiationElement.launch(populationEducated, *m_validationSet);

            m_featurePool = m_featuresPoolElement.launch(poptrained, *m_trainingSet);


            auto superIndividualsSize = static_cast<unsigned int>(m_algorithmConfig.m_populationSize * m_algorithmConfig.m_superIndividualAlpha);
            Population<SvmFeatureSetMemeticChromosome> superIndividualsPopulation{ generate(superIndividualsSize) };

            if (!m_featurePool.empty())
            {
                superIndividualsPopulation = m_superIndividualsGenerationElement.launch(superIndividualsSize, m_featurePool, m_numberOfClassExamples);
                m_trainingSuperIndividualsElement.launch(superIndividualsPopulation, *m_trainingSet);
                m_validationSuperIndividualsElement.launch(superIndividualsPopulation, *m_validationSet);
            }
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
                auto image = m_createVisualizationElement.launch(m_population, *m_trainingSet, *m_validationSet);
                m_savePngElement.launch(image, m_pngNameSource);
            }

            auto localGlobal = dynamic_cast<LocalGlobalAdaptationSelection<SvmFeatureSetMemeticChromosome>*>(m_algorithmConfig.m_parentSelection.get());
            if (localGlobal != nullptr)
            {
                localGlobal->setMode(IsModeLocal);
            }

            logResults(m_population, testPopulation);

            createHistogram(m_population);
        }
    }
    catch (const std::exception& exception)
    {
		LOG_F(ERROR, "Error: %s", exception.what());
    }
}

void MemeticFeaturesSelection::initialize()
{
    if (m_trainingSet == nullptr)
    {
        m_trainingSet = &m_loadingWorkflow.getTraningSet();       
    }

    m_validationSet = &m_loadingWorkflow.getValidationSet();  //why only traning set
    m_testSet = &m_loadingWorkflow.getTestSet();
    m_featureNumberAll = m_validationSet->getSample(0).size();

    initializeGeneticAlgorithm();
}

void MemeticFeaturesSelection::setupKernelParameters(const SvmKernelChromosome& kernelParameters)
{
    auto trainingElement = dynamic_cast<SvmTraining<SvmFeatureSetMemeticChromosome>*>(m_algorithmConfig.m_training.get());
    if (trainingElement != nullptr)
    {
        trainingElement->updateParameters(kernelParameters);
        m_needRetrain = true;
        return;
    }
    throw BadTrainingElement();
}

GeneticWorkflowResultLogger& MemeticFeaturesSelection::getResultLogger()
{
    return m_resultLogger;
}

SvmFeatureSetMemeticChromosome MemeticFeaturesSelection::getBestChromosomeInGeneration() const
{
    auto population = m_population;
    return population.getBestOne();
}

Population<SvmFeatureSetMemeticChromosome> MemeticFeaturesSelection::getPopulation() const
{
    return m_population;
}

dataset::Dataset<std::vector<float>, float> MemeticFeaturesSelection::getFilteredTraningSet()
{
    return getBestChromosomeInGeneration().convertChromosome(*m_trainingSet);
}

dataset::Dataset<std::vector<float>, float> MemeticFeaturesSelection::getFilteredValidationSet()
{
    return getBestChromosomeInGeneration().convertChromosome(*m_validationSet);
}

dataset::Dataset<std::vector<float>, float> MemeticFeaturesSelection::getFilteredTestSet()
{
    return getBestChromosomeInGeneration().convertChromosome(*m_testSet);
}

void MemeticFeaturesSelection::setupTrainingSet(const dataset::Dataset<std::vector<float>, float>& trainingSet)
{
    m_trainingSet22222 = trainingSet;
    m_trainingSet = &m_trainingSet22222;
    m_needRetrain = true;
}

geneticComponents::Population<svmComponents::SvmFeatureSetMemeticChromosome> MemeticFeaturesSelection::initNoEvaluate(int popSize)
{
    try
    {
        if (m_trainingSet == nullptr)
        {
            m_trainingSet = &m_loadingWorkflow.getTraningSet();
        }

        m_validationSet = &m_loadingWorkflow.getValidationSet();  //why only traning set
        m_testSet = &m_loadingWorkflow.getTestSet();
        m_featureNumberAll = m_validationSet->getSample(0).size();

        auto population = createPopulationWithTimeMeasurement(popSize);
        m_population = population;

        return population;
    }
    catch (const std::exception& exception)
    {
		LOG_F(ERROR, "Error: %s", exception.what());
        std::cout << std::string("Unknown exception: ") + exception.what();
        throw;
    }
}

geneticComponents::Population<svmComponents::SvmFeatureSetMemeticChromosome> MemeticFeaturesSelection::initNoEvaluate(int /*popSize*/, int /*seed*/)
{
    throw std::exception("Not implemented initNoEvaluate(int popSize, int seed) in MemeticFeaturesSelection");
}

//used in SE-SVM 
void MemeticFeaturesSelection::performGeneticOperations(geneticComponents::Population<svmComponents::SvmFeatureSetMemeticChromosome>& population)
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

    auto populationEducated = m_educationElement.launch(result, m_featurePool, parents, *m_trainingSet);

    populationEducated = m_mutationElement.launch(populationEducated);
    
    m_featurePool = m_featuresPoolElement.launch(population, *m_trainingSet);

    auto superIndividualsSize = static_cast<unsigned int>(m_algorithmConfig.m_populationSize * m_algorithmConfig.m_superIndividualAlpha);
    auto superIndividualsPopulation = m_superIndividualsGenerationElement.launch(superIndividualsSize, m_featurePool, m_numberOfClassExamples);

    auto combinedPopulation = m_populationCombinationElement.launch(populationEducated, superIndividualsPopulation);

    auto[IsModeLocal, NumberOfClassExamples] = m_adaptationElement.launch(m_population);
    m_numberOfClassExamples = NumberOfClassExamples;

    population = combinedPopulation;

    auto localGlobal = dynamic_cast<LocalGlobalAdaptationSelection<SvmFeatureSetMemeticChromosome>*>(m_algorithmConfig.m_parentSelection.get());
    if (localGlobal != nullptr)
    {
        localGlobal->setMode(IsModeLocal);
    }
}

std::shared_ptr<Timer> MemeticFeaturesSelection::getTimer()
{
    return m_timer;
}

void MemeticFeaturesSelection::setTimer(std::shared_ptr<Timer> timer)
{
    m_timer = timer;
}

void MemeticFeaturesSelection::initializeGeneticAlgorithm()
{
    try
    {
        auto population = createPopulationWithTimeMeasurement(m_algorithmConfig.m_populationSize);
    	
        m_trainingSvmClassifierElement.launch(population, *m_trainingSet);
        m_population = m_valdiationElement.launch(population, *m_validationSet);
        auto testPopulation = m_valdiationTestDataElement.launch(population, *m_testSet);

        if (m_algorithmConfig.m_svmConfig.m_doVisualization)
        {
            setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, m_config, m_generationNumber);
            auto image = m_createVisualizationElement.launch(population, *m_trainingSet, *m_validationSet);
            m_savePngElement.launch(image, m_pngNameSource);
        }

        logResults(m_population, testPopulation);
    }
    catch (const std::exception& exception)
    {
		LOG_F(ERROR, "Error: %s", exception.what());
		std::cout << std::string("Unknown exception: ") + exception.what();
    }
}
} // namespace genetic
