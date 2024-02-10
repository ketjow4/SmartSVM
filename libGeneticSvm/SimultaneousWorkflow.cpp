#include "SimultaneousWorkflow.h"
#include "libSvmComponents/SvmSimultaneousChromosome.h"
#include "SvmTrainingStrategy.h"
#include "libSvmComponents/ConfusionMatrixMetrics.h"
#include "AllModelsLogger.h"
#include "libPlatform/StringUtils.h"

namespace genetic
{
//USED ONLY FOR 2D dataset visualization -- in order to use this code need to be modified 08.02.2023
class MockedFeaturesSelection : public ISvmAlgorithm, public IFeatureSelectionWorkflow<svmComponents::SvmFeatureSetMemeticChromosome>
{
public:
	std::shared_ptr<phd::svm::ISvm> run() override
	{
        throw std::exception("Mocked feature selection. See simultaneousWorkflow");
	}

	void initialize() override
	{
	}

	void runGeneticAlgorithm() override
	{
	}

	SvmFeatureSetMemeticChromosome getBestChromosomeInGeneration() const override
	{
        std::vector<Feature> features{ 0,1 };
        SvmFeatureSetMemeticChromosome chromosome(std::move(features));
        return chromosome;
	}

	geneticComponents::Population<SvmFeatureSetMemeticChromosome> getPopulation() const override
	{
        return m_pop;
	}

	GeneticWorkflowResultLogger& getResultLogger() override
	{
        return m_logger;
	}

	geneticComponents::Population<SvmFeatureSetMemeticChromosome> initNoEvaluate(int popSize) override
	{
        std::vector<Feature> features{ 0,1 };
        SvmFeatureSetMemeticChromosome chromosome(std::move(features));
        std::vector<SvmFeatureSetMemeticChromosome> pop(popSize, chromosome);
       
        m_pop = Population(std::move(pop));
        return m_pop;
	}

	geneticComponents::Population<SvmFeatureSetMemeticChromosome> initNoEvaluate(int popSize, int /*seed*/) override
	{
        std::vector<Feature> features{ 0,1 };
        SvmFeatureSetMemeticChromosome chromosome(std::move(features));
        std::vector<SvmFeatureSetMemeticChromosome> pop(popSize, chromosome);

        m_pop = Population(std::move(pop));
        return m_pop;
	}

	void performGeneticOperations(geneticComponents::Population<SvmFeatureSetMemeticChromosome>& population) override
	{
        std::vector<Feature> features{ 0,1 };
        SvmFeatureSetMemeticChromosome chromosome(std::move(features));
        //std::vector<SvmFeatureSetMemeticChromosome> pop(popSize, chromosome);

        if(population.size() != 12)
        {
	        auto pop = population.get();
	        pop.push_back(chromosome);
	        pop.push_back(chromosome);
			m_pop = Population(std::move(pop));
	        population = m_pop;
        }
        
	}

	void setupKernelParameters(const svmComponents::SvmKernelChromosome& /*kernelParameters*/) override
	{
	}

	dataset::Dataset<std::vector<float>, float> getFilteredTraningSet() override
	{
        throw std::exception("Method not supported in mocked feature selection");
	}

	dataset::Dataset<std::vector<float>, float> getFilteredValidationSet() override
	{
        throw std::exception("Method not supported in mocked feature selection");
	}

	dataset::Dataset<std::vector<float>, float> getFilteredTestSet() override
	{
        throw std::exception("Method not supported in mocked feature selection");
	}

	void setupTrainingSet(const dataset::Dataset<std::vector<float>, float>& /*trainingSet*/) override
	{
	}
private:
    Population<SvmFeatureSetMemeticChromosome> m_pop;
    GeneticWorkflowResultLogger m_logger;
};


SimultaneousWorkflow::SimultaneousWorkflow(const SvmWokrflowConfiguration& config, 
                                           SimultaneousWorkflowConfig algorithmConfig,
                                           IDatasetLoader& workflow)
    : m_trainingSetOptimization(std::move(algorithmConfig.m_trainingSetOptimization))
    , m_kernelOptimization(std::move(algorithmConfig.m_kernelOptimization))
    , m_featureSetOptimization(std::move(algorithmConfig.m_featureSetOptimization))
    , m_resultFilePath(std::filesystem::path(config.outputFolderPath.string() + config.txtLogFilename))
    , m_algorithmConfig(std::move(algorithmConfig))
    , m_svmTraining(*m_algorithmConfig.m_svmTraining)
    , m_validation(*algorithmConfig.m_svmConfig.m_estimationMethod, false)
    , m_validationTest(*algorithmConfig.m_svmConfig.m_estimationMethod, true)
    , m_stopConditionElement(*algorithmConfig.m_stopCondition)
    , m_trainingSet(workflow.getTraningSet())
    , m_validationSet(workflow.getValidationSet())
    , m_testSet(workflow.getTestSet())
    , m_selectionElement(*m_algorithmConfig.m_selectionElement)
    , m_generationNumber(0)
	, m_workflow(workflow)
	, m_config(config)
	, m_createVisualizationElement(m_algorithmConfig.m_svmConfig)
	, m_timer(std::make_shared<Timer>())
{
    //m_featureSetOptimization = std::make_shared<MockedFeaturesSelection>();
}

void SimultaneousWorkflow::logAllModels(AllModelsLogger& logger)
{
    auto bestOneConfustionMatrix = m_pop.getBestOne().getConfusionMatrix().value();
    auto bestOneIndex = m_pop.getBestIndividualIndex();
    auto testbestOneConfustionMatrix = m_popTestSet[bestOneIndex].getConfusionMatrix().value();
    logger.log(m_pop,
               m_popTestSet,
               *m_timer,
               m_algorithmName,
               m_generationNumber++,
               Accuracy(bestOneConfustionMatrix),
               m_pop.getBestOne().featureSetSize(),
               bestOneConfustionMatrix,
			   testbestOneConfustionMatrix);
}


void SimultaneousWorkflow::createVisualization()
{
	std::filesystem::path m_pngNameSource;
	strategies::FileSinkStrategy m_savePngElement;
	setVisualizationFilenameAndFormat(m_algorithmConfig.m_svmConfig.m_visualizationFormat, m_pngNameSource, m_config, m_generationNumber);
	auto image = m_createVisualizationElement.launch(m_pop, m_trainingSet, m_testSet);
	m_savePngElement.launch(image, m_pngNameSource);
}

std::shared_ptr<phd::svm::ISvm> SimultaneousWorkflow::run()
{
	static unsigned int numberOfRun = 1;
	auto outputPaht = m_config.outputFolderPath.string();
   
	AllModelsLogger logger{ numberOfRun, outputPaht, m_workflow };

    std::vector<svmComponents::BaseSvmChromosome> classifier;
    std::vector<std::string> logentries;

    for (int i = 0; i < 5; i++)
    {

        init();
        evaluate();

        if (m_algorithmConfig.m_svmConfig.m_doVisualization)
        {
	        createVisualization();
        }

        if (m_config.verbosity == platform::Verbosity::All)
        {
            logAllModels(logger);
        }
        while (!isFinished())
        {
            performEvolution();
            evaluate();

            if (m_algorithmConfig.m_svmConfig.m_doVisualization)
            {
                createVisualization();
            }

            if (m_config.verbosity == platform::Verbosity::All)
            {
                logAllModels(logger);
            }
        }

        classifier.push_back(m_pop.getBestOne()); //return best in here
        logentries.push_back(*m_resultLogger.getLogEntries().crbegin());

        if (m_config.verbosity != platform::Verbosity::None)
        {
            m_resultLogger.logToFile(m_resultFilePath);
        }
        m_resultLogger.clearLog();
    }

    if (numberOfRun == 1)
    {
    	//adding time twice so we need to decrease it in here on the first run
        std::ifstream timeOfInitPython(m_config.outputFolderPath.string() + "\\timeOfEnsembleFeatures.txt");
        double time = 0.0;
        timeOfInitPython >> time;
        //std::cout << time << "\n";
        m_timer->decreaseTime(time * 1000); //converting to miliseconds
    }
    
	if(numberOfRun != 1)
    {
        //std::cout << "SESVM Adding time of feature selection\n";
	    std::ifstream timeOfInitPython(m_config.outputFolderPath.string() + "\\timeOfEnsembleFeatures.txt");
	    double time = 0.0;
	    timeOfInitPython >> time;
	    //std::cout << time << "\n";
	    m_timer->addTime(time * 1000); //converting to miliseconds
    }

    auto bestOne = std::max_element(classifier.begin(), classifier.end(),
                                    [](svmComponents::BaseSvmChromosome left, svmComponents::BaseSvmChromosome right)
                                    {
                                        return left.getFitness() < right.getFitness();
                                    });
    auto it = std::find_if(classifier.begin(), classifier.end(),
                           [&bestOne](svmComponents::BaseSvmChromosome element)
                           {
                               return element.getFitness() == bestOne->getFitness();
                           });
    auto pos = std::distance(classifier.begin(), it);
    std::vector<std::string> a;


    auto finalEntry = ::platform::stringUtils::splitString(logentries[pos], '\t');

    auto time = ::platform::stringUtils::splitString(*logentries.rbegin(), '\t')[3];

    //finalEntry[3] = time;
    finalEntry[3] = std::to_string(m_timer->getTimeMiliseconds().count());

    std::string s = std::accumulate(std::begin(finalEntry), std::end(finalEntry), std::string(),
                                    [](std::string &ss, std::string &s)
                                    {
                                        return ss.empty() ? s : ss + "\t" + s;
                                    });
    //s.append("\n");

    a.push_back(s);
    
    if (m_config.verbosity == platform::Verbosity::All)
	{
        logger.save(outputPaht + "\\" + std::to_string(numberOfRun) + "\\populationTextLog.txt");
    }
    //m_resultLogger.logToFile(m_resultFilePath);

    m_resultLogger.setEntries(a);
    if (m_config.verbosity != platform::Verbosity::None)
    {
        m_resultLogger.logToFile(m_resultFilePath);
    }

    numberOfRun++;

    return bestOne->getClassifier(); //m_pop.getBestOne().getClassifier(); //return best in here
}

void SimultaneousWorkflow::init()
{   
    auto popSize = m_algorithmConfig.m_populationSize;
    auto features = m_featureSetOptimization->initNoEvaluate(popSize);
    auto kernels = m_kernelOptimization->initNoEvaluate(popSize);
    auto traningSets = m_trainingSetOptimization->initNoEvaluate(popSize);

    std::vector<svmComponents::SvmSimultaneousChromosome> vec;
    vec.reserve(popSize);
    for(auto i = 0u; i < features.size(); ++i)
    {
        svmComponents::SvmSimultaneousChromosome c{ kernels[i], traningSets[i], features[i] };
        vec.emplace_back(c);
    }
    geneticComponents::Population<svmComponents::SvmSimultaneousChromosome> pop{ vec };
    m_pop = pop;
}

geneticComponents::Population<svmComponents::SvmKernelChromosome> SimultaneousWorkflow::getKernelPopulation()
{
    std::vector<svmComponents::SvmKernelChromosome> vec;
    vec.reserve(m_pop.size());
    for (auto i = 0u; i < m_pop.size(); ++i)
    {
        svmComponents::SvmKernelChromosome c{ m_pop[i].getKernelType(), m_pop[i].getKernelParameters(), false };
        vec.emplace_back(c);
    }
    geneticComponents::Population<svmComponents::SvmKernelChromosome> pop{ vec };
    return pop;
}

geneticComponents::Population<svmComponents::SvmFeatureSetMemeticChromosome> SimultaneousWorkflow::getFeaturesPopulation()
{
    std::vector<svmComponents::SvmFeatureSetMemeticChromosome> vec;
    vec.reserve(m_pop.size());
    for (auto i = 0u; i < m_pop.size(); ++i)
    {
        svmComponents::SvmFeatureSetMemeticChromosome c{ m_pop[i].getFeaturesChromosome() };
        vec.emplace_back(c);
    }
    geneticComponents::Population<svmComponents::SvmFeatureSetMemeticChromosome> pop{ vec };
    return pop;
}

geneticComponents::Population<svmComponents::SvmTrainingSetChromosome> SimultaneousWorkflow::getTrainingSetPopulation()
{
    std::vector<svmComponents::SvmTrainingSetChromosome> vec;
    vec.reserve(m_pop.size());
    for (auto i = 0u; i < m_pop.size(); ++i)
    {
        svmComponents::SvmTrainingSetChromosome c{ m_pop[i].getTraining() };
        vec.emplace_back(c);
    }
    geneticComponents::Population<svmComponents::SvmTrainingSetChromosome> pop{ vec };
    return pop;
}

void SimultaneousWorkflow::setKernelPopulation(geneticComponents::Population<svmComponents::SvmKernelChromosome>& population, 
                                               geneticComponents::Population<svmComponents::SvmSimultaneousChromosome>& newPop)
{
    int i = 0;
    for(const auto& individual : population)
    {
        newPop[i].setKernel(individual);
        ++i;
    }
}

 void SimultaneousWorkflow::setTrainingPopulation(geneticComponents::Population<svmComponents::SvmTrainingSetChromosome>& population,
                                                  geneticComponents::Population<svmComponents::SvmSimultaneousChromosome>& newPop)
{
     int i = 0;
     for (const auto& individual : population)
     {
         newPop[i].setTraining(individual);
         ++i;
     }
}

void SimultaneousWorkflow::setFeaturesPopulation(geneticComponents::Population<svmComponents::SvmFeatureSetMemeticChromosome>& population,
                                                 geneticComponents::Population<svmComponents::SvmSimultaneousChromosome>& newPop)
{
    int i = 0;
    for (const auto& individual : population)
    {
        newPop[i].setFeatures(individual);
        ++i;
    }
}

void fixKernelPop(const geneticComponents::Population<svmComponents::SvmSimultaneousChromosome>& currentPop,
                  geneticComponents::Population<svmComponents::SvmSimultaneousChromosome>& newPop)
{
    auto bestKernel = currentPop.getBestOne().getKernel();

    for (auto& individual : newPop)
    {
        if(individual.getKernelType() == phd::svm::KernelTypes::Custom)
        {
            individual.setKernel(bestKernel);
        }
    }
}

void SimultaneousWorkflow::performEvolution()
{
    auto kernels = getKernelPopulation();
    auto features = getFeaturesPopulation();
    auto trainingSets = getTrainingSetPopulation();

    m_featureSetOptimization->performGeneticOperations(features);
    m_kernelOptimization->performGeneticOperations(kernels);
    m_trainingSetOptimization->performGeneticOperations(trainingSets);

    std::vector<svmComponents::SvmSimultaneousChromosome> vec;
    vec.reserve(m_algorithmConfig.m_populationSize);
    for (auto i = 0u; i < trainingSets.size(); ++i)
    {
        svmComponents::SvmSimultaneousChromosome c{};
        vec.emplace_back(c);
    }
    geneticComponents::Population<svmComponents::SvmSimultaneousChromosome> newPopulation{ vec };

    setFeaturesPopulation(features, newPopulation);
    setKernelPopulation(kernels, newPopulation);
    setTrainingPopulation(trainingSets, newPopulation);

    fixKernelPop(m_pop, newPopulation);

    m_svmTraining.launch(newPopulation, m_trainingSet);

    m_popTestSet = newPopulation;   //copy in here!!!
    m_validation.launch(newPopulation, m_validationSet);

    auto m_pop2 = m_selectionElement.launch(m_pop, newPopulation);
    //m_algorithmConfig.m_selectionElement->selectNextGeneration()
    //select 
    m_pop = m_pop2;
}

void SimultaneousWorkflow::evaluate()
{
    m_svmTraining.launch(m_pop, m_trainingSet);

    m_popTestSet = m_pop;   //copy in here!!!
    m_validation.launch(m_pop, m_validationSet);
    m_validationTest.launch(m_popTestSet, m_testSet);

    log();
}

bool SimultaneousWorkflow::isFinished()
{
    return m_stopConditionElement.launch(m_pop);
}

void SimultaneousWorkflow::log()
{
    //TODO add header to logs and modify python analysis to use this header
    auto bestOneConfustionMatrix = m_pop.getBestOne().getConfusionMatrix().value();
    auto bestOneIndex = m_pop.getBestIndividualIndex();
    auto testbestOneConfustionMatrix = m_popTestSet[bestOneIndex].getConfusionMatrix().value();
    
    
    //auto featureNumber = m_validationSet.getSamples()[0].size();

    m_resultLogger.createLogEntry(m_pop,
                                  m_popTestSet,
                                  *m_timer,
                                  m_algorithmName,
                                  m_generationNumber,
                                  Accuracy(bestOneConfustionMatrix),
                                  m_pop.getBestOne().featureSetSize(),
					              m_pop.getBestOne().trainingSetSize() * 2, //TODO fix for multiclass
                                  bestOneConfustionMatrix,
								  testbestOneConfustionMatrix);
}
} // namespace genetic
