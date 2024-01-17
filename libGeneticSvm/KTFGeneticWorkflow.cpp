#include "KTFGeneticWorkflow.h"
#include "libPlatform/StringUtils.h"
#include <sstream>
#include <iterator>


namespace genetic
{
KTFGeneticWorkflow::KTFGeneticWorkflow(const SvmWokrflowConfiguration& config,
                                       KTFGeneticEvolutionConfiguration algorithmConfig)
    : m_trainingSetOptimization(std::move(algorithmConfig.m_trainingSetOptimization))
    , m_kernelOptimization(std::move(algorithmConfig.m_kernelOptimization))
    , m_featureSetOptimization(std::move(algorithmConfig.m_featureSetOptimization))
    , m_resultFilePath(std::filesystem::path(config.outputFolderPath.string() + config.txtLogFilename))
    , m_algorithmConfig(std::move(algorithmConfig))
{
}

//std::shared_ptr<phd::svm::ISvm> KTFGeneticWorkflow::run()
//{
//    m_trainingSetOptimization->initialize();
//
//    unsigned int featureNumber = static_cast<unsigned>(m_trainingSetOptimization->getBestTrainingSet().getSample(0).size());
//
//    m_featureSetOptimization->setupTrainingSet(m_trainingSetOptimization->getBestTrainingSet());
//    m_featureSetOptimization->initialize();
//
//    m_kernelOptimization->setDatasets(m_featureSetOptimization->getFilteredTraningSet(),
//                                      m_featureSetOptimization->getFilteredValidationSet(),
//                                      m_featureSetOptimization->getFilteredTestSet());
//
//    m_kernelOptimization->initialize();
//    log(*m_trainingSetOptimization);
//    log(*m_featureSetOptimization);
//    log(*m_kernelOptimization);
//
//    while (true)
//    {
//        m_kernelOptimization->runGeneticAlgorithm();
//        log(*m_kernelOptimization);
//
//        if (isFinished())
//        {
//            return (m_kernelOptimization->getBestChromosomeInGeneration().getClassifier());
//        }
//        m_trainingSetOptimization->setupKernelParameters(m_kernelOptimization->getBestChromosomeInGeneration());
//        m_trainingSetOptimization->setupFeaturesSet(convertToOldChromosome(m_featureSetOptimization->getBestChromosomeInGeneration(), featureNumber));
//
//        m_trainingSetOptimization->runGeneticAlgorithm();
//        log(*m_trainingSetOptimization);
//
//        if (isFinished())
//        {
//            return (m_trainingSetOptimization->getBestChromosomeInGeneration().getClassifier());
//        }
//        m_featureSetOptimization->setupTrainingSet(m_trainingSetOptimization->getBestTrainingSet());
//        m_featureSetOptimization->setupKernelParameters(m_kernelOptimization->getBestChromosomeInGeneration());
//
//        m_featureSetOptimization->runGeneticAlgorithm();
//        log(*m_featureSetOptimization);
//
//        if (isFinished())
//        {
//            return (m_featureSetOptimization->getBestChromosomeInGeneration().getClassifier());
//        }
//
//        m_kernelOptimization->setDatasets(m_featureSetOptimization->getFilteredTraningSet(),
//                                          m_featureSetOptimization->getFilteredValidationSet(),
//                                          m_featureSetOptimization->getFilteredTestSet());
//    }
//}




//5 times and take the best one - GECCO 2019
std::shared_ptr<phd::svm::ISvm> KTFGeneticWorkflow::run()
{
    std::vector<svmComponents::BaseSvmChromosome> classifier;
    std::vector<std::string> logentries;

    Timer timer;
	
    for (int i = 0; i < 5; i++)
    {
        m_trainingSetOptimization->initialize();

        unsigned int featureNumber = static_cast<unsigned>(m_trainingSetOptimization->getBestTrainingSet().getSample(0).size());

        m_featureSetOptimization->setupTrainingSet(m_trainingSetOptimization->getBestTrainingSet());
        m_featureSetOptimization->initialize();
        m_kernelOptimization->setDatasets(m_featureSetOptimization->getFilteredTraningSet(),
                                          m_featureSetOptimization->getFilteredValidationSet(),
                                          m_featureSetOptimization->getFilteredTestSet());

        m_kernelOptimization->initialize();
        log(*m_trainingSetOptimization);
        log(*m_featureSetOptimization);
        log(*m_kernelOptimization);

        while (true)
        {
            m_kernelOptimization->runGeneticAlgorithm();
            log(*m_kernelOptimization);

            if (isFinished())
            {
				m_kernelOptimization->getBestChromosomeInGeneration().getClassifier()->setFeatureSet(m_featureSetOptimization->getBestChromosomeInGeneration().getDataset(), featureNumber);
                classifier.push_back(m_kernelOptimization->getBestChromosomeInGeneration());
                logentries.push_back(*m_kernelOptimization->getResultLogger().getLogEntries().crbegin());
				
                break;
            }
            clearlog(*m_kernelOptimization);

            m_trainingSetOptimization->setupKernelParameters(m_kernelOptimization->getBestChromosomeInGeneration());
            m_trainingSetOptimization->setupFeaturesSet(convertToOldChromosome(m_featureSetOptimization->getBestChromosomeInGeneration(), featureNumber));

            m_trainingSetOptimization->runGeneticAlgorithm();
            log(*m_trainingSetOptimization);

            if (isFinished())
            {
				m_trainingSetOptimization->getBestChromosomeInGeneration().getClassifier()->setFeatureSet(m_featureSetOptimization->getBestChromosomeInGeneration().getDataset(), featureNumber);
                classifier.push_back(m_trainingSetOptimization->getBestChromosomeInGeneration());
                logentries.push_back(*m_trainingSetOptimization->getResultLogger().getLogEntries().crbegin());
               
                break;
            }
            clearlog(*m_trainingSetOptimization);

            m_featureSetOptimization->setupTrainingSet(m_trainingSetOptimization->getBestTrainingSet());
            m_featureSetOptimization->setupKernelParameters(m_kernelOptimization->getBestChromosomeInGeneration());

            m_featureSetOptimization->runGeneticAlgorithm();
            log(*m_featureSetOptimization);

            if (isFinished())
            {
				m_featureSetOptimization->getBestChromosomeInGeneration().getClassifier()->setFeatureSet(m_featureSetOptimization->getBestChromosomeInGeneration().getDataset(), featureNumber);
                classifier.push_back(m_featureSetOptimization->getBestChromosomeInGeneration());
                logentries.push_back(*m_featureSetOptimization->getResultLogger().getLogEntries().crbegin());
              
                break;
            }
            clearlog(*m_featureSetOptimization);

            m_kernelOptimization->setDatasets(m_featureSetOptimization->getFilteredTraningSet(),
                                              m_featureSetOptimization->getFilteredValidationSet(),
                                              m_featureSetOptimization->getFilteredTestSet());
        }
    }


    //std::ifstream timeOfInitPython(m_config.outputFolderPath.string() + "\\timeOfEnsembleFeatures.txt");
    //double time = 0.0;
    //timeOfInitPython >> time;
    //std::cout << time << "\n";
    //m_timer->decreaseTime(time * 1000); //converting to miliseconds
	
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

	finalEntry[3] = time;

	std::string s = std::accumulate(std::begin(finalEntry), std::end(finalEntry), std::string(),
		[](std::string &ss, std::string &s)
	{
		return ss.empty() ? s : ss + "\t" + s;
	});
	//s.append("\n");

	a.push_back(s);

    m_kernelOptimization->getResultLogger().setEntries(a);
    m_kernelOptimization->getResultLogger().logToFile(m_resultFilePath);

    return bestOne->getClassifier();
}

bool KTFGeneticWorkflow::isFinished() const
{
    return m_algorithmConfig.m_stopKernel->isFinished(m_kernelOptimization->getPopulation()) &&
            m_algorithmConfig.m_stopTrainingSet->isFinished(m_trainingSetOptimization->getPopulation()) &&
            m_algorithmConfig.m_stopFeatureSet->isFinished(m_featureSetOptimization->getPopulation());
}
} // namespace genetic
