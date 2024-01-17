
#include "AlgaWorkflow.h"
#include "libPlatform/StringUtils.h"
#include <sstream>
#include <iterator>


namespace genetic
{
AlgaWorkflow::AlgaWorkflow(const SvmWokrflowConfiguration& config,
                           GeneticAlternatingEvolutionConfiguration algorithmConfig)
    : m_trainingSetOptimization(std::move(algorithmConfig.m_trainingSetOptimization))
    , m_kernelOptimization(std::move(algorithmConfig.m_kernelOptimization))
    , m_resultFilePath(std::filesystem::path(config.outputFolderPath.string() + config.txtLogFilename))
    , m_algorithmConfig(std::move(algorithmConfig))
{
}

std::shared_ptr<phd::svm::ISvm> AlgaWorkflow::run()
{
    m_trainingSetOptimization->initialize();
    m_kernelOptimization->setupTrainingSet(m_trainingSetOptimization->getBestTrainingSet());
    m_kernelOptimization->initialize();
    log(*m_trainingSetOptimization);
    log(*m_kernelOptimization);

    while (true)
    {
        m_kernelOptimization->runGeneticAlgorithm();
        log(*m_kernelOptimization);

        if (isFinished())
        {
            return m_kernelOptimization->getBestChromosomeInGeneration().getClassifier();
        }
        m_trainingSetOptimization->setupKernelParameters(m_kernelOptimization->getBestChromosomeInGeneration());
        m_trainingSetOptimization->runGeneticAlgorithm();
        log(*m_trainingSetOptimization);

        if (isFinished())
        {
            return m_trainingSetOptimization->getBestChromosomeInGeneration().getClassifier();
        }
        m_kernelOptimization->setupTrainingSet(m_trainingSetOptimization->getBestTrainingSet());
    }
}



//5 times and take the best one
//std::shared_ptr<phd::svm::ISvm> AlgaWorkflow::run()
//{
//	std::vector<svmComponents::BaseSvmChromosome> classifier;
//	std::vector<std::string> logentries;
//    auto timer = std::make_shared<Timer>();
//
//    auto gen = 0;
//
//	for (int i = 0; i < 2; i++)
//	{
//        
//        m_trainingSetOptimization->setTimer(timer);
//        m_kernelOptimization->setTimer(timer);
//
//		m_trainingSetOptimization->initialize();
//		m_kernelOptimization->setupTrainingSet(m_trainingSetOptimization->getBestTrainingSet());
//		m_kernelOptimization->initialize();
//		log(*m_trainingSetOptimization);
//		log(*m_kernelOptimization);
//        clearlog(*m_kernelOptimization);
//        clearlog(*m_trainingSetOptimization);
//
//        gen++;
//
//		while (true)
//		{
//			m_kernelOptimization->runGeneticAlgorithm();
//			log(*m_kernelOptimization);
//
//			if (isFinished())
//			{
//				classifier.push_back(m_kernelOptimization->getBestChromosomeInGeneration());
//				logentries.push_back(*m_kernelOptimization->getResultLogger().getLogEntries().crbegin());
//
//                break;
//			}
//            clearlog(*m_kernelOptimization);
//            gen++;
//
//			m_trainingSetOptimization->setupKernelParameters(m_kernelOptimization->getBestChromosomeInGeneration());
//			m_trainingSetOptimization->runGeneticAlgorithm();
//			log(*m_trainingSetOptimization);
//
//			if (isFinished())
//			{
//				classifier.push_back(m_trainingSetOptimization->getBestChromosomeInGeneration());
//				logentries.push_back(*m_trainingSetOptimization->getResultLogger().getLogEntries().crbegin());
//
//                break;
//			}
//            clearlog(*m_trainingSetOptimization);
//
//			m_kernelOptimization->setupTrainingSet(m_trainingSetOptimization->getBestTrainingSet());
//
//    
//
//            gen++;
//		}
//	}
//
//    
//
//	auto bestOne = std::max_element(classifier.begin(), classifier.end(),
//		[](svmComponents::BaseSvmChromosome left, svmComponents::BaseSvmChromosome right)
//	{
//		return left.getFitness() < right.getFitness();
//	});
//	auto it = std::find_if(classifier.begin(), classifier.end(),
//		[&bestOne](svmComponents::BaseSvmChromosome element)
//	{
//		return element.getFitness() == bestOne->getFitness();
//	});
//	auto pos = std::distance(classifier.begin(), it);
//	std::vector<std::string> a;
//
//
//	auto finalEntry = ::platform::stringUtils::splitString(logentries[pos], '\t');
//
//	auto time = ::platform::stringUtils::splitString(*logentries.rbegin(), '\t')[3];
//
//	finalEntry[3] = time;
//
//	std::string s = std::accumulate(std::begin(finalEntry), std::end(finalEntry), std::string(),
//		[](std::string &ss, std::string &s)
//	{
//		return ss.empty() ? s : ss + "\t" + s;
//	});
//	//s.append("\n");
//
//	a.push_back(s);
//
//	m_kernelOptimization->getResultLogger().setEntries(a);
//	m_kernelOptimization->getResultLogger().logToFile(m_resultFilePath);
//  
//	return bestOne->getClassifier();
//}


bool AlgaWorkflow::isFinished() const
{
    return m_algorithmConfig.m_stopKernel->isFinished(m_kernelOptimization->getPopulation()) &&
            m_algorithmConfig.m_stopTrainingSet->isFinished(m_trainingSetOptimization->getPopulation());
}
} // namespace genetic
