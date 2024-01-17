

#include "EnsembleWorkflow.h"

namespace genetic
{
EnsembleWorkflow::EnsembleWorkflow(const SvmWokrflowConfiguration& config, GeneticAlternatingEvolutionConfiguration algorithmConfig, IDatasetLoader& workflow)
    : m_trainingSetOptimization(std::move(algorithmConfig.m_trainingSetOptimization))
    , m_kernelOptimization(std::move(algorithmConfig.m_kernelOptimization))
    , m_resultFilePath(std::filesystem::path(config.outputFolderPath.string() + config.txtLogFilename))
    , m_algorithmConfig(std::move(algorithmConfig))
	, m_loadingWorkflow(workflow)
	, m_outputPath(config.outputFolderPath)
	, m_config(config)
{
}


std::shared_ptr<phd::svm::ISvm> EnsembleWorkflow::run()
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
            auto tempPop = m_trainingSetOptimization->getPopulation();
            ensemble(tempPop);
            return m_kernelOptimization->getBestChromosomeInGeneration().getClassifier();
        }
        m_trainingSetOptimization->setupKernelParameters(m_kernelOptimization->getBestChromosomeInGeneration());
        m_trainingSetOptimization->runGeneticAlgorithm();
        log(*m_trainingSetOptimization);

        if (isFinished())
        {
            auto tempPop = m_trainingSetOptimization->getPopulation();
            ensemble(tempPop);
            return m_trainingSetOptimization->getBestChromosomeInGeneration().getClassifier();
        }
        m_kernelOptimization->setupTrainingSet(m_trainingSetOptimization->getBestTrainingSet());
    }
}

bool EnsembleWorkflow::isFinished() const
{
    return m_algorithmConfig.m_stopKernel->isFinished(m_kernelOptimization->getPopulation()) &&
        m_algorithmConfig.m_stopTrainingSet->isFinished(m_trainingSetOptimization->getPopulation());
}
} // namespace genetic
