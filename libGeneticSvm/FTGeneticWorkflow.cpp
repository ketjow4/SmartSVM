
#pragma once

#include "libGeneticSvm/FTGeneticWorkflow.h"

namespace genetic
{
FTGeneticWorkflow::FTGeneticWorkflow(const SvmWokrflowConfiguration& config, TFGeneticEvolutionConfiguration algorithmConfig)
    : m_trainingSetOptimization(std::move(algorithmConfig.m_trainingSetOptimization))
    , m_featureSetOptimization(std::move(algorithmConfig.m_featureSetOptimization))
    , m_resultFilePath(std::filesystem::path(config.outputFolderPath.string() + config.txtLogFilename))
    , m_algorithmConfig(std::move(algorithmConfig))
{
}

std::shared_ptr<phd::svm::ISvm> FTGeneticWorkflow::run()
{
    /*m_featureSetOptimization->initialize();

    m_trainingSetOptimization->setupFeaturesSet(m_featureSetOptimization->getBestChromosomeInGeneration());
    m_trainingSetOptimization->initialize();

    log(*m_featureSetOptimization);
    log(*m_trainingSetOptimization);

    while (true)
    {
        m_featureSetOptimization->setupTrainingSet(m_trainingSetOptimization->getBestTrainingSet());

        m_featureSetOptimization->runGeneticAlgorithm();
        log(*m_featureSetOptimization);

        if (isFinished())
        {
            return m_trainingSetOptimization->getBestChromosomeInGeneration().getClassifier();
        }

        m_trainingSetOptimization->setupFeaturesSet(m_featureSetOptimization->getBestChromosomeInGeneration());
        m_trainingSetOptimization->runGeneticAlgorithm();
        log(*m_trainingSetOptimization);

        if (isFinished())
        {
            return m_trainingSetOptimization->getBestChromosomeInGeneration().getClassifier();
        }
    }*/
    return nullptr;
}

bool FTGeneticWorkflow::isFinished() const
{
    return m_algorithmConfig.m_stopTrainingSet->isFinished(m_trainingSetOptimization->getPopulation()) &&
            m_algorithmConfig.m_stopFeatureSet->isFinished(m_featureSetOptimization->getPopulation());
}
} // namespace genetic
