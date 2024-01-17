
#pragma once

#include "libGeneticSvm/ISvmAlgorithm.h"
#include "libGeneticSvm/CombinedAlgorithmsConfig.h"

namespace genetic
{
// @wdudzik This is combined workflow of optimization of svm hyperparameters, traning set and feature set
// KTF stands for kernel -> traning set -> feature set which indicates order how these algorithms have to run 
class KTFGeneticWorkflow : public ISvmAlgorithm
{
public:
    KTFGeneticWorkflow(const SvmWokrflowConfiguration& config,
                           KTFGeneticEvolutionConfiguration algorithmConfig);

    std::shared_ptr<phd::svm::ISvm> run() override;

private:
    bool isFinished() const;

    template <class chromosome>
    void log(IGeneticWorkflow<chromosome>& workflow);

    template <class chromosome>
    void clearlog(IGeneticWorkflow<chromosome>& workflow);

    TrainingSetOptimizationWorkflow m_trainingSetOptimization;
    KernelOptimizationWorkflow m_kernelOptimization;
    FeatureSetOptimizationWorkflow m_featureSetOptimization;
    std::filesystem::path m_resultFilePath;
    KTFGeneticEvolutionConfiguration m_algorithmConfig;

    //logger::LogFrontend m_logger;
};

template <class chromosome>
void KTFGeneticWorkflow::log(IGeneticWorkflow<chromosome>& workflow)
{
    workflow.getResultLogger().logToFile(m_resultFilePath);
    //workflow.getResultLogger().clearLog();
}


template <class chromosome>
void KTFGeneticWorkflow::clearlog(IGeneticWorkflow<chromosome>& workflow)
{
    workflow.getResultLogger().clearLog();
}
} // namespace genetic
