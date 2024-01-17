
#pragma once

#include "libGeneticSvm/ISvmAlgorithm.h"
#include "libGeneticSvm/CombinedAlgorithmsConfig.h"

namespace genetic
{
// @wdudzik Alga is short for Alternating genetic algorithm
class AlgaWorkflow : public ISvmAlgorithm
{
public:
    AlgaWorkflow(const SvmWokrflowConfiguration& config,
                 GeneticAlternatingEvolutionConfiguration algorithmConfig);

    std::shared_ptr<phd::svm::ISvm> run() override;

private:
    bool isFinished() const;

    template<class chromosome>
    void log(IGeneticWorkflow<chromosome>& workflow);

    template <class chromosome>
    void clearlog(IGeneticWorkflow<chromosome>& workflow);

    TrainingSetOptimizationWorkflow m_trainingSetOptimization;
    KernelOptimizationWorkflow m_kernelOptimization;
    std::filesystem::path m_resultFilePath;
    GeneticAlternatingEvolutionConfiguration m_algorithmConfig;

    //logger::LogFrontend m_logger;
};

template <class chromosome>
void AlgaWorkflow::log(IGeneticWorkflow<chromosome>& workflow)
{
    workflow.getResultLogger().logToFile(m_resultFilePath);
    workflow.getResultLogger().logToConsole();
    workflow.getResultLogger().clearLog();
}

template <class chromosome>
void AlgaWorkflow::clearlog(IGeneticWorkflow<chromosome>& workflow)
{
    workflow.getResultLogger().clearLog();
}
} // namespace genetic
