
#pragma once

#include "libGeneticSvm/ISvmAlgorithm.h"
#include "libGeneticSvm/CombinedAlgorithmsConfig.h"

namespace genetic
{
class FTGeneticWorkflow : public ISvmAlgorithm
{
public:
    FTGeneticWorkflow(const SvmWokrflowConfiguration& config,
                      TFGeneticEvolutionConfiguration algorithmConfig);

    std::shared_ptr<phd::svm::ISvm> run() override;

private:
    bool isFinished() const;

    template <class chromosome>
    void log(IGeneticWorkflow<chromosome>& workflow);

    TrainingSetOptimizationWorkflow m_trainingSetOptimization;
    FeatureSetOptimizationWorkflow m_featureSetOptimization;
    std::filesystem::path m_resultFilePath;
    TFGeneticEvolutionConfiguration m_algorithmConfig;

    //logger::LogFrontend m_logger;
};

template <class chromosome>
void FTGeneticWorkflow::log(IGeneticWorkflow<chromosome>& workflow)
{
    workflow.getResultLogger().logToFile(m_resultFilePath);
    workflow.getResultLogger().clearLog();
}
} // namespace genetic
