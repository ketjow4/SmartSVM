
#pragma once

#include "libGeneticSvm/ISvmAlgorithm.h"
#include "libGeneticSvm/CombinedAlgorithmsConfig.h"

namespace genetic
{
class TFGeneticWorkflow : public ISvmAlgorithm
{
public:
    TFGeneticWorkflow(const SvmWokrflowConfiguration& config,
                      TFGeneticEvolutionConfiguration algorithmConfig);

    std::shared_ptr<phd::svm::ISvm> run() override;

private:
    bool isFinished() const;

    template <class chromosome>
    void log(IGeneticWorkflow<chromosome>& workflow);

	template <class chromosome>
	void clearlog(IGeneticWorkflow<chromosome>& workflow);

    TrainingSetOptimizationWorkflow m_trainingSetOptimization;
    FeatureSetOptimizationWorkflow m_featureSetOptimization;
    std::filesystem::path m_resultFilePath;
    TFGeneticEvolutionConfiguration m_algorithmConfig;

    //logger::LogFrontend m_logger;
};

template <class chromosome>
void TFGeneticWorkflow::log(IGeneticWorkflow<chromosome>& workflow)
{
    workflow.getResultLogger().logToFile(m_resultFilePath);
    //workflow.getResultLogger().clearLog();
}

template <class chromosome>
void TFGeneticWorkflow::clearlog(IGeneticWorkflow<chromosome>& workflow)
{
	workflow.getResultLogger().clearLog();
}
} // namespace genetic
