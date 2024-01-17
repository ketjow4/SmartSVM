
#pragma once

#include <unordered_map>
#include "libPlatform/Subtree.h"
#include "libGeneticSvm/CombinedAlgorithmsConfig.h"

namespace genetic
{
enum class TrainingSetOptimizationAlgorithms
{
    Unknown,
    GaSvm,
    Memetic,
};

class TrainingSetOptimizationWorkflowFactory
{
public:
    static TrainingSetOptimizationWorkflow create(const platform::Subtree& config,
                                                  IDatasetLoader& loadingWorkflow,
                                                  const std::string& node);
private:
    const static std::unordered_map<std::string, TrainingSetOptimizationAlgorithms> m_translations;
};
} // namespace genetic
