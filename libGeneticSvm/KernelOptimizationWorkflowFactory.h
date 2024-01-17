
#pragma once

#include <unordered_map>
#include "libPlatform/Subtree.h"
#include "libGeneticSvm/CombinedAlgorithmsConfig.h"

namespace genetic
{
enum class KernelOptimizationAlgorithms
{
    Unknown,
    GeneticKernelEvolution,
};

class KernelOptimizationWorkflowFactory
{
public:
    static KernelOptimizationWorkflow create(const platform::Subtree& config,
                                             IDatasetLoader& loadingWorkflow,
                                             const std::string& node);
private:
    const static std::unordered_map<std::string, KernelOptimizationAlgorithms> m_translations;
};
} // namespace genetic
