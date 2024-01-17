
#pragma once

#include <unordered_map>
#include "libPlatform/Subtree.h"
#include "libGeneticSvm/CombinedAlgorithmsConfig.h"

namespace genetic
{
enum class FeatureSetOptimizationAlgorithms
{
    Unknown,
    FeatureSetSelection,
    MemeticFeatureSetSelection,
};

class FeatureSetOptimizationWorkflowFactory
{
public:
    static FeatureSetOptimizationWorkflow create(const platform::Subtree& config,
                                                 IDatasetLoader& loadingWorkflow,
                                                 const std::string& node);
private:
    const static std::unordered_map<std::string, FeatureSetOptimizationAlgorithms> m_translations;
};
} // namespace genetic
