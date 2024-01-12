#pragma once

#include <memory>
#include <unordered_map>
#include "libPlatform/Subtree.h"
#include "libSvmComponents/GeneticAlgorithmsConfigs.h"

namespace svmComponents
{
enum class SvmMemeticFeatureSetGeneration
{
    Unknown,
    Random,
    MutualInfo,
};

class SvmMemeticFeatureSetPopulationFactory
{
public:
    static PopulationGeneration<SvmFeatureSetMemeticChromosome> create(const platform::Subtree& config,
                                                                       const dataset::Dataset<std::vector<float>, float>& trainingSet,
                                                                       std::string trainingDataPath,
                                                                       std::string outputPath);

private:
    const static std::unordered_map<std::string, SvmMemeticFeatureSetGeneration> m_translationsGeneration;
};
} // namespace svmComponents
