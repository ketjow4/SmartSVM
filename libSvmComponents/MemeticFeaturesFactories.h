#pragma once

#include <memory>
#include <unordered_map>
#include "libPlatform/Subtree.h"
#include "libSvmComponents/GeneticAlgorithmsConfigs.h"

namespace svmComponents
{
enum class SvmMemeticFeatureSetMutation
{
    Unknown,
    GaSvm
};

class SvmMemeticFeatureSetMutationFactory
{
public:
    static MutationOperator<SvmFeatureSetMemeticChromosome> create(const platform::Subtree& config,
                                                                   const dataset::Dataset<std::vector<float>, float>& trainingSet,
                                                                   const std::vector<unsigned int>& labelsCount);

private:
    const static std::unordered_map<std::string, enum class SvmMemeticFeatureSetMutation> m_translationsSvmKernelMutation;
};






enum class SvmMemeticFeatureSetCrossover
{
    Unknown,
    Memetic
};

class SvmMemeticFeatureSetCrossoverFactory
{
public:
    static CrossoverOperator<SvmFeatureSetMemeticChromosome> create(const platform::Subtree& config);

private:
    const static std::unordered_map<std::string, enum class SvmMemeticFeatureSetCrossover> m_translationsSvmKernelCrossover;
};

} // namespace svmComponents
