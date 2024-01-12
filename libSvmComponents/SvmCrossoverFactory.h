

#pragma once

#include <memory>
#include <unordered_map>
#include "libPlatform/Subtree.h"
#include "libGeneticComponents/BaseCrossoverOperator.h"
#include "libSvmComponents/SvmKernelChromosome.h"

namespace svmComponents
{
enum class SvmKernelCrossover
{
    Unknown,
    Heuristic
};

class SvmKernelCrossoverFactory
{
public:
    static std::unique_ptr<geneticComponents::BaseCrossoverOperator<SvmKernelChromosome>> create(const platform::Subtree& config);
private:
    const static std::unordered_map<std::string, SvmKernelCrossover> m_translations;
};
} // namespace svmComponents