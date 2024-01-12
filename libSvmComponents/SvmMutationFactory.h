

#pragma once

#include <memory>
#include <unordered_map>
#include "libPlatform/Subtree.h"
#include "libGeneticComponents/IMutationOperator.h"
#include "libSvmComponents/SvmKernelChromosome.h"

namespace svmComponents
{
enum class SvmKernelMutation
{
    Unknown,
    ParameterMutation
};

class SvmMutationFactory
{
public:
    static std::unique_ptr<geneticComponents::IMutationOperator<SvmKernelChromosome>> create(const platform::Subtree& config);

private:
    const static std::unordered_map<std::string, SvmKernelMutation> m_translationsSvmKernelMutation;
};
} // namespace svmComponents