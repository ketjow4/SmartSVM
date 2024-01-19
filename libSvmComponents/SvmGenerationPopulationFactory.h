

#pragma once

#include <memory>
#include <unordered_map>
//#include <opencv2/ml.hpp>
#include "libPlatform/Subtree.h"
#include "libGeneticComponents/IPopulationGeneration.h"
#include "libSvmComponents/SvmKernelChromosome.h"

namespace svmComponents
{
enum class SvmKernelGeneration
{
    Unknown,
    RandomInRange,
    Grid
};

class SvmGenerationPopulationFactory
{
    using IPopulationGenerationSvmKernel = std::unique_ptr<geneticComponents::IPopulationGeneration<SvmKernelChromosome>>;
public:
    static IPopulationGenerationSvmKernel create(const platform::Subtree& config, phd::svm::KernelTypes kernelType);
private:
    const static std::unordered_map<std::string, SvmKernelGeneration> m_translationsGeneration;
};
} // namespace svmComponents
