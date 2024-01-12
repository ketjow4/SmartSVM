

#include "libPlatform/EnumStringConversions.h"
#include "libRandom/RandomNumberGeneratorFactory.h"
#include "SvmGenerationPopulationFactory.h"
#include "SvmUtils.h"
#include "SvmComponentsExceptions.h"
#include "SvmKernelRandomGeneration.h"
#include "SvmKernelGridGeneration.h"

namespace svmComponents
{
const std::unordered_map<std::string, SvmKernelGeneration> SvmGenerationPopulationFactory::m_translationsGeneration =
{
    {"RandomInRange", SvmKernelGeneration::RandomInRange},
    {"Grid", SvmKernelGeneration::Grid}
};

SvmGenerationPopulationFactory::IPopulationGenerationSvmKernel SvmGenerationPopulationFactory::create(const platform::Subtree& config,
                                                                                                      phd::svm::KernelTypes kernelType)
{
    auto name = config.getValue<std::string>("Generation.Name");

    switch (platform::stringToEnum(name, m_translationsGeneration))
    {
    case SvmKernelGeneration::RandomInRange:
    {
        auto initialKernelParametersRange = svmUtils::getRange("Generation.RandomInRange", config);
        auto isRegression = config.getValue<bool>("Generation.isRegression");
        return std::make_unique<SvmKernelRandomGeneration>(initialKernelParametersRange,
                                                           kernelType,
                                                           std::move(random::RandomNumberGeneratorFactory::create(config)),
                                                           isRegression);
    }
    case SvmKernelGeneration::Grid:
    {
        auto initialKernelParametersRange = svmUtils::getRange("Generation.Grid", config);
        auto isRegression = config.getValue<bool>("Generation.isRegression");
        return std::make_unique<SvmKernelGridGeneration>(initialKernelParametersRange,
                                                         kernelType,
                                                         std::move(random::RandomNumberGeneratorFactory::create(config)),
                                                         isRegression);
    }
    default:
        throw UnknownEnumType(name, typeid(SvmKernelGeneration).name());
    }
}
} // namespace svmComponents
