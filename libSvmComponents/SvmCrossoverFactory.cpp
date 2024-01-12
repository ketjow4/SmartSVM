

#include "libPlatform/EnumStringConversions.h"
#include "libGeneticComponents/CrossoverSelectionFactory.h"
#include "libRandom/RandomNumberGeneratorFactory.h"
#include "SvmCrossoverFactory.h"
#include "SvmUtils.h"
#include "HeuristicCrossover.h"

namespace svmComponents
{
const std::unordered_map<std::string, SvmKernelCrossover> SvmKernelCrossoverFactory::m_translations =
{
    {"Heuristic", SvmKernelCrossover::Heuristic}
};

std::unique_ptr<geneticComponents::BaseCrossoverOperator<SvmKernelChromosome>> SvmKernelCrossoverFactory::create(const platform::Subtree& config)
{
    auto name = config.getValue<std::string>("Crossover.Name");

    switch (platform::stringToEnum(name, m_translations))
    {
    case SvmKernelCrossover::Heuristic:
    {
        auto alphaRange = svmUtils::getRange("Crossover.Heuristic.AlphaRange", config);
        return std::make_unique<HeuristicCrossover>(
            std::move(random::RandomNumberGeneratorFactory::create(config)),
            alphaRange);
    }
    default:
        throw UnknownEnumType(name, typeid(SvmKernelCrossover).name());
    }
}
} // namespace svmComponents
