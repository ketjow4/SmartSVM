

#pragma once

#include <memory>
#include <unordered_map>
#include "libRandom/RandomNumberGeneratorFactory.h"
#include "libPlatform/Subtree.h"
#include "libPlatform/EnumStringConversions.h"
#include "LibGeneticComponents/HighLowFitSelection.h"
#include "LibGeneticComponents/GeneticExceptions.h"
#include "LibGeneticComponents/LocalGlobalAdaptationSelection.h"
#include "libPlatform/Percent.h"

namespace geneticComponents
{
enum class CrossoverSelection
{
    Unknown,
    HighLowFit,
    LocalGlobalSelection
};

class CrossoverSelectionFactory
{
public:
    template <typename chromosome>
    static std::unique_ptr<ICrossoverSelection<chromosome>> create(const platform::Subtree& config);

private:
    const static std::unordered_map<std::string, CrossoverSelection> m_crossoverSelectionTranslations;
};

template <typename chromosome>
std::unique_ptr<ICrossoverSelection<chromosome>> CrossoverSelectionFactory::create(const platform::Subtree& config)
{
    auto name = config.getValue<std::string>("CrossoverSelection.Name");

    switch (platform::stringToEnum(name, m_crossoverSelectionTranslations))
    {
    case CrossoverSelection::HighLowFit:
    {
        auto highLowCoefficient = config.getValue<double>("CrossoverSelection.HighLowFit.HighLowCoefficient");
        return std::make_unique<HighLowFitSelection<chromosome>>(platform::Percent(highLowCoefficient),
                                                                 std::move(random::RandomNumberGeneratorFactory::create(config)));
    }
    case CrossoverSelection::LocalGlobalSelection:
    {
        auto highLowCoefficient = config.getValue<double>("CrossoverSelection.LocalGlobalSelection.HighLowCoefficient");
        auto isLocalMode = config.getValue<bool>("CrossoverSelection.LocalGlobalSelection.IsLocalMode");
        return std::make_unique<LocalGlobalAdaptationSelection<chromosome>>(isLocalMode,
                                                                            platform::Percent(highLowCoefficient),
                                                                            std::move(random::RandomNumberGeneratorFactory::create(config)));
    }
    default:
        throw UnknownEnumType(name, typeid(CrossoverSelection).name());
    }
}
} // namespace geneticComponents
