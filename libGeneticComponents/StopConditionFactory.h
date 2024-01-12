

#pragma once

#include <memory>
#include <unordered_map>
#include "libPlatform/Subtree.h"
#include "libPlatform/EnumStringConversions.h"
#include "LibGeneticComponents/IStopCondition.h"
#include "LibGeneticComponents/MeanFitnessStopCondition.h"
#include "LibGeneticComponents/BestFitnessStopCondition.h"

namespace geneticComponents
{
enum class StopCondition
{
    Unknown,
    MeanFitness,
    BestFitness,
};

class StopConditionFactory
{
public:
    template <typename chromosome>
    static std::unique_ptr<IStopCondition<chromosome>> create(const platform::Subtree& config);

private:
    const static std::unordered_map<std::string, StopCondition> m_stopConditionTranslations;
};

template <typename chromosome>
std::unique_ptr<IStopCondition<chromosome>> StopConditionFactory::create(const platform::Subtree& config)
{
    const auto name = config.getValue<std::string>("StopCondition.Name");

    switch (platform::stringToEnum(name, m_stopConditionTranslations))
    {
    case StopCondition::MeanFitness:
    {
        return std::make_unique<MeanFitnessStopCondition<chromosome>>(config.getValue<double>("StopCondition." + name + ".Epsilon"));
    }
    case StopCondition::BestFitness:
    {
        return std::make_unique<BestFitnessStopCondition<chromosome>>(config.getValue<double>("StopCondition." + name + ".Epsilon"));
    }
    default:
        throw UnknownEnumType(name, typeid(StopCondition).name());
    }
}
} // namespace geneticComponents
