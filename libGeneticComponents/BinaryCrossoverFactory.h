
#pragma once

#include <memory>
#include <unordered_map>
#include "libRandom/RandomNumberGeneratorFactory.h"
#include "libPlatform/Subtree.h"
#include "libPlatform/EnumStringConversions.h"
#include "LibGeneticComponents/GeneticExceptions.h"
#include "LibGeneticComponents/BaseCrossoverOperator.h"
#include "LibGeneticComponents/OnePointCrossover.h"
#include "LibGeneticComponents/FeaturesSelectionOnePointCrossover.h"

namespace geneticComponents
{
enum class BinaryCrossover
{
    Unknown,
    OnePoint,
    FeaturesSelectionOnePoint
};

class BinaryCrossoverFactory
{
public:
    template <typename chromosome>
    static std::unique_ptr<BaseCrossoverOperator<chromosome>> create(const platform::Subtree& config);

private:
    const static std::unordered_map<std::string, BinaryCrossover> m_crossoverTranslations;
};

template <typename chromosome>
std::unique_ptr<BaseCrossoverOperator<chromosome>> BinaryCrossoverFactory::create(const platform::Subtree& config)
{
    static_assert(std::is_base_of<BinaryChromosome, chromosome>::value, "Cannot create element for class not derived from BinaryChromosome");

    const auto name = config.getValue<std::string>("Crossover.Name");

    switch (platform::stringToEnum(name, m_crossoverTranslations))
    {
    case BinaryCrossover::OnePoint:
    {
        return std::make_unique<OnePointCrossover<chromosome>>(
            std::move(my_random::RandomNumberGeneratorFactory::create(config)));
    }
    case BinaryCrossover::FeaturesSelectionOnePoint:
    {
        return std::make_unique<FeaturesSelectionOnePointCrossover<chromosome>>(
            std::move(my_random::RandomNumberGeneratorFactory::create(config)));
    }
    default:
        throw UnknownEnumType(name, typeid(BinaryCrossover).name());
    }
}
} // namespace geneticComponents
