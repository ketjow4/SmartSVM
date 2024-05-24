

#pragma once

#include <memory>
#include <unordered_map>
#include "libRandom/RandomNumberGeneratorFactory.h"
#include "libPlatform/Subtree.h"
#include "libPlatform/EnumStringConversions.h"
#include "LibGeneticComponents/GeneticExceptions.h"
#include "LibGeneticComponents/IPopulationGeneration.h"
#include "LibGeneticComponents/RandomGeneration.h"
#include "LibGeneticComponents/FeaturesSelectionRandomGeneration.h"

namespace geneticComponents
{
enum class BinaryGeneration
{
    Unknown,
    Random,
    FeaturesSelectionRandom,
};

class BinaryGenerationFactory
{
public:
    template <typename chromosome>
    static std::unique_ptr<IPopulationGeneration<chromosome>> create(const platform::Subtree& config,
                                                                     const unsigned int chromosomeSize);

private:
    const static std::unordered_map<std::string, BinaryGeneration> m_generationTranslations;
};

template <typename chromosome>
std::unique_ptr<IPopulationGeneration<chromosome>> BinaryGenerationFactory::create(const platform::Subtree& config,
                                                                                   const unsigned int chromosomeSize)
{
    static_assert(std::is_base_of<BinaryChromosome, chromosome>::value, "Cannot create element for class not derived from BinaryChromosome");

    const auto name = config.getValue<std::string>("Generation.Name");

    switch (platform::stringToEnum(name, m_generationTranslations))
    {
    case BinaryGeneration::Random:
    {
        const auto percentageOfFill = config.getValue<double>("Generation.PercentageOfFill");
        return std::make_unique<RandomGeneration<chromosome>>(
            chromosomeSize,
            platform::Percent(percentageOfFill),
            std::move(my_random::RandomNumberGeneratorFactory::create(config)));
    }
    case BinaryGeneration::FeaturesSelectionRandom:
    {
        const auto percentageOfFill = config.getValue<double>("Generation.PercentageOfFill");
        return std::make_unique<FeaturesSelectionRandomGeneration<chromosome>>(
            chromosomeSize,
            platform::Percent(percentageOfFill),
            std::move(my_random::RandomNumberGeneratorFactory::create(config)));
    }
    default:
        throw UnknownEnumType(name, typeid(BinaryGeneration).name());
    }
}
} // namespace geneticComponents
