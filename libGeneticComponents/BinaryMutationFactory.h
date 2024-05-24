

#pragma once

#include <memory>
#include <unordered_map>
#include "libRandom/RandomNumberGeneratorFactory.h"
#include "libPlatform/Subtree.h"
#include "libPlatform/EnumStringConversions.h"
#include "LibGeneticComponents/GeneticExceptions.h"
#include "LibGeneticComponents/IMutationOperator.h"
#include "LibGeneticComponents/BitFlipMutation.h"
#include "LibGeneticComponents/FeaturesSelectionBitFlipMutation.h"

namespace geneticComponents
{
enum class BinaryMutation
{
    Unknown,
    BitFlip,
    FeaturesSelectionBitFlip
};

class BinaryMutationFactory
{
public:
    template <typename chromosome>
    static std::unique_ptr<IMutationOperator<chromosome>> create(const platform::Subtree& config);

private:
    const static std::unordered_map<std::string, BinaryMutation> m_mutationTranslations;
};

template <typename chromosome>
std::unique_ptr<IMutationOperator<chromosome>> BinaryMutationFactory::create(const platform::Subtree& config)
{
    static_assert(std::is_base_of<BinaryChromosome, chromosome>::value, "Cannot create element for class not derived from BinaryChromosome");

    auto name = config.getValue<std::string>("Mutation.Name");

    switch (platform::stringToEnum(name, m_mutationTranslations))
    {
    case BinaryMutation::BitFlip:
    {
        const auto bitFlipProbability = config.getValue<double>("Mutation.BitFlipProbability");
        return std::make_unique<BitFlipMutation<chromosome>>(
            platform::Percent(bitFlipProbability),
            std::move(my_random::RandomNumberGeneratorFactory::create(config)));
    }
    case BinaryMutation::FeaturesSelectionBitFlip:
    {
        const auto bitFlipProbability = config.getValue<double>("Mutation.BitFlipProbability");
        return std::make_unique<FeaturesSelectionBitFlipMutation<chromosome>>(
            platform::Percent(bitFlipProbability),
            std::move(my_random::RandomNumberGeneratorFactory::create(config)));
    }
    default:
        throw UnknownEnumType(name, typeid(BinaryMutation).name());
    }
}
} // namespace geneticComponents
