#include "MemeticFeaturesCompensationGeneration.h"

namespace svmComponents
{
MemeticFeaturesCompensationGeneration::MemeticFeaturesCompensationGeneration(std::unique_ptr<my_random::IRandomNumberGenerator> randomNumberGenerator)
    : m_rngEngine(std::move(randomNumberGenerator))
{
    if (m_rngEngine == nullptr)
    {
        throw RandomNumberGeneratorNullPointer();
    }
}

std::vector<unsigned> MemeticFeaturesCompensationGeneration::generate(
    const std::vector<geneticComponents::Parents<SvmFeatureSetMemeticChromosome>>& parents, unsigned numberOfFeatures) const
{
    std::vector<unsigned int> compensationInfo(parents.size());

    std::transform(parents.begin(), parents.end(), compensationInfo.begin(), [&, this](const auto& parentsPair)
    {
        auto parentsMax = std::max(parentsPair.first.getDataset().size(), parentsPair.second.getDataset().size());

        std::uniform_int_distribution<int> sizeOfChild(static_cast<int>(parentsMax), static_cast<int>(numberOfFeatures));
        auto newSize = m_rngEngine->getRandom(sizeOfChild);

        return static_cast<unsigned int>(newSize - parentsMax);
    });
    return compensationInfo;
}
} // namespace svmComponents
