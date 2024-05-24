#pragma once

#include <memory>
#include "libRandom/IRandomNumberGenerator.h"
#include "libGeneticComponents/BaseCrossoverOperator.h"
#include "SvmFeatureSetMemeticChromosome.h"

namespace svmComponents
{
class MemeticFeatureCrossover : public geneticComponents::BaseCrossoverOperator<SvmFeatureSetMemeticChromosome>
{
public:
    using chromosomeType = SvmFeatureSetMemeticChromosome;

    explicit MemeticFeatureCrossover(std::unique_ptr<my_random::IRandomNumberGenerator> rngEngine,
                                     unsigned int numberOfClasses);

    chromosomeType crossoverChromosomes(const chromosomeType& parentA, const chromosomeType& parentB) override;

private:
    void tryToInsert(const std::vector<Feature>::const_iterator& parentAIt,
                     const std::vector<Feature>::const_iterator& parentBIt,
                     std::unordered_set<uint64_t>& childSet,
                     std::vector<Feature>& child) const;
    std::vector<Feature> crossoverInternals(const chromosomeType& parentA,
                                            const chromosomeType& parentB,
                                            unsigned int datasetSize);

    std::unique_ptr<my_random::IRandomNumberGenerator> m_rngEngine;
    unsigned int m_numberOfClasses;
};

inline MemeticFeatureCrossover::MemeticFeatureCrossover(std::unique_ptr<my_random::IRandomNumberGenerator> rngEngine, unsigned numberOfClasses)
    : m_rngEngine(std::move(rngEngine))
    , m_numberOfClasses(numberOfClasses)
{
    if (m_rngEngine == nullptr)
    {
        throw RandomNumberGeneratorNullPointer();
    }
}

void MemeticFeatureCrossover::tryToInsert(const std::vector<Feature>::const_iterator& parentAIt,
                                          const std::vector<Feature>::const_iterator& parentBIt,
                                          std::unordered_set<uint64_t>& childSet,
                                          std::vector<Feature>& child) const
{
    if (childSet.insert(parentAIt->id).second)
    {
        child.emplace_back(*parentAIt);
    }
    else if (childSet.insert(parentBIt->id).second)
    {
        child.emplace_back(*parentBIt);
    }
}

std::vector<Feature> MemeticFeatureCrossover::crossoverInternals(const chromosomeType& parentA,
                                                                 const chromosomeType& parentB,
                                                                 unsigned int datasetSize)
{
    std::vector<Feature> child;
    std::unordered_set<uint64_t> childSet;
    child.reserve(datasetSize);
    childSet.reserve(datasetSize);

    auto parentAIt = parentA.getDataset().begin();
    auto parentBIt = parentB.getDataset().begin();

    while (true)
    {
        std::uniform_real_distribution<double> randomParent(platform::Percent::m_minPercent, platform::Percent::m_maxPercent);
        constexpr auto halfRange = platform::Percent::m_maxPercent / 2;
        auto parentChoose = m_rngEngine->getRandom(randomParent);
        if (parentChoose > halfRange)
        {
            tryToInsert(parentAIt, parentBIt, childSet, child);
        }
        else
        {
            tryToInsert(parentBIt, parentAIt, childSet, child);
        }

        if (child.size() == datasetSize)
        {
            break;
        }

        if (++parentBIt == parentB.getDataset().end())
        {
            parentBIt = parentB.getDataset().begin();
        }
        if (++parentAIt == parentA.getDataset().end())
        {
            parentAIt = parentA.getDataset().begin();
        }
    }
    return child;
}

inline MemeticFeatureCrossover::chromosomeType MemeticFeatureCrossover::crossoverChromosomes(const chromosomeType& parentA, const chromosomeType& parentB)
{
    auto sizeOfChild = std::max(parentA.getDataset().size(), parentB.getDataset().size());
    auto datasetSize = sizeOfChild;

    auto child = crossoverInternals(parentA, parentB, static_cast<unsigned int>(datasetSize));

    return SvmFeatureSetMemeticChromosome(std::move(child));
}
} // namespace svmComponents
