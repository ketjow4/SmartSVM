
#pragma once

#include <memory>
#include "libRandom/IRandomNumberGenerator.h"
#include "LibGeneticComponents/OnePointCrossover.h"
#include "LibGeneticComponents/GeneticUtils.h"

namespace geneticComponents
{
template <class binaryChromosome>
class FeaturesSelectionOnePointCrossover : public OnePointCrossover<binaryChromosome>
{
public:
    explicit FeaturesSelectionOnePointCrossover(std::unique_ptr<random::IRandomNumberGenerator> rngEngine);

    binaryChromosome crossoverChromosomes(const binaryChromosome& parentA, const binaryChromosome& parentB) override;

private:
    //logger::LogFrontend m_logger;
};

template <class binaryChromosome>
FeaturesSelectionOnePointCrossover<binaryChromosome>::FeaturesSelectionOnePointCrossover(std::unique_ptr<random::IRandomNumberGenerator> rngEngine)
    : OnePointCrossover(std::move(rngEngine))
{
}

template <class binaryChromosome>
binaryChromosome FeaturesSelectionOnePointCrossover<binaryChromosome>::crossoverChromosomes(const binaryChromosome& parentA, const binaryChromosome& parentB)
{
    const auto child = OnePointCrossover<binaryChromosome>::crossoverChromosomes(parentA, parentB);
    auto& genes = child.getGenes();
    if (geneticUtils::allZero(genes))
    {
        //m_logger.LOG(logger::LogLevel::Warning, "All bits of chromosome were 0 after crossover");
        return std::max(parentA, parentB);
    }
    return child;
}
} // namespace geneticComponents
