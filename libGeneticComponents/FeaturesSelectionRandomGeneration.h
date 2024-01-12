
#pragma once

#include <memory>
#include "libRandom/IRandomNumberGenerator.h"
#include "libPlatform/Percent.h"
#include "LibGeneticComponents/RandomGeneration.h"
#include "LibGeneticComponents/GeneticUtils.h"

namespace geneticComponents
{
template <class binaryChromosome>
class FeaturesSelectionRandomGeneration : public RandomGeneration<binaryChromosome>
{
public:
    FeaturesSelectionRandomGeneration(unsigned int chromosomeSize,
                             platform::Percent percentageOfFill,
                             std::unique_ptr<random::IRandomNumberGenerator> randomNumberGenerator);
private:
    binaryChromosome createIndividual() override;

    //logger::LogFrontend m_logger;
};

template <class binaryChromosome>
FeaturesSelectionRandomGeneration<binaryChromosome>::FeaturesSelectionRandomGeneration(unsigned chromosomeSize,
                                                                     platform::Percent percentageOfFill,
                                                                     std::unique_ptr<random::IRandomNumberGenerator> randomNumberGenerator)
    : RandomGeneration(chromosomeSize,percentageOfFill,std::move(randomNumberGenerator))
{
}

template <class binaryChromosome>
binaryChromosome FeaturesSelectionRandomGeneration<binaryChromosome>::createIndividual()
{
    auto individual = RandomGeneration<binaryChromosome>::createIndividual();
    while (geneticUtils::allZero(individual.getGenes()))
    {
        //m_logger.LOG(logger::LogLevel::Warning, "All bits of chromosome was 0 when generating individual");
        individual = RandomGeneration<binaryChromosome>::createIndividual();
    }
    return individual;
}
} // namespace geneticComponents
